"""
Real-world LLM context pressure measurement.

Wraps HuggingFace transformers models to measure how much context
truncation affects output distributions. This is the practical
projection-free óthismos for LLMs.

Requires: pip install othismos[torch] transformers

Usage:
    >>> from othismos.llm import LLMPressureAnalyzer
    >>> analyzer = LLMPressureAnalyzer(model_name="gpt2")
    >>> result = analyzer.measure_context_pressure(
    ...     prompt="The meaning of life is",
    ...     full_context_tokens=100,
    ...     truncate_to=50,
    ... )
    >>> print(f"Context pressure: {result['pressure']:.4f}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from othismos.context_pressure import ContextPressureGauge, ContextPressureMeasurement


@dataclass
class LLMPressureResult:
    """Result of measuring context pressure on an LLM."""

    pressure: float
    method: str
    full_tokens: int
    truncated_tokens: int
    tokens_dropped: int
    full_next_token_prob: np.ndarray
    truncated_next_token_prob: np.ndarray
    top5_full: list[tuple[int, float]]
    top5_truncated: list[tuple[int, float]]
    metadata: dict = field(default_factory=dict)

    @property
    def is_high_pressure(self) -> bool:
        """True if dropped tokens significantly affect output."""
        return self.pressure > 0.5  # nats — tunable threshold

    def summary(self) -> str:
        lines = [
            f"Context Pressure: {self.pressure:.4f} {self.method}",
            f"Tokens: {self.full_tokens} → {self.truncated_tokens} ({self.tokens_dropped} dropped)",
            f"High pressure: {self.is_high_pressure}",
            "",
            "Top-5 next tokens (full context):",
        ]
        for token_id, prob in self.top5_full:
            lines.append(f"  token {token_id}: {prob:.4f}")
        lines.append("")
        lines.append("Top-5 next tokens (truncated context):")
        for token_id, prob in self.top5_truncated:
            lines.append(f"  token {token_id}: {prob:.4f}")
        return "\n".join(lines)


class LLMPressureAnalyzer:
    """Measure context-window pressure in HuggingFace LLMs.

    This is the practical projection-free óthismos: instead of measuring
    parameter-space constraint violations, measure how much the output
    distribution shifts when context is truncated.

    High pressure = the truncated tokens are structurally important.
    Low pressure = safe to truncate.

    Args:
        model_name: HuggingFace model name (e.g., "gpt2", "meta-llama/Llama-2-7b")
        device: "cpu", "cuda", or "auto"
        distance_metric: "kl" (symmetric KL), "l2", "cosine", "js" (Jensen-Shannon)
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        device: str = "auto",
        distance_metric: str = "js",
    ):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "LLMPressureAnalyzer requires torch and transformers. "
                "Install with: pip install othismos[torch] transformers"
            )

        self.model_name = model_name
        self.device = device
        self._torch = torch

        # Load model and tokenizer
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

        # Internal gauge
        self.gauge = ContextPressureGauge(distance_fn=self._get_distance_fn(distance_metric))
        self._distance_metric = distance_metric

    def _get_distance_fn(self, name: str):
        """Get distance function by name."""
        if name == "kl":
            return self._symmetric_kl
        elif name == "js":
            return self._jensen_shannon
        elif name == "l2":
            from othismos.context_pressure import l2_distance
            return l2_distance
        elif name == "cosine":
            from othismos.context_pressure import cosine_distance
            return cosine_distance
        else:
            raise ValueError(f"Unknown distance metric: {name}. Use kl, js, l2, or cosine.")

    @staticmethod
    def _symmetric_kl(p: np.ndarray, q: np.ndarray) -> float:
        """Symmetric KL divergence (Jeffrey's)."""
        eps = 1e-12
        p = np.asarray(p, dtype=np.float64) + eps
        q = np.asarray(q, dtype=np.float64) + eps
        p = p / p.sum()
        q = q / q.sum()
        return float(np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p)))

    @staticmethod
    def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon divergence (symmetric, bounded)."""
        eps = 1e-12
        p = np.asarray(p, dtype=np.float64) + eps
        q = np.asarray(q, dtype=np.float64) + eps
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))
        return float(0.5 * (kl_pm + kl_qm))

    def _get_next_token_probs(self, input_ids) -> np.ndarray:
        """Get next-token probability distribution."""
        torch = self._torch
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]  # last token's logits
            probs = torch.softmax(logits, dim=-1)
            return probs[0].cpu().numpy()

    def measure_context_pressure(
        self,
        prompt: str,
        truncate_to: int | None = None,
        drop_from: str = "left",  # "left" (beginning) or "right" (end)
        stride: int | None = None,
    ) -> LLMPressureResult:
        """Measure how much truncating context changes the next-token distribution.

        Args:
            prompt: Input text
            truncate_to: How many tokens to keep. If None, uses half.
            drop_from: "left" drops from the beginning (keep recent context),
                       "right" drops from the end (keep initial context)
            stride: If set, measure pressure at multiple truncation points
                    (truncate_to, truncate_to+stride, truncate_to+2*stride, ...)
                    and return the one with highest pressure.

        Returns:
            LLMPressureResult with pressure, probabilities, and top tokens.
        """
        torch = self._torch

        # Tokenize
        full_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        full_len = full_ids.shape[1]

        if truncate_to is None:
            truncate_to = full_len // 2

        if truncate_to >= full_len:
            raise ValueError(
                f"truncate_to ({truncate_to}) >= full length ({full_len}). "
                f"Nothing to truncate."
            )

        # Full context probability
        full_probs = self._get_next_token_probs(full_ids)

        # Truncated context
        if drop_from == "left":
            truncated_ids = full_ids[:, -truncate_to:]
        else:
            truncated_ids = full_ids[:, :truncate_to]

        truncated_probs = self._get_next_token_probs(truncated_ids)

        # Compute pressure
        pressure = self.gauge._distance_fn(full_probs, truncated_probs)

        # Record in gauge
        self.gauge.measure(
            full_output=full_probs,
            constrained_output=truncated_probs,
            context_tokens_dropped=full_len - truncate_to,
        )

        # Top-5 tokens
        top5_full_idx = np.argsort(full_probs)[::-1][:5]
        top5_full = [(int(i), float(full_probs[i])) for i in top5_full_idx]
        top5_trunc_idx = np.argsort(truncated_probs)[::-1][:5]
        top5_trunc = [(int(i), float(truncated_probs[i])) for i in top5_trunc_idx]

        return LLMPressureResult(
            pressure=pressure,
            method=self._distance_metric,
            full_tokens=full_len,
            truncated_tokens=truncate_to,
            tokens_dropped=full_len - truncate_to,
            full_next_token_prob=full_probs,
            truncated_next_token_prob=truncated_probs,
            top5_full=top5_full,
            top5_truncated=top5_trunc,
            metadata={
                "model": self.model_name,
                "drop_from": drop_from,
                "device": self.device,
            },
        )

    def find_safe_truncation_point(
        self,
        prompt: str,
        max_pressure: float = 0.1,
        min_tokens: int = 10,
    ) -> int:
        """Binary search for the largest truncation that stays under max_pressure.

        This answers: "how many tokens can I safely drop from this prompt?"

        Args:
            prompt: Input text
            max_pressure: Maximum acceptable pressure (JSD or KL)
            min_tokens: Minimum tokens to keep

        Returns:
            Number of tokens that can be kept while staying under max_pressure.
        """
        full_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        full_len = full_ids.shape[1]

        if full_len <= min_tokens:
            return full_len

        full_probs = self._get_next_token_probs(full_ids)

        lo, hi = min_tokens, full_len
        best = full_len

        while lo < hi:
            mid = (lo + hi) // 2
            truncated_ids = full_ids[:, -mid:]
            trunc_probs = self._get_next_token_probs(truncated_ids)
            pressure = self.gauge._distance_fn(full_probs, trunc_probs)

            if pressure <= max_pressure:
                best = mid
                hi = mid  # try to truncate more
            else:
                lo = mid + 1

        return best

    def pressure_profile(
        self,
        prompt: str,
        points: int = 10,
    ) -> list[tuple[int, float]]:
        """Measure pressure at multiple truncation points.

        Returns a list of (tokens_kept, pressure) pairs showing how pressure
        grows as you truncate more aggressively.

        Useful for visualization: plot tokens_kept vs pressure to see where
        the "critical" tokens are.
        """
        full_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        full_len = full_ids.shape[1]
        full_probs = self._get_next_token_probs(full_ids)

        results = []
        for i in range(points):
            ratio = (i + 1) / points
            keep = max(1, int(full_len * (1 - ratio)))
            if keep >= full_len:
                continue

            truncated_ids = full_ids[:, -keep:]
            trunc_probs = self._get_next_token_probs(truncated_ids)
            pressure = self.gauge._distance_fn(full_probs, trunc_probs)
            results.append((keep, pressure))

        return results
