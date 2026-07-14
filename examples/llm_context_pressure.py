"""
Example: Measure context-window pressure in a real model.

This example uses GPT-2 (small, runs on CPU) to demonstrate how óthismos
measures the cost of context truncation. The same approach works on any
HuggingFace causal LM.

Run: python3.11 examples/llm_context_pressure.py
"""

import sys
sys.path.insert(0, "src")


def demo_without_model():
    """Demonstrate the concept without requiring torch/transformers."""
    import numpy as np
    from othismos import ContextPressureGauge, cosine_distance

    print("=" * 60)
    print("Óthismos LLM Context Pressure — Concept Demo")
    print("=" * 60)
    print()

    # Simulate two probability distributions over a vocabulary
    # "Full context" model is confident: token 42 is likely
    full_probs = np.zeros(1000)
    full_probs[42] = 0.8
    full_probs[100] = 0.1
    full_probs[200] = 0.05
    full_probs[999] = 0.05

    # "Truncated context" model is less confident: token 42 probability drops
    # This means the truncated tokens WERE structurally important
    truncated_high_pressure = np.zeros(1000)
    truncated_high_pressure[42] = 0.3
    truncated_high_pressure[100] = 0.3
    truncated_high_pressure[200] = 0.2
    truncated_high_pressure[999] = 0.2

    # Another truncation where it doesn't matter much
    truncated_low_pressure = full_probs.copy()
    truncated_low_pressure[42] = 0.78
    truncated_low_pressure[100] = 0.12

    gauge = ContextPressureGauge()

    # Measure high pressure
    result = gauge.measure(
        full_output=full_probs,
        constrained_output=truncated_high_pressure,
        context_tokens_dropped=500,
    )
    print(f"High-pressure truncation (500 tokens dropped):")
    print(f"  Pressure: {result.pressure:.4f}")
    print(f"  → The dropped tokens significantly changed the output distribution")
    print(f"  → These tokens carry structural information")
    print()

    # Measure low pressure
    result2 = gauge.measure(
        full_output=full_probs,
        constrained_output=truncated_low_pressure,
        context_tokens_dropped=500,
    )
    print(f"Low-pressure truncation (500 tokens dropped):")
    print(f"  Pressure: {result2.pressure:.4f}")
    print(f"  → The dropped tokens barely changed the output distribution")
    print(f"  → These tokens are safe to truncate")
    print()

    # Show the pressure profile over multiple truncation levels
    print("Pressure vs. truncation level:")
    print("-" * 40)
    for drop_frac in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        trunc = full_probs.copy()
        # More aggressive truncation = more distribution shift
        shift = drop_frac * 0.5
        trunc[42] = max(0.01, full_probs[42] - shift)
        trunc[100] = min(0.99, full_probs[100] + shift * 0.5)
        trunc = trunc / trunc.sum()

        r = gauge.measure(full_probs, trunc, context_tokens_dropped=int(drop_frac * 1000))
        bar = "█" * int(r.pressure * 50)
        print(f"  {drop_frac*100:5.0f}% dropped: {r.pressure:.4f} {bar}")

    print()
    print("The pressure curve tells you WHERE the critical tokens are.")
    print("Sharp increase = you're hitting structurally important context.")
    print("Flat region = redundant context that can be safely truncated.")


def demo_with_model():
    """Full demo with a HuggingFace model (requires torch + transformers)."""
    try:
        from othismos.llm import LLMPressureAnalyzer
    except ImportError as e:
        print(f"\n(Full LLM demo requires torch + transformers. Skipping.)")
        print(f"Install with: pip install othismos[torch] transformers")
        return

    print("\n" + "=" * 60)
    print("Óthismos LLM Context Pressure — GPT-2 Demo")
    print("=" * 60)

    analyzer = LLMPressureAnalyzer(model_name="gpt2", device="cpu", distance_metric="js")

    prompts = [
        ("Short factual", "The capital of France is Paris. The capital of Germany is Berlin. The capital of Japan is"),
        ("Long with context", "Alice was beginning to get very tired of sitting by her sister on the bank, "
         "and of having nothing to do: once or twice she had peeped into the book her sister "
         "was reading, but it had no pictures or conversations in it, \"and what is the use of "
         "a book,\" thought Alice \"without pictures or conversations?\" So she was considering "
         "in her own mind, whether the pleasure of a warm summer day would feel like"),
        ("Code", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) +"),
    ]

    for label, prompt in prompts:
        print(f"\n--- {label} ({len(prompt)} chars) ---")

        result = analyzer.measure_context_pressure(
            prompt=prompt,
            truncate_to=len(prompt.split()) // 2,  # rough token estimate
            drop_from="left",
        )
        print(result.summary())

        # Find safe truncation point
        safe = analyzer.find_safe_truncation_point(prompt, max_pressure=0.01)
        total = len(analyzer.tokenizer.encode(prompt))
        print(f"\nSafe truncation: can drop to {safe} tokens (from {total})")
        print(f"Savings: {(1 - safe/total)*100:.0f}% context reduction at <0.01 pressure")


if __name__ == "__main__":
    demo_without_model()
    demo_with_model()
