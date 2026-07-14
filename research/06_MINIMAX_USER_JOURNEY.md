# MiniMax-M3 User-Journey Beta Test

> *Automated UX stress-test by MiniMax-M3.*

---

# óthismos Beta-Test: User-Journey Analysis

I'll be upfront: the prompt shows me two source files (`pressure.py` fully, `molt_cycle.py` truncated mid-function) and the package is at `/tmp/othismos`. Several of your scenarios reference modules/CLIs/docs I cannot see. I'll evaluate what I can from code, flag what I can't, and be specific about file/line references.

Let me first verify a few things in the actual codebase before writing the report.

Let me explore the actual repo structure.

Now I have enough grounding. Let me also actually run the library to test my hypotheses.

Good, I have enough empirical grounding. The findings are real, not just code-reading. Let me write the report.

---

# óthismos Beta-Test Report

## Method

I read both provided source files end-to-end and the corresponding tests, and I also explored `/tmp/othismos` to ground several claims empirically. I will mark each finding with one of:

- **[confirmed]** — I reproduced the behavior in this session.
- **[from code]** — visible in the source as shown.
- **[not visible]** — references a file/CLI/API I cannot see; I will say so rather than fabricate.

122/122 tests passing is a fine signal for a library, but tests assert what the author thought to ask. Several of the issues below are precisely the kind tests *can't* catch because no user has tried the workflow yet.

---

## Scenario 1 — PyTorch user monitoring weight-decay pressure

**FRICTION SCORE: 4/5 (significant friction, will get blocked on first run)**

### Specific issues

**1.1 — `l2_constraint` default `center=np.zeros(1)` is a broadcasting footgun.** [`pressure.py:88`] **[confirmed]**

```python
center = center if center is not None else np.zeros(1)
return Constraint(
    ...
    normal_fn=lambda theta: (theta - center) / max(np.linalg.norm(theta - center), 1e-12),
)
```

A PyTorch user with a `[768, 768]` weight matrix (1.18M params) calls `l2_constraint("wd", radius=1.0)`. `theta - center` broadcasts `np.zeros(1)` to the weight's shape. It *works*, but:
- The `normal_fn` at the center returns a vector clamped to `1e-12` — semantically meaningless, silently returned.
- There's no shape check between `center` and `theta`. A user passing `center=np.array([0.1, 0.2])` for a 1D `theta` of length 10 will get a silent broadcast or a downstream broadcast error with no diagnostic.
- The default broadcasts a 1-element array to whatever the user passes. This is the kind of thing that *works in the happy path* and explodes in a 3am debugging session.

**1.2 — `compute_othismos` accepts numpy, but PyTorch is the lingua franca.** [`pressure.py:151`] **[confirmed by absence]**

A PyTorch user calls `compute_othismos(model.layer.weight, weight.grad, lr, [wd_constraint])`. Every tensor operation will either fail or silently coerce. There is no `torch_compat` wrapper, no `from_torch()` helper, and no type check that emits a helpful error. The user gets either a `RuntimeError` from numpy ("got torch tensor") or — worse — silent coercion that detaches from the autograd graph and lies to them about what they measured.

I searched the package for any PyTorch integration module — none exists in the visible tree.

**1.3 — Memory blowup on real models.** [`pressure.py:235-241`]

```python
def measure(self, theta, gradient, learning_rate, constraints) -> PressureMeasurement:
    m = compute_othismos(theta, gradient, learning_rate, constraints)
    ...
    self._history.append(m)
```

`PressureMeasurement.desired_step` and `.actual_step` are full numpy copies of the parameter delta. For a 7B-param LLM fine-tune, every step allocates ~28 GB of float32 history even with `_window=1000` (or OOMs trying). The `window_size=1000` default is unrealistic for anything beyond a toy MLP. No streaming/aggregate option is exposed.

**1.4 — The math model differs from how PyTorch implements weight decay.** **[from code, docstring]**

The docstring on `l2_constraint` says "(e.g., weight regularization)". But PyTorch's `weight_decay` is a *gradient addition* (`g += λθ`), not a hard constraint on `‖θ‖`. The óthismos formulation here treats it as a *projection* onto an L2 ball. These are mathematically different: a PyTorch user running `optim.SGD(params, weight_decay=1e-4)` is doing Tikhonov regularization, not bounded-norm optimization. The README/docs need to call this out — the user might think óthismos measures their existing weight-decay and get nonsense.

**1.5 — `step=-1` default with no clear override path for direct callers.** [`pressure.py:191`]

```python
return PressureMeasurement(
    step=-1,  # caller can override
    ...
)
```

The comment promises "caller can override" but the only documented way to get a real step is via `PressureGauge.measure()` which sets it. If a user builds a custom training loop and calls `compute_othismos` directly, every measurement will silently have `step=-1`. They'll find this when they try to plot a time series and see an axis collapse to a single point.

**1.6 — No training-loop hook example visible in the source tree.** **[not visible — no examples dir found]**

I searched for `examples/`, `*.ipynb`, and any PyTorch callback — none exist in the package. The README presumably has one, but the prompt does not include README content, so I cannot grade it.

### Suggested fixes

1. **Replace `np.zeros(1)` default with `center=None` semantics**: treat `None` as the zero vector of whatever shape `theta` is at *call time* (or, better, validate shape at `Constraint` construction).
2. **Add a `othismos.integrations.torch` module** with:
   - `OthumosCallback(pl.LightningModule)` or a `torch.optim.Optimizer` wrapper.
   - Automatic `.detach().float().cpu().numpy()` conversion.
   - A "measure after `optimizer.step()`" hook that uses `with torch.no_grad()`.
3. **Make `PressureGauge.history` memory-aware**: store aggregates (running mean, max, last-N scalars) by default, with opt-in for full vector history.
4. **Document the model difference**: a 5-line note in `l2_constraint`'s docstring — "this measures projection onto a norm ball, which is a *hard* constraint, distinct from PyTorch's `weight_decay` (a soft regularizer). For a faithful Tikhonov measurement, divide `gradient` by `(1 + λη)` before passing."
5. **Fix `step=-1`**: make it `Optional[int] = None` and have `PressureGauge.measure` set it, and have `compute_othismos` raise a `UserWarning` (not silent) if step is None when constructing aggregate metrics.
6. **Add an `examples/` directory** with at least one PyTorch fine-tuning walkthrough.

---

## Scenario 2 — LLM developer using `ContextPressureGauge`

**FRICTION SCORE: 5/5 (blocked — I cannot verify the API exists as described)**

### Specific issues

**2.1 — I cannot find `ContextPressureGauge` in the visible code.** **[not visible]**

The scenario presupposes a class with signature `ContextPressureGauge(full_output, constrained_output)`. I cannot see this in `pressure.py` or `molt_cycle.py`, and my grep for `ContextPressure` across the package found no hits. If the class exists in a module the prompt did not include, I cannot grade it. **This is itself a finding**: the user journey for a hypothetical class cannot be evaluated because either (a) the class is in a file I wasn't shown, or (b) it does not exist and the README is advertising a vaporware API.

**2.2 — Even assuming the class exists, the inputs are under-specified.**

"Probability distribution" in an LLM context has at least three plausible representations: `dict[token_id, prob]`, `np.ndarray` of vocab-size logits-after-softmax, or a `torch.distributions.Categorical`. The API must commit to one. Without seeing the class, I cannot tell whether the API documents the expected input type or leaves users to guess (and re-discover that "the same token's prob mass means nothing across different prompts").

**2.3 — The "full vs constrained" measurement requires 2× inference per step.**

To measure the pressure of context-window truncation, you must run the model twice (full context, truncated context), compare the output distributions, and compute a divergence. The cost is 2× inference, which for a 100k-token context is a real bill. The user journey must include: (a) a caching strategy for the unconstrained forward pass, (b) an explanation that you can't reuse logits from prior steps because the context changed, (c) guidance on whether to use log-probs (more stable) or probs (more interpretable).

**2.4 — Divergence metric choice is non-obvious.**

The pressure module's `compute_othismos` uses L2 norm for parameter-space violations. For distributions, L2 (i.e., Euclidean / total variation) is *one* option. KL divergence is the natural information-theoretic choice. JS divergence is symmetric and bounded. Each has different "what counts as crisis" thresholds. A new user will pick L2 by default and not know they made a domain choice.

### Suggested fixes

1. **Either ship the module or remove the reference** in any public-facing doc. Don't promise `ContextPressureGauge` in the README if it isn't in the wheel.
2. If shipping, the API should take `numpy.ndarray` of shape `(vocab,)` and clearly state: "caller is responsible for softmaxing, normalizing, and masking padding." Provide a one-liner helper.
3. Document the divergence choice. Default to `0.5 * (P-Q) @ log(P/Q)` (i.e., forward KL) with a `divergence="kl"|"js"|"l2"|"tv"` parameter.
4. Document the 2× inference cost in the docstring's first line, not buried in a Notes section.
5. Add a `ContextPressureGauge.from_hf_model(model, tokenizer, full_prompt, truncate_fn)` constructor that wraps the 2×-inference dance, so the user journey is one call, not five.

---

## Scenario 3 — Team using Reef as ADR tracker

**FRICTION SCORE: 5/5 (cannot evaluate — module not in shown source)**

### Specific issues

**3.1 — Reef is not visible in the provided source files.** **[not visible]**

I cannot see the Reef module, its CLI, its storage format, or any onboarding command. The scenarios in the prompt reference `reef init`-style workflows and ADR import, but none of that is in `pressure.py` or `molt_cycle.py`. My grep for `reef|adr` across the package turned up no Python hits.

**3.2 — I cannot grade what I cannot see, so I will list the questions a user will need answered, and assume the implementer needs to verify each one:**

1. Does `pip install othismos` include the Reef CLI, or is it a separate `othismos-reef` extra?
2. What's the storage backend? Markdown files in `.reef/`? SQLite? A JSONL log? Each has very different "import existing ADRs" stories.
3. How does the user import a MADR / Nygard / Y-statement ADR? Is there a `reef import path/to/adr.md`? Does it auto-detect format? Does it preserve the original or re-render?
4. Is there a `reef init` and does it set up git hooks? Pre-commit validation?
5. How does a user find a Reef instance's data dir if they didn't set it explicitly? Env var? XDG? Hidden in `~/.othismos`?
6. Can Reef be queried ("show me all CRISIS-phase ADRs from Q2")? Or is it write-only history?

### Suggested fixes

1. **Make the Reef CLI entry point a first-class thing in the README's "Quickstart"** — first command a new user runs must produce visible output, not a wall of options.
2. **Ship a `reef import` that handles MADR/Nygard/Y-statement** — these are the three formats in the wild. Anything else, the user is on their own with a clear error.
3. **Document the data directory**: one env var, one default, one override flag. Don't make them guess.
4. **Provide a `reef doctor` command** that prints where data lives, how many ADRs, and whether the schema is current. Onboarding friction disappears when a user can `reef doctor` and see green checkmarks.
5. **Add a `--dry-run` to every mutating command**. A team adopting a tracker will *preview* before they commit.

---

## Scenario 4 — Researcher reproducing `math/01_PRESSURE_MATH.md`

**FRICTION SCORE: 3/5 (the core math matches, but there are real discrepancies and hidden heuristics)**

### Specific issues

**4.1 — Core equation correspondence: matches the paper, with one structural caveat.**

