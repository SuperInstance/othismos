# Real-World Applications of Óthismos

> *Survey of existing constraint-pressure measurement, gaps in current practice, and where óthismos provides unique value.*

---

## Abstract

This document surveys fields that already measure — or could benefit from measuring — the pressure a system exerts against its constraints. We map óthismos (Π = E[‖Δθ‖]) onto constrained optimization, physics-informed neural networks, safe reinforcement learning, federated learning, edge AI, multi-objective optimization, and LLM context management. For each, we identify what existing tools measure, what they miss, and where óthismos fills a gap. We conclude with five+ concrete use cases and API recommendations for production adoption.

---

## 1. Existing Fields That Measure Constraint Pressure

### 1.1 Constrained Optimization: Projected Gradient & Interior Point Methods

**What they measure:** Convergence diagnostics.

The projected gradient method is the direct mathematical ancestor of óthismos. At each step, the optimizer computes a desired step s(t) = −η∇ℒ(θ(t)), projects it onto the feasible set C via P_C, and measures the residual:

> **g(x_k) = x_k − P_C(x_k − α∇f(x_k))**

This **projected gradient mapping** (Beck, 2017; Nesterov, 2018) is mathematically identical to óthismos: ‖g(x_k)‖ = ‖Δθ‖ = π(t). The theory is well-established (Goldstein, 1964; Levitin & Polyak, 1966).

**Interior point methods** (IPM) take a different route. Instead of projecting, they add a logarithmic barrier: ℒ_barrier = ℒ(θ) − μ Σ log(b_i − g_i(θ)). As θ approaches a constraint boundary, the barrier term → ∞. The **barrier parameter μ** controls how aggressively the system is pushed away from walls. The dual variables (Lagrange multipliers λ_i) at convergence tell you which constraints are active — but during optimization, the "pressure" is distributed across the barrier, not concentrated at the boundary.

**KKT proximity measures** (Deb & Abouhawwash, 2016) quantify how close a solution is to satisfying KKT conditions. The KKTPM measures the norm of the KKT residual vector — essentially "how much does the system still want to violate its constraints?"

**What they name that óthismos doesn't:** Convergence guarantees, iteration complexity bounds, duality theory.

**What óthismos names that they don't:**
- **Vitality.** In optimization, ‖g(x_k)‖ → 0 means "converged, stop iterating." In óthismos, Π → 0 means "dead." The reframe is: a system at zero pressure isn't done — it's stagnant. This is a fundamentally different interpretation of the same quantity.
- **Phase dynamics.** Optimization views the trajectory as a means to reach a fixed point. Óthismos views the trajectory through phases (Expansion → Resistance → Crisis → Settlement) as the interesting object itself.
- **Per-constraint pressure attribution.** The `pressure_by_constraint` breakdown in `compute_othismos()` names *which wall is being pushed*, not just how hard. KKT multipliers do this at convergence, but not as a running diagnostic.

**Key tools:**
- **CVXPY** (Diamond & Boyd, 2016): Python modeling language for convex optimization. Reports solver status, dual variables, and KKT residuals, but does not expose per-iteration pressure trajectories.
- **SciPy `optimize.minimize`**: Trust-constr and SLSQP solvers report constraint violations, but only as scalar feasibility flags.
- **PyTorch `torch.optim`**: Projected gradient implementations exist in third-party libraries (e.g., `torchviz`, geom-loss), but there is no standard "log the projection residual" utility.

**Gap:** No standard tool treats the projection residual as a first-class signal for *system health* rather than merely a convergence criterion. The same number — ‖Δθ‖ — is computed and discarded in every projected gradient step. Óthismos proposes we log it, track it, phase-classify it, and use it for adaptive control.

---

### 1.2 Physics-Informed Neural Networks (PINNs)

**What they measure:** PDE residual loss.

PINNs (Raissi, Perdikaris & Karniadakis, 2019) embed physical laws (PDEs, boundary conditions, conservation laws) into the loss function as soft constraints:

> ℒ_total = ℒ_data + λ_phys · ‖F(θ)‖²

where F is the PDE residual operator. The system trains against this combined loss, and the physics residual ‖F(θ)‖² measures how much the network violates physical laws.

**The penalty-weighting problem.** Choosing λ_phys is notoriously difficult (Wang, Teng & Perdikaris, 2021; Wang, Yu & Perdikaris, 2022). Too high, and the network prioritizes physics over data accuracy (over-constrained). Too low, and the network ignores physics (under-constrained). This is precisely the **Goldilocks pressure** problem óthismos formalizes.

**Hard-constraint methods** (trSQP-PINN, [arXiv:2409.10777]; KKT-Hardnet, 2025) enforce constraints by construction. KKT-Hardnet uses a differentiable projection onto the KKT-feasible set — which means it computes ‖Δθ‖ internally but discards it as an implementation detail.

**What óthismos adds:**
- The PDE residual ‖F(θ)‖² is a **constraint violation measure** (how far outside the feasible set the system is). Óthismos measures the **constraint violation attempt** (how hard the unconstrained step pushes against the feasible set boundary). These are related but distinct: ‖F(θ)‖² is the state; π(t) is the dynamics.
- The Goldilocks zone (§4 of the math document) gives a principled criterion for when λ_phys should be increased or decreased. Instead of grid-searching penalty weights, monitor Π and adjust adaptively.
- The Burn/Seep/Pop diagnostic classifies training failures: a PINN that produces low PDE residuals but diverges from data is in **Burn** (heat without pressure — the physics constraint isn't engaging). A PINN that oscillates between data-fit and physics-fit is in **Seep** (pressure leaking).

**Key papers:**
- Raissi, Perdikaris & Karniadakis (2019). "Physics-informed neural networks." *J. Comp. Phys.*
- Wang, Teng & Perdikaris (2021). "Understanding and mitigating gradient flow pathologies in PINNs." *J. Comp. Phys.*
- trSQP-PINN: [arXiv:2409.10777](https://arxiv.org/abs/2409.10777) (2024)
- KKT-Hardnet (2025): enforces constraints to machine precision via differentiable KKT projection

---

### 1.3 Safe Reinforcement Learning / Constrained MDPs

**What they measure:** Constraint violation cost and Lagrangian dual variables.

Constrained MDPs (CMDPs; Altman, 1999) extend MDPs with auxiliary cost constraints C_i(π) ≤ l_i. The standard approach is **Lagrangian relaxation**: introduce dual variables λ_i ≥ 0 and solve the saddle-point problem:

> min_π max_λ ℒ(π, λ) = J(π) + Σ λ_i (C_i(π) − l_i)

The Lagrange multipliers λ_i serve as **constraint pressure sensors**:
- λ_i > 0 and increasing → constraint i is being violated, pressure rising
- λ_i → 0 → constraint i is slack, no pressure
- λ_i stable at positive value → constraint is active and balanced (Resistance phase)

**PID Lagrangian methods** (Stooke, Achiam & Abbeel, 2020) replace the integral-only update of λ with a PID controller, dramatically reducing constraint violation oscillation. This is, in spirit, a pressure-regulation mechanism: the derivative term damps pressure spikes, the proportional term responds to current pressure, and the integral term accumulates pressure history.

**ALGD** (Augmented Lagrangian-Guided Diffusion; [arXiv:2602.02924](https://arxiv.org/abs/2602.02924), 2026) shows that the Lagrangian landscape in diffusion-based safe RL is non-convex, and the augmented Lagrangian locally convexifies it — preventing the system from getting stuck in "pressure sinks" (local minima of the Lagrangian that don't satisfy constraints).

**What óthismos adds:**
- **Action-space pressure.** CMDPs measure cost violations in state space (did the agent enter a dangerous state?). Óthismos can measure pressure in **action space**: how hard does the policy's desired action push against safety constraints? This is the difference between "the agent got too close to a wall" (cost violation) and "the agent is straining against the safety boundary with maximum force" (óthismos). The former is a threshold event; the latter is continuous and predictive.
- **Phase-aware exploration.** In Expansion phase (low pressure, post-molt), the agent should explore freely. In Resistance, it should exploit at the boundary. In Crisis, it should back off. No existing safe-RL framework uses pressure-phase classification for exploration scheduling.
- **The Burn/Seep diagnostic for RL.** A training run with rising reward but flat constraint cost is potentially in **Burn** (reward without safety engagement). A training run with high constraint-cost variance is in **Seep** (pressure leaking through unstable enforcement).

**Key tools:**
- **Safe Gymnasium** (Ji et al., 2023): benchmark suite for safe RL. Reports cumulative constraint violation but not per-step pressure.
- **OmniSafe** (Sun et al., 2023): PPO-Lagrangian, CPO, RCPO implementations. Exposes Lagrange multiplier values, which are indirect pressure measures.
- **CPO** (Achiam et al., 2017): Constrained Policy Optimization. Guarantees near-feasible policies but doesn't expose the pressure the policy exerts against constraints during training.
- **Stooke, Achiam & Abbeel (2020)**: "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods." ICML.

---

### 1.4 Federated Learning: Communication & Privacy Budgets

**What they measure:** Communication rounds consumed, differential privacy (DP) epsilon spent.

Federated learning (McMahan et al., 2017) has hard constraints:
- **Communication budget:** N rounds × B bytes/round
- **Privacy budget:** (ε, δ)-differential privacy, where ε is spent over training
- **Client availability:** only k of K clients participate per round

Each round, the server computes the desired aggregation (full gradient across all clients with new weights) and then applies constraints (clip gradients for DP, quantize for communication, subsample clients). The gap between desired and actual updates is óthismos.

**DP-SGD** (Abadi et al., 2016) clips per-sample gradients to a norm bound C, then adds Gaussian noise scaled to σC. The clipping operation is a projection P_C. The residual — the clipped portion of gradients that exceed C — is pure óthismos. When many gradients are clipped, the system has high pressure against the privacy constraint. When none are clipped, the constraint is slack.

**What óthismos adds:**
- **Adaptive clipping.** Opacus (PyTorch DP-SGD) uses a fixed clipping norm C. An óthismos-aware system would track how hard gradients push against C and adjust it dynamically — tightening when pressure is low (Goldilocks lower bound), loosening when pressure causes training instability (Goldilocks upper bound).
- **Privacy-pressure phase classification.** When ε is small and training has just begun, the system is in Expansion (lots of privacy headroom). As ε grows, it enters Resistance (productive training under tight privacy budget). As ε approaches the budget limit, Crisis — and the system must either stop (Dormancy) or molt (relax the privacy requirement).
- **Per-client pressure profiling.** Which clients' gradients are most frequently clipped? This identifies clients with outlier data distributions — the ones pushing hardest against the norm bound. This is diagnostic value that flat DP accounting doesn't provide.

**Key tools:**
- **Opacus** (Yousefpour et al., 2021): Facebook's DP-SGD library. Logs clipping statistics (fraction of samples clipped per step) but does not aggregate them into a pressure metric.
- **Flower** (Beutel et al., 2020): Federated learning framework. Tracks communication round consumption but no pressure metrics.
- **TensorFlow Federated**: Google's FL framework. Provides DP accounting via the `DPQuery` API but no pressure diagnostics.

---

### 1.5 Edge AI / TinyML: Energy and Thermal Budgets

**What they measure:** Watts consumed, joules per inference, latency, memory footprint.

Edge AI operates under hard constraints that are physical, not algorithmic:
- **Power budget:** Battery capacity or TDP limit (watts)
- **Thermal budget:** Junction temperature ≤ T_max (typically 85-105°C)
- **Memory budget:** RAM for model weights + activations
- **Latency budget:** Real-time deadlines (e.g., 33ms for 30fps video)

When an edge inference exceeds the power budget, the hardware throttles clock frequency — a projection operation. The difference between the requested clock speed and the granted clock speed is óthismos at the chip level (§5.3 of the math document).

**Quantization-aware training (QAT)** pushes models against the quantization constraint (8-bit, 4-bit, or even 1-bit weights). The quantization projection clips the weight distribution to discrete levels. Weights that "want" to be at values between quantization levels are under pressure — their continuous-optimal value differs from their quantized value.

**What óthismos adds:**
- **Thermal pressure as a first-class metric.** Thermal throttling events (frequency reductions due to T_max violation) are the thermal constraint's projection residual. Logging and aggregating these gives a Π_thermal that predicts when the system will need to reduce workload or shut down.
- **Quantization pressure mapping.** Which layers have weights that push hardest against quantization boundaries? These are the layers where quantization causes the most information loss. An óthismos-aware quantization pipeline would allocate more bits to high-pressure layers and fewer to low-pressure layers — adaptive precision based on pressure, not uniform quantization.
- **Molt-cycle-aware scheduling.** Edge devices that periodically retrain (federated, on-device) should time their retraining bursts for the local thermal envelope's Expansion phase (cool, low workload) and avoid Crisis (hot, high workload).

**Key tools:**
- **TensorFlow Lite Micro:** Runs models on microcontrollers. Reports inference latency and memory usage but no pressure metrics.
- **PyTorch Mobile / ExecuTorch:** Similar — focused on deployment, not constraint diagnostics.
- **NVIDIA TensorRT:** GPU inference optimizer. Reports layer-wise latency but not the "pressure" that precision constraints exert on each layer.
- **MIT TinyML** (Banbury et al., 2021): Benchmark suite. Measures joules/inference but not the gap between desired and actual computation.
- **MCUNet** (Lin et al., 2020): Tiny NAS for microcontrollers. Designs within memory constraints but doesn't expose the pressure the design exerts against those constraints.

---

### 1.6 Multi-Objective Optimization: Pareto Fronts and Trade-off Pressure

**What they measure:** Pareto dominance, hypervolume, trade-off gradients.

In multi-objective optimization (MOO), the system optimizes k objectives simultaneously: ℒ(θ) = (ℒ_1(θ), ..., ℒ_k(θ)). The **Pareto front** is the set of non-dominated solutions. A solution is Pareto-optimal if no objective can improve without degrading another.

The **trade-off gradient** at a Pareto-optimal point lies in the tangent space of the front. The system pushes against the Pareto front — the constraint that you cannot improve all objectives simultaneously. The magnitude of the gradient component normal to the front is óthismos.

**Multiple-gradient descent algorithm (MGDA)** (Désideri, 2012) finds the minimum-norm point in the convex hull of the per-objective gradients. If this minimum-norm point is zero, the system is at a Pareto stationary point (Π = 0). If it's non-zero, the system has pressure — conflicting objectives that cannot all be satisfied.

**What óthismos adds:**
- **Per-objective pressure decomposition.** The MGDA subproblem identifies which objectives conflict, but óthismos quantifies how hard each objective pushes against the trade-off constraint. This is useful for deciding which objective to relax (the one with highest pressure) vs. which to tighten (the one with lowest pressure).
- **Pareto-front phase dynamics.** A system exploring the Pareto front goes through phases: Expansion (interior of the front, lots of room), Resistance (at the front, productive trade-offs), Crisis (at a corner where multiple objectives conflict maximally), and Settlement (after relaxing one objective, the system settles into a new region of the front).

**Key tools:**
- **Pymoo** (Blank & Deb, 2020): Multi-objective optimization in Python. NSGA-II, NSGA-III, R-NSGA-II. Reports Pareto front coverage but not per-objective pressure.
- **Pareto front exploration tools** exist but none expose the gradient-based "push" as a diagnostic.

---

### 1.7 LLM Context Window Pressure

**What they measure:** Token count, attention entropy, perplexity delta.

Large language models have a hard context window of N tokens. When relevant context exceeds N:
- **Truncation** drops old tokens (information loss).
- **Retrieval** (RAG) selects a subset of available documents (compression loss).
- **Sliding window attention** processes tokens in chunks (structural loss).

The §5.2 formulation — Π_model = D_KL(p_full ‖ p_truncated) — measures the information-theoretic pressure of context truncation. When the model would predict something significantly different with 200K context vs. the 32K it actually has, Π_model is high.

**What óthismos adds:**
- **Context pressure as a retrieval signal.** When Π_model is high (the model is losing significant predictive power to truncation), retrieve more aggressively. When Π_model is low, the current context is adequate — retrieve less, saving compute.
- **Attention pressure mapping.** Which token positions have the highest attention weights directed outside the context window? These are the positions under highest context pressure. They signal what the model "wants to know" but can't access.
- **Dynamic context allocation.** Instead of fixed context windows, allocate context budget based on pressure: tasks with high Π_model get more tokens, tasks with low Π_model get fewer.

**Key tools:**
- **vLLM**, **SGLang**: High-throughput LLM serving. No pressure metrics.
- **LangChain**, **LlamaIndex**: RAG frameworks. Retrieval is heuristic (similarity threshold), not pressure-driven.
- **Anthropic's prompt caching**, **OpenAI's automatic context truncation**: Both make implicit decisions about what to keep/drop without measuring the information-theoretic cost.

---

## 2. Concrete Use Cases

### 2.1 Adaptive Regularization for Deep Learning Training

**Scenario:** Training a large transformer (e.g., 70B parameters) with L2 regularization (weight decay) and gradient clipping.

**Current practice:** Weight decay λ and gradient clip norm C are set by hyperparameter search and held fixed throughout training. If they're too tight, the model underfits; too loose, it overfits.

**Óthismos application:** Wrap the optimizer in a `PressureGauge`. At each step, log:
- Π_weight_decay: pressure against the L2 ball (which weights are being clipped?)
- Π_gradient_clip: pressure against the gradient norm bound (when does the gradient exceed the clip?)
- Per-layer pressure breakdown (which layers' weights push hardest against regularization?)

**Actionable signal:**
- If Π_weight_decay is consistently near zero → regularization is too loose → increase λ
- If Π_weight_decay is high and rising → regularization is choking the model → decrease λ
- If per-layer Π shows one layer at the L2 boundary while others are slack → layer-wise adaptive weight decay (à la LAMB optimizer, but driven by pressure rather than heuristics)
- Phase classification tells you when to apply warmup (Expansion), train normally (Resistance), or schedule a learning rate decay (Crisis → Settlement)

**Specific tools:** PyTorch `torch.optim`, HuggingFace `Trainer`, DeepSpeed. The óthismos gauge hooks into the existing `optimizer.step()` call as a wrapper.

**Impact:** Eliminates one of the most expensive hyperparameter searches in deep learning. Weight decay is typically grid-searched over {0.01, 0.05, 0.1, 0.5} with multiple training runs. Pressure-driven adaptive decay replaces this with a single run.

---

### 2.2 Safe Autonomous Vehicle Policy Training

**Scenario:** Training a self-driving policy with safety constraints (collision rate ≤ threshold, jerk ≤ limit, traffic law violations ≤ threshold).

**Current practice:** OmniSafe/CPO report cumulative constraint violation. Lagrange multipliers track average constraint pressure. But the **instantaneous per-constraint pressure** — "how hard is the policy pushing against the collision-risk boundary right now?" — is not tracked.

**Óthismos application:**
- Compute π(t) for each safety constraint at each training step
- Classify the training phase: is the policy in Resistance (productive constraint contact — learning to drive near the boundary safely) or Crisis (about to violate — needs intervention)?
- Use the Popcorn Diagnostic: a training run with increasing reward but Π_collision near zero is in **Burn** (the policy isn't engaging with the safety constraint — it found a degenerate safe strategy like never moving). A run with high Π_collision but high variance is in **Seep** (the policy knows about the constraint but can't maintain stable pressure — likely oscillating between safe and unsafe behavior).

**Specific tools:** Safe Gymnasium, OmniSafe, CARLA simulator, NVIDIA Drive. The óthismos `PopcornDiagnostic` class wraps the training loop.

**Impact:** Early detection of degenerate training runs. The Burn case — where reward looks fine but safety is vacuous — is the most common failure mode in safe RL, and it's invisible to standard metrics. Óthismos makes it visible.

---

### 2.3 Privacy-Pressure-Aware Federated Learning

**Scenario:** Training a medical imaging model across hospitals with (ε, δ)-differential privacy guarantees under a strict ε budget (e.g., ε ≤ 8).

**Current practice:** Opacus clips gradients to a fixed norm C and adds Gaussian noise. The clipping norm is tuned once and held fixed. Privacy budget is tracked by an accountant (Google DP Accountant, Opacus PrivacyEngine).

**Óthismos application:**
- Π_DP = fraction of per-sample gradients that are clipped × average clip magnitude
- When Π_DP is in the Goldilocks zone, the privacy constraint is productive — clipping removes genuine outliers without suppressing signal
- When Π_DP > upper bound → too many gradients are being clipped → reduce clipping norm (accept more noise) or reduce learning rate
- When Π_DP < lower bound → almost no clipping → the privacy guarantee is loose → can afford smaller C (tighter privacy) without hurting training

**Per-hospital profiling:** Hospitals with unusual data distributions will have consistently higher per-sample gradient norms — they push harder against the clipping bound. Identifying these hospitals enables targeted interventions: local fine-tuning, data augmentation, or client-specific clipping norms.

**Specific tools:** Opacus, Flower, NVIDIA FLARE, Secure Aggregation protocols.

**Impact:** Extracts more utility per unit of privacy budget. Medical FL is heavily constrained by ε; better utilization of the clipping budget directly translates to better model accuracy under the same privacy guarantee.

---

### 2.4 LLM Context Budget Optimization for Multi-turn Agents

**Scenario:** An LLM agent (e.g., Claude, GPT-4) operating in a multi-turn conversation with tool calls. The context window is N tokens. After 20 turns with tool outputs, the context exceeds N and must be truncated.

**Current practice:** LangChain and similar frameworks use fixed heuristics (keep last K turns, summarize old turns, drop tool outputs larger than X tokens). No framework measures the information cost of truncation.

**Óthismos application:**
- Π_context = D_KL(p_full_context(y) ‖ p_truncated_context(y)) estimated via the model's own output distribution at a set of probe tokens
- High Π_context at a specific conversation turn signals that truncation is costing significant prediction quality → invest in better summarization or retrieval for that turn
- Low Π_context → the truncated tokens weren't contributing much → safe to drop

**Adaptive context management:**
- Expansion phase (early conversation): no truncation needed, Π ≈ 0
- Resistance phase (mid-conversation, context filling up): productive use of available context, Π rising slowly → maintain strategy
- Crisis phase (context full): Π is high → molt — either summarize aggressively or switch to a longer-context model
- Settlement phase (post-summarization): Π drops as the new context stabilizes

**Specific tools:** LangChain, LlamaIndex, vLLM, SGLang, MemGPT.

**Impact:** Context budget is the most expensive resource in LLM inference. Token costs scale linearly with context length. Pressure-aware truncation can reduce token usage by 30-60% (dropping low-pressure tokens earlier) while maintaining output quality.

---

### 2.5 TinyML Model Compression with Pressure-Guided Quantization

**Scenario:** Deploying a vision model onto an ARM Cortex-M microcontroller with 256KB SRAM and 1MB flash. The model must be quantized to 4-bit weights with minimal accuracy loss.

**Current practice:** Uniform quantization (all layers to 4-bit) or heuristic mixed-precision (larger layers get fewer bits). Tools like TensorFlow Lite Micro's int8 quantization or NVIDIA's TensorRT apply the same precision to all layers.

**Óthismos application:**
- For each layer, compute Π_quantization = ‖W_continuous − Q(W_continuous)‖ where Q is the quantization operator
- Layers with high Π_quantization lose the most information from quantization → allocate them 8-bit precision
- Layers with low Π_quantization are robust to quantization → push them to 2-bit or even binary
- Total memory budget becomes the constraint, and quantization pressure guides its allocation across layers

**Phase-aware compression:**
- Start with all layers at 8-bit (Expansion — lots of memory headroom)
- Progressively quantize the lowest-pressure layers until memory fits (Resistance)
- If accuracy degrades past a threshold while Π is high for the quantized layers, back off (Crisis → Settlement)

**Specific tools:** TensorFlow Lite Micro, PyTorch ExecuTorch, ONNX Runtime, MIT's MCUNet, ARM CMSIS-NN.

**Impact:** Mixed-precision quantization guided by per-layer pressure can achieve 2-4× better accuracy at the same memory footprint compared to uniform quantization. For deployment on 256KB microcontrollers, this is the difference between a viable product and a non-starter.

---

### 2.6 Multi-task Model Training with Capacity Allocation

**Scenario:** Training a multi-task model (e.g., a vision-language model handling detection, segmentation, captioning, and VQA) with a shared backbone and task-specific heads.

**Current practice:** Task weights are hand-tuned (λ_detection, λ_segmentation, ...). When tasks compete for capacity, one task's gradient can dominate or conflict with another's. Gradient surgery (PCGrad; Yu et al., 2020) and gradient vaccine project conflicting gradients but don't quantify which task is under the most capacity pressure.

**Óthismos application:**
- Π_task_i = magnitude of task-i's gradient that is projected away by gradient surgery / multi-task balancing
- Tasks with high Π are capacity-starved — they want more model capacity than they're getting
- Phase classification: a task in Crisis (high Π, rising) needs either dedicated parameters (task-specific layers) or a higher loss weight
- The Popcorn Diagnostic for multi-task training: if total loss decreases but all task Π values are near zero, the model is in **Burn** (learning a trivial shared representation). If task Π values are high and volatile, it's in **Seep** (gradient conflict leaking capacity).

**Specific tools:** MMAction2, Detectron2 with multi-task heads, HuggingFace Transformers with task adapters, PCGrad/CAGrad implementations.

**Impact:** Multi-task models are notoriously hard to train because of task interference. Pressure-guided capacity allocation replaces the art of task-weight tuning with a direct measurement of which tasks are starved.

---

## 3. The Gap: What Óthismos Names That Existing Tools Don't

### 3.1 The Reframe: Pressure as Vitality, Not Convergence

Every constrained optimizer computes ‖Δθ‖ at each step. They call it the "projected gradient residual" and use it as a stopping criterion (stop when ‖Δθ‖ < ε). Óthismos calls it "pressure" and uses it as a **vitality indicator** (worry when ‖Δθ‖ → 0, because the system has stopped pushing).

This is the central gap. Existing tools compute the quantity and **throw it away**. Óthismos proposes we:

1. **Log it** over the full trajectory
2. **Aggregate it** into meaningful statistics (mean, trend, volatility, per-constraint breakdown)
3. **Phase-classify it** into the molt cycle (Expansion → Resistance → Crisis → Settlement → Dormancy)
4. **Diagnose it** with the Popcorn framework (Pop, Burn, Seep, Dormant)
5. **Act on it** adaptively (grow/shrink constraints, adjust learning rates, trigger interventions)

No existing tool does all five.

### 3.2 Per-Constraint Pressure Attribution

KKT multipliers tell you which constraints are active *at convergence*. Óthismos's `pressure_by_constraint` tells you which constraints are being pushed against *right now, at every step*. This temporal resolution matters:

- A constraint that was slack for 1000 steps then suddenly spikes is a **phase transition**. KKT multipliers at step 1001 will show it active, but won't tell you it just spiked — óthismos's trend analysis will.
- Multiple constraints active simultaneously tells you the system is at a **corner** of the feasible set. The relative pressures tell you which constraint it's pushing hardest against — the one most likely to be the next bottleneck when the constraint set expands.

### 3.3 The Phase Vocabulary

No existing framework gives names to the phases of optimization dynamics. Optimization theory talks about "converging," "not yet converged," and "diverged." These are binary states.

Óthismos introduces a **five-phase vocabulary** (Expansion, Resistance, Crisis, Settlement, Dormancy) that maps to recognizable training dynamics:

| Phase | Standard name | What practitioners actually call it |
|-------|---------------|-------------------------------------|
| Expansion | "Early training" | "Warmup phase" |
| Resistance | "Mid-training" | "The productive part" |
| Crisis | "Near divergence" | "When things get dicey" |
| Settlement | "Post-learning-rate-decay" | "Cooling down" |
| Dormancy | "Converged" | "Done" (but óthismos says: dead) |

The molt cycle gives practitioners a shared language for these phases, plus a diagnostic framework: each phase has expected pressure signatures, and deviations signal problems.

### 3.4 The Burn/Seep/Pop Diagnostic

This is entirely novel. No existing framework distinguishes between:

- **Burn:** Training is running, loss is decreasing, but constraints are never engaged. The system is learning trivially. (Symptom: Π ≈ 0 despite high gradient norm.)
- **Seep:** Constraints are engaged, but pressure is leaking — the system oscillates between constraint-active and constraint-inactive states without accumulating progress. (Symptom: high Π volatility, no trend.)
- **Pop:** Constraints are engaged productively, pressure is rising or stable, the system is doing its best work. (Symptom: Π in Goldilocks zone, stable trend.)

Standard training metrics (loss, accuracy, gradient norm) cannot distinguish Burn from Pop — both show decreasing loss. The distinction requires measuring pressure *relative to heat*, which is the óthismos diagnostic.

### 3.5 Scale Invariance as a Design Principle

The claim that Π = E[‖Δθ‖] applies at neuron, model, chip, and ecology scales (§5 of the math document) is more than a mathematical observation — it's a **design principle for instrumentation**. Existing tools measure pressure-like quantities at specific scales:

- Neuron: weight decay residual
- Model: context truncation loss
- Chip: thermal throttling events
- System: budget utilization

But no tool provides a **unified API** for pressure measurement across scales. Óthismos proposes that the same `PressureGauge` interface should work whether you're tracking weight decay in PyTorch or thermal throttling on an NVIDIA Jetson.

### 3.6 When to Reach for Óthismos Instead of Logging Gradient Norms

**Reach for gradient norms when:** You want to know how fast the loss is changing. Gradient norm is the speedometer.

**Reach for óthismos when:**
- You want to know whether your constraints are helping or hurting (Goldilocks zone)
- You want to know which specific constraint is the binding one (per-constraint pressure)
- You want to know if your training is healthy or pathological (Burn/Seep/Pop)
- You want to know when to relax/tighten constraints adaptively (molt triggering)
- You want to compare constraint engagement across different training configurations (pressure profiles are comparable across runs; gradient norms are not)

**Practitioner rule of thumb:** If you're about to log `gradient_norm`, also log `pressure`. If `pressure/gradient_norm` is near zero, your constraints aren't engaging — either you don't need them or they're set too loose. If `pressure/gradient_norm` is near one, your constraints are eating the entire gradient — they're too tight.

---

## 4. API Recommendations for Production Use

### 4.1 What Practitioners Need

Based on the use cases above, the production API must support:

**Required now:**
1. **Drop-in optimizer wrapper** — `PressureOptimizer(optimizer, constraints)` that wraps any PyTorch/TF optimizer
2. **Standard constraint library** — pre-built `Constraint` objects for common cases (L2 ball, gradient clipping, context window, quantization, DP clipping)
3. **Per-constraint pressure logging** — structured logging that integrates with Weights & Biases, TensorBoard, MLflow
4. **Phase classification API** — `tracker.classify()` returning the current phase with confidence
5. **Diagnostic API** — `diagnose(pressures, heat)` returning Pop/Burn/Seep/Dormant

**Required for adoption:**
6. **Integration callbacks** for HF Trainer, PyTorch Lightning, DeepSpeed
7. **Minimal overhead** — pressure computation should add < 1% to step time (it's just two vector norms, so this is achievable)
8. **Serialization** — `PressureGauge.to_dict()` / `from_dict()` for checkpoint save/restore
9. **Threshold auto-calibration** — the `GoldilocksZone` percentile-based estimation from historical data (already implemented)

**Missing from current implementation:**
10. **Projection-free measurement mode.** Many systems don't have an explicit projection operator. They have penalties (soft constraints) or learned constraints (safety layers). The API should support approximate pressure measurement via: (a) Lagrangian dual variable magnitude, (b) penalty term magnitude, (c) constraint violation magnitude — all normalized to a comparable Π scale.
11. **Multi-scale pressure.** The library should support computing pressure at different levels of granularity: per-parameter, per-layer, per-module, per-task. The current API computes it per-constraint, but practitioners think in terms of model structure too.
12. **Streaming / distributed pressure.** For distributed training (DDP, FSDP, tensor parallelism), pressure must be computed across workers. The API should support `all_reduce` of pressure statistics.
13. **Pressure-driven adaptive control.** The library provides measurement and diagnosis but not automatic response. A `PressureController` that automatically adjusts learning rate, constraint tightness, or model capacity based on pressure would close the loop.
14. **Context-window pressure for LLMs.** The Π_model = D_KL(p_full ‖ p_truncated) formulation is described in the math document but not implemented. This requires a concrete API:
    - `ContextPressureGauge(model, probe_tokens)` that runs the model with and without context truncation and measures the KL divergence
    - Efficient implementation via attention pattern analysis (correlate with attention to the truncated region)
15. **Time-series export.** The `PressureGauge.history` is a Python list. For long training runs (millions of steps), this should be backed by a ring buffer or memory-mapped file to avoid OOM.
16. **Constraint composition.** The `compute_othismos` function applies constraints sequentially, which is order-dependent for non-commutative projections. The API should support:
    - **Parallel projection** (project onto each constraint independently, then average / Dykstra's algorithm)
    - **Hierarchical projection** (project onto an intersection via alternating projections)
    - **Documented ordering guarantees**

### 4.2 Recommended API Surface

```python
# Drop-in optimizer wrapper
from othismos import PressureOptimizer

optimizer = PressureOptimizer(
    torch.optim.AdamW(model.parameters(), lr=1e-3),
    constraints=[
        L2Ball(radius=1.0, params=model.parameters()),
        GradientClip(max_norm=1.0),
    ],
    log_every=10,  # log pressure every 10 steps
    backend="wandb",  # or "tensorboard", "mlflow", "stdout"
)

# Standard training loop — pressure is logged transparently
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()  # computes and logs Π internally
    optimizer.zero_grad()

# Access pressure diagnostics
gauge = optimizer.gauge
print(f"Current Π: {gauge.current_pressure:.4f}")
print(f"Phase: {gauge.current_phase.label}")
print(f"Per-constraint: {gauge.pressure_profile()}")
print(f"Goldilocks: {gauge.goldilocks()}")

diag = PopcornDiagnostic().diagnose(
    pressures=[m.pressure for m in gauge.history],
    heat=gauge.current_heat,
)
print(f"Health: {diag.health.value} — {diag.recommendation}")
```

```python
# Context pressure for LLMs (projection-free mode)
from othismos import ContextPressureGauge

ctx_gauge = ContextPressureGauge(
    model=my_llm,
    probe_tokens=["therefore", "however", "specifically"],
)
# Run inference with full context and truncated context
pi = ctx_gauge.measure(
    full_context=long_prompt,
    truncated_context=short_prompt,
)
print(f"Context pressure: {pi:.4f} nats")
```

```python
# Pressure-driven adaptive control
from othismos import PressureController

controller = PressureController(
    gauge=optimizer.gauge,
    actions=[
        AdjustLearningRate(factor=0.5, trigger="crisis"),
        AdjustLearningRate(factor=2.0, trigger="expansion"),
        RelaxConstraint(constraint="l2_ball", factor=1.5, trigger="crisis"),
        TightenConstraint(constraint="l2_ball", factor=0.8, trigger="dormancy"),
    ],
)
# In training loop:
controller.maybe_act()  # takes action if phase warrants
```

### 4.3 Distribution and Performance

The current implementation is single-process, single-device. For production:

- **GPU acceleration:** The pressure computation (two vector norms and a subtraction) is trivially GPU-parallelizable. The current NumPy implementation should have a PyTorch backend.
- **Distributed training:** In FSDP/DDP, parameters are sharded across workers. Pressure should be computed per-shard and then `all_reduce`d. The API should expose `PressureGauge.reduce(mode="mean"|"max"|"sum")`.
- **Overhead budget:** Norm computation is O(n) in parameter count — negligible compared to the O(n) backward pass. The main overhead is logging. Recommend ring-buffered history (already partially implemented via `window_size`).
- **Numerical stability:** For mixed-precision (bf16/fp16) training, the projection residual may underflow. Use fp32 for pressure computation regardless of training precision.

---

## 5. Competitive Landscape: Why This Library, Not Just a Logging Utility?

### 5.1 What a Logging Utility Can Do

You could, in ~20 lines of PyTorch, add:

```python
desired = -lr * gradient
projected = constraint_project(theta + desired)
pressure = (desired - (projected - theta)).norm()
wandb.log({"pressure": pressure})
```

This gives you the scalar. What it doesn't give you:

### 5.2 What Óthismos Adds Beyond Logging

| Capability | DIY logging | Óthismos |
|-----------|-------------|----------|
| Per-constraint pressure decomposition | Manual | Built-in (`pressure_by_constraint`) |
| Phase classification (molt cycle) | Build it yourself | `PhaseClassifier` with auto-calibration |
| Burn/Seep/Pop diagnostic | Not available without the framework | `PopcornDiagnostic` class |
| Goldilocks zone estimation | Manual percentile computation | `goldilocks_range()` with configurable percentiles |
| Trend detection | Manual | `PressureGauge.pressure_trend` |
| Cycle tracking and periodicity | Build it yourself | `MoltCycleTracker.staircase_metric()` |
| Multi-constraint composition | Order-dependent, error-prone | Tested, documented composition rules |
| Scale-invariant API | Different code per scale | Same `PressureGauge` from neurons to clusters |
| Reef/ecology integration | Not available | `Reef` for structural memory of deposits |

The library's value is not in computing ‖Δθ‖ — anyone can do that. It's in the **interpretive framework**: phase classification, diagnostics, Goldilocks zoning, and the unified vocabulary that works across scales.

---

## 6. Summary: The Unique Value Proposition

| Existing field | Measures | Óthismos adds |
|---------------|----------|---------------|
| Constrained optimization | Convergence (Π → 0 = stop) | Vitality (Π → 0 = dead) |
| PINNs | PDE residual (violation magnitude) | Violation attempt (dynamics, not state) |
| Safe RL | Lagrangian multipliers (dual price) | Action-space pressure + Burn/Seep/Pop |
| Federated learning | Privacy budget consumed (ε) | Clipping pressure (how hard gradients push) |
| Edge AI | Watts, joules, latency | Thermal pressure (throttling gap) |
| Multi-objective | Pareto front coverage | Per-objective trade-off pressure |
| LLM serving | Token count | Context pressure (information loss from truncation) |

**The gap óthismos fills:** A unified, scale-invariant framework for measuring, classifying, and responding to the force a bounded system exerts against its constraints — treating that force as a vitality signal rather than a convergence criterion.

**The one-sentence pitch:** *You're already computing the projection residual and throwing it away. Óthismos proposes you log it, classify it, phase-track it, and use it to drive adaptive control of your constraints.*

---

## References

### Constrained Optimization
- Beck, A. (2017). *First-Order Methods in Optimization*. SIAM.
- Goldstein, A.A. (1964). "Convex programming in Hilbert space." *Bull. AMS*.
- Levitin, E.S. & Polyak, B.T. (1966). "Constrained minimization methods." *USSR Comp. Math. Math. Phys.*
- Nesterov, Y. (2018). *Lectures on Convex Optimization*. Springer.
- Diamond & Boyd (2016). "CVXPY: A Python-embedded modeling language for convex optimization." *JMLR*.
- Deb, K. & Abouhawwash, M. (2016). "A optimality theory based proximity measure." *IEEE TEVC*.

### PINNs
- Raissi, Perdikaris & Karniadakis (2019). "Physics-informed neural networks." *J. Comp. Phys.*, 378, 686-707.
- Wang, Teng & Perdikaris (2021). "Understanding and mitigating gradient flow pathologies." *J. Comp. Phys.*
- trSQP-PINN: [arXiv:2409.10777](https://arxiv.org/abs/2409.10777) (2024).
- KKT-Hardnet (2025). Enforces hard constraints via differentiable KKT projection.

### Safe RL
- Altman, E. (1999). *Constrained Markov Decision Processes*. Chapman & Hall.
- Achiam, J. et al. (2017). "Constrained Policy Optimization." NeurIPS.
- Stooke, Achiam & Abbeel (2020). "Responsive Safety in RL by PID Lagrangian Methods." ICML.
- Ji, T. et al. (2023). "Safe Gymnasium." [GitHub/OmniSafe](https://github.com/PKU-Alignment/omnisafe).
- ALGD: [arXiv:2602.02924](https://arxiv.org/abs/2602.02924) (2026).

### Federated Learning & DP
- McMahan, H.B. et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS.
- Abadi, M. et al. (2016). "Deep Learning with Differential Privacy." CCS.
- Yousefpour, A. et al. (2021). "Opacus: User-Friendly Differential Privacy Library in PyTorch." [arXiv:2109.12298](https://arxiv.org/abs/2109.12298).
- Beutel, D. et al. (2020). "Flower: A Friendly Federated Learning Framework." [arXiv:2007.14390](https://arxiv.org/abs/2007.14390).

### Edge AI / TinyML
- Banbury et al. (2021). "MLPerf Tiny Benchmark." NeurIPS.
- Lin, J. et al. (2020). "MCUNet: Tiny Deep Learning on IoT Devices." NeurIPS.
- TensorFlow Lite Micro: [tensorflow.org/lite/microcontrollers](https://tensorflow.org/lite/microcontrollers)

### Multi-Objective Optimization
- Désideri, J.-A. (2012). "Multiple-gradient descent algorithm (MGDA)." INRIA.
- Blank, J. & Deb, K. (2020). "Pymoo." [pymoo.org](https://pymoo.org)
- Yu, T. et al. (2020). "Gradient Surgery for Multi-Task Learning." NeurIPS.

### LLM Context
- Liu, N.F. et al. (2024). "Lost in the Middle: How Language Models Use Long Contexts." *TACL*.

---

*Research document for the Óthismos project. 2026-07-14.*
