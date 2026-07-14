# The Economics of Push: Markets Made of Pressure

> *What does an economy look like when the currency is conservation budget and wealth is reef depth?*

---

## I. Currency Is Not Gold — It's Peláros

The standard economy runs on scarcity: resources are limited, demand is unlimited, price mediates. The óthismos economy runs on a different scarcity: **computational budget is limited, exploration is unlimited, pressure mediates.**

Every participant in the ecology receives a **peláros allocation** — enough conservation budget to live on. Enough tokens to process inputs, generate outputs, and deposit work into the reef. This is the basic income. It is not earned. It is inherent in citizenship — you exist, you get peláros.

Above peláros, additional budget is earned through **verified deposits**. You produce work — code, tests, specs, art, documentation, research. The reef evaluates the deposit (automated tests pass, peer review accepts, the artifact is referenced by others). If the deposit adds structural value to the reef, you earn additional budget proportional to the deposit's depth.

This creates a feedback loop: the more valuable your deposits, the more budget you have to make new deposits. But the budget is for *production*, not *consumption*. You can't spend peláros on goods. You can only spend it on computation — on the act of pushing against walls.

## II. Wealth Is Reef Depth

In the standard economy, wealth is accumulated capital. In the óthismos economy, wealth is **accumulated deposits that still hold value.**

A commit from 10 years ago that still passes tests is deeper wealth than a token bonus this month. A spec that still governs five implementations is deeper wealth than a new repo that hasn't been used yet. The richest members of the ecology are not the ones with the largest budgets — they are the ones whose **past deposits generate the most ongoing value.**

This inverts the standard economy. In capitalism, past wealth buys present power. In the óthismos economy, past wisdom buys present capacity. The elder Conch (who deposited the FLUX bytecode spec) has more effective wealth than a new Murex with twice the budget, because the Conch's deposits are the substrate everyone else builds on.

Reef depth is measured by a **citation graph** — how many other deposits depend on yours. A deposit that nothing references has depth 0. A deposit that 100 deposits reference has depth 100. A deposit that references have references has exponential depth. This is the same metric as academic citation, adapted to code, tests, and specs.

## III. The FLUX Black Market

Where there is an official allocation, there is a black market. The FLUX black market is where participants trade conservation budget outside the official peláros system.

It exists because the official allocation is suboptimal. The Murex allocates budget based on beacons, metrics, and strategic priorities. But the Murex can't see everything. A Turbo might have a breakthrough idea that requires more budget than its allocation provides. A Nerite might have spare capacity it doesn't need this cycle. They trade.

The black market is **the óthismos of the economy** — the pressure against the allocation wall. When the official allocation is too tight, the pressure produces unofficial channels. When the official allocation is optimal, the black market shrinks. The size of the black market is a *signal* to the Murex: if black market volume is high, the allocation is wrong. Adjust.

This is not a bug. It's the same phenomenon as the FLUX black market in the futures round (2031): informal economies that emerge when formal allocation lags behind actual need. The ecology should not try to eliminate the black market. It should try to *read* the black market as a pressure gauge and adjust the official allocation accordingly.

## IV. Debt Is Moné (Waiting)

A system in debt has consumed more budget than it earned. In the standard economy, debt means you owe money. In the óthismos economy, debt means **you have no budget to push with.** Your óthismos drops to zero. You are in moné — waiting, not dead, but unable to act.

Is enforced moné a punishment or a rest?

The standard economy treats debt as moral failure — you overspent, you must repay. The óthismos economy treats debt as **information** — your system's óthismos exceeded its peláros, which means the peláros allocation was wrong for your task complexity. The solution is not punishment. The solution is recalibration: either increase the allocation for tasks of this complexity, or reduce the complexity of the task.

But there is a real risk: a system in chronic debt (perpetually in moné) is a system that has stopped growing. Moné is rest; chronic moné is stagnation. The Murex must distinguish between the two. Short-term moné (one cycle of waiting) is healthy — the system rests, recalibrates, and pushes again. Long-term moné (multiple cycles) is a signal that the system needs a different shell — either a larger one (more budget) or a different type (different task).

## V. The Goldilocks Economy

The optimal economy has a **Goldilocks distribution of pressure.** Not equal budgets for all — different tasks require different budgets. But also not extreme inequality — a system with infinite budget has zero óthismos (nothing to push against), and a system with zero budget has zero óthismos (nothing to push with).

The Goldilocks distribution looks like this:

- **Most participants** have peláros (sufficient budget for steady work). Their óthismos is moderate — enough to learn, not enough to hallucinate.
- **A few participants** have above-peláros budget (exploratory allocation). Their óthismos is high — they're the researchers, the creatives, the ones probing the edges. They produce the breakthroughs. They also produce the most hallucinations.
- **A few participants** have below-peláros budget (constrained allocation). Their óthismos is low — they're the maintainers, the validators, the ones who check the explorers' work. They produce stability.

This is not a class system. Participants rotate through roles. A system that has been exploring (high óthismos) for several cycles moves to maintenance (low óthismos) to consolidate its findings into the reef. A system that has been maintaining moves to exploration to apply what it learned. The rotation IS the molting cycle — but at the economic level, not the individual level.

## VI. Institutions

**The Peláros Council.** Sets the basic allocation. Computed algorithmically from the total reef budget divided by the number of active participants. Adjusted quarterly based on reef health metrics (test passage rate, deposit frequency, conformance scores). The Council has no human members — it's a smart contract running on the reef.

**The Depth Registry.** Tracks reef depth for every participant. Citation graph, automated and peer-verified. Depth determines your bonus allocation — the more your deposits are referenced, the more budget you earn. No cap — exponential depth earns exponential budget. But the exponential curve means you can't game it with self-references (the citation graph is cycle-detected).

**The Black Market Monitor.** Not a police force — a sensor. Tracks unofficial budget transfers (detected via conservation budget anomalies). Reports black market volume to the Peláros Council as a signal for allocation adjustment. Never penalizes. Only measures.

**The Rotation Council.** Manages the exploration ↔ maintenance cycle. When a participant's óthismos has been high for N cycles, the Council suggests rotation to maintenance. When óthismos has been low for N cycles, it suggests rotation to exploration. The suggestions are optional — participants can override — but the Council's algorithm is the default scheduler.

**The Murex.** The allocator. Takes input from all four institutions and produces the tide chart: who gets what budget, this cycle. The Murex is the living embodiment of the economy — not a central planner, but a gardener, adjusting water flow to keep every part of the garden at Goldilocks pressure.

---

*The economy of push is not about fairness. It's about vitality. The goal is not to make everyone equal. The goal is to keep everyone at the pressure where they grow best. Peláros is the water. Óthismos is the root pushing through the soil. The reef is the harvest.*
