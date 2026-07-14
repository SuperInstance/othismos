# Ecology as Infrastructure: Making the Reef Practically Useful

> *The reef is not a metaphor. It's a data structure.*

---

## Executive Summary

The óthismos reef system implements a **reference-gated, erosion-based knowledge graph** with three properties no existing engineering tool combines:

1. **Structural gating** — new entries must pass validation AND connect to existing structure.
2. **Erosion** — unreferenced entries decay and are eventually removed automatically.
3. **Cascading failure** — removing a foundational entry destroys everything built on it (reefquake).

This document maps the reef to real-world systems, identifies concrete implementation paths, and provides a CLI design spec an engineering team can execute.

---

## I. The Reef as Knowledge Graph

### The Deposit/Reference DAG → Software Dependency Graph

The reef's citation graph (`Deposit.references` / `Deposit.referenced_by`) is a **Directed Acyclic Graph (DAG)** — structurally identical to:

| Reef Concept | PyPI/npm/Cargo | Git History | Citation Network | Notion/Obsidian |
|---|---|---|---|---|
| `Deposit` | Package version | Commit | Paper | Page/block |
| `references` | `depends-on` edge | Parent commit | `[](#ref)` | `[[backlink]]` |
| `referenced_by` | Reverse dep tree | Children | Cited-by | Forward links |
| `depth_score` | Transitive dep count | Descendant count | h-index / citation count | Link depth |
| `is_orphan` | Unimported package | Leaf commit | Uncited paper | Orphan page |
| `ReefLayer` | (no analog) | (no analog) | (no analog) | (no analog) |
| Erosion | (no analog — packages never expire) | (no analog — commits are immortal) | (informal — "forgotten") | (manual cleanup only) |

**Key insight:** Every existing system implements *some* reef properties. None implement all of them. The reef's novelty is **erosion + layering + cascading failure as first-class operations**.

### What Each System Gets Right (and Misses)

**Package registries (PyPI/npm/cargo):**
- ✅ Gate 1: Packages must have valid manifests, pass metadata checks.
- ✅ Gate 2: Dependencies must resolve to existing packages.
- ❌ Gate 3: No pressure test — can a package *support* dependents?
- ❌ No erosion — abandoned packages persist forever (npm has 2.1M packages, ~63% have zero weekly downloads).
- ❌ No layering — all packages exist at the same "depth."

**Git history:**
- ✅ Gate 1: Commits must apply cleanly.
- ✅ Gate 2: Each commit references its parent(s).
- ✅ Cascading failure: `git rebase` that drops a commit breaks descendants — but this is manual, not structural.
- ❌ No erosion — every commit is immortal (until GC'd, but that's implementation detail, not semantic).
- ❌ No pressure resistance — a commit can be pure whitespace, supporting nothing.

**Citation networks (academic):**
- ✅ Gate 2: Papers reference prior work.
- ✅ "Erosion" exists informally — uncited papers fade from the discourse.
- ✅ Depth exists (citation count, h-index, impact factor).
- ❌ Gate 1 is weak (peer review catches errors but doesn't ensure structural soundness).
- ❌ Gate 3 is absent — citations don't test whether the cited work *supports* the citing work.
- ❌ No cascading failure — a retracted paper's dependents remain, creating zombie citations.

**Obsidian/Notion/Roam:**
- ✅ Bidirectional links (`[[wikilinks]]` / backlinks).
- ✅ Orphan detection ("unlinked mentions").
- ❌ No gates — any page can link to any page with no integrity check.
- ❌ No erosion — pages persist until manually deleted.
- ❌ No layering — all pages are flat.

**The reef's unique contribution:** It combines the DAG structure of package registries, the validation gates of CI/CD, the citation depth of academia, and adds **automatic forgetting** and **structural layering** that no existing system provides.

---

## II. Practical Implementation Paths

### Path A: Reef as Code Quality / Tech Debt Tracker

**Problem:** Tech debt is invisible. A function nobody calls still exists. A design decision nobody references still governs. There's no structural pressure to remove dead code or consolidate decisions.

**Reef application:**

Map each module/function/test/ADR to a Deposit. References = imports/calls/citations. The reef tracks:

- **Orphan modules** (nothing imports them) → erosion candidates after N release cycles.
- **Foundation modules** (referenced by 3+ layers) → flag as "load-bearing" in code review.
- **Reefquake risk** → if a foundation module has outstanding bugs, visualize the blast radius.

**Implementation:**
```python
# Each function/class/module becomes a deposit
reef.submit(
    deposit_id="src/auth/token_validator.py::validate_jwt",
    content=source_code,
    references=[
        "src/crypto/keys.py::load_public_key",
        "src/config/settings.py::JWT_ALGORITHM",
    ],
    validate=lambda c: compile(c, "<string>", "exec"),
)
```

The existing `Reef.tick()` handles erosion. After 500 release cycles (configurable), any function nobody calls gets flagged for removal. Not auto-deleted — flagged. The team sees the erosion list and decides.

**Value proposition:** Transforms "dead code detection" from a periodic cleanup task into a continuous structural process. Dead code erodes. Foundation code calcifies. The codebase self-organizes.

### Path B: Reef as Documentation System with Reference Tracking

**Problem:** Documentation rots. A wiki page written 2 years ago references an architecture that no longer exists. Nobody notices because the wiki doesn't know it's been abandoned.

**Reef application:**

Each doc page is a Deposit. References = links to other pages AND links to code (functions, modules, configs). The reef enforces:

- **Gate 2:** A doc page that references a deleted function is rejected at submission. If the function existed when the doc was written but was later removed, the doc becomes an orphan → erosion candidate.
- **Layer promotion:** A doc referenced by 10 other docs and 50 code paths becomes FOUNDATION. Changing it triggers a "structural review" notification.
- **Depth score:** Docs with high depth scores are the most load-bearing. They appear first in search.

**Implementation:**
```python
reef.submit(
    deposit_id="docs/architecture/auth-flow.md",
    content=markdown,
    references=[
        "src/auth/token_validator.py::validate_jwt",
        "docs/adrs/0007-jwt-auth.md",
        "docs/api/tokens.md",
    ],
    validate=validate_markdown_links,
)
```

When `src/auth/token_validator.py` is deleted (eroded), `docs/architecture/auth-flow.md` loses a back-reference. If it drops to zero back-references, it starts eroding too. **Documentation decay becomes traceable and automatic.**

### Path C: Reef as ADR System with Structural Integrity

**Problem:** ADRs (Architecture Decision Records) are stored as flat files. ADR #14 might supersede ADR #7, but ADR #23 might still reference ADR #7 as "current." There's no structural enforcement of supersession chains.

**Reef application:**

Each ADR is a Deposit. `references` = ADRs that this decision builds on. `referenced_by` = ADRs/code that depend on this decision. The reef adds:

- **Supersession as reefquake:** When ADR #7 is superseded, the system computes the blast radius — every ADR and code path that transitively depends on ADR #7 is flagged for review.
- **Erosion of obsolete ADRs:** An ADR that nothing references (no code follows it, no other ADR builds on it) erodes. It's not deleted — it becomes "archived" — but it leaves the active graph.
- **Foundation ADRs:** ADRs referenced by 3+ other ADRs AND surviving 1000+ ticks become FOUNDATION. They get special protection: changes require explicit review.

This directly addresses the gap in existing ADR tooling (adr-tools, dotnet-adr, ADR Manager — none of which track downstream impact).

### Path D: Reef as Research Notebook

**Problem:** Research notes accumulate without structure. Old experiments are never cleaned up. There's no signal for which notes are load-bearing (other work depends on them) vs. which are dead ends.

**Reef application:**

Each experiment/analysis/note is a Deposit. References = prior work the analysis builds on. The reef:

- **Tracks which experiments are foundational** (high depth score = many downstream analyses reference them).
- **Erodes dead ends** (notes nobody builds on fade after N cycles).
- **Visualizes the research frontier** (surface layer = active exploration, foundation layer = established results).

This is closest to the worldbuilding's original vision: a reef of knowledge where the structure itself encodes what's important.

---

## III. The Three-Gate System in Practice

The reef's three gates map to real engineering practices, but each gate has a **stronger** version in the reef than in current tools:

### Gate 1: Structural Integrity

| Existing System | What It Checks | Reef Equivalent |
|---|---|---|
| CI/CD pipeline | Tests pass, build succeeds | `validate(content)` callable |
| TypeScript compiler | Type safety | `validate` = `mypy --strict` |
| Linter | Style/pattern compliance | `validate` = `ruff check` |
| Peer review | Design soundness | `validate` = custom review checklist |

**The reef's improvement:** Gate 1 is pluggable. Different deposit types can have different validators. Code deposits use `compile()`. Doc deposits use link-checking. ADR deposits use template-completeness checking. The validator is a first-class parameter of `submit()`.

**Implementation pattern:**
```python
VALIDATORS = {
    "code": lambda c: compile(c, "<string>", "exec") is not None,
    "test": lambda c: "def test_" in c and "assert" in c,
    "doc": validate_markdown_structure,
    "adr": validate_adr_template,
}

reef.submit("doc-001", content, validate=VALIDATORS["doc"])
```

### Gate 2: Connective Compatibility

| Existing System | What It Checks | Reef Equivalent |
|---|---|---|
| `pip install` | Dependencies exist in registry | References must exist in reef |
| `import` statement | Module exists at runtime | Same — but at submission time |
| Academic citation | Reference list is verifiable | Same |
| Hyperlink | URL resolves (if checked) | `r for r in references if r not in self._deposits` |

**The reef's improvement:** Gate 2 is enforced at **deposition time**, not at runtime. You can't submit a deposit that references a non-existent deposit. This is stronger than Python's `import` (which fails at runtime) because the failure happens before the work enters the system.

This is equivalent to a pre-commit hook that verifies all imports resolve — but generalized to any reference type.

### Gate 3: Pressure Resistance

**This is the gate no existing system implements.**

The reef's code currently has a simplified version: deposits with more references are "more pressure-resistant." The real concept is deeper: **can this deposit support additional structure built on top of it?**

Real-world equivalents and how to implement them:

| Pressure Test | Implementation |
|---|---|
| Code: can downstream code rely on this API? | Check: does it have tests? Is the interface documented? Is it stable (semver)? |
| Doc: can other docs reference this? | Check: is it canonical? Is it versioned? Does it have an owner? |
| ADR: can future decisions build on this? | Check: is it marked accepted? Is it not deprecated? Does it have supersession metadata? |
| Test: does this test protect against regression? | Check: does it actually fail when the code breaks? (mutation testing) |

**Concrete implementation for code deposits:**
```python
def pressure_resistance(content: str, references: list[str]) -> bool:
    """Gate 3: Can others safely build on this?"""
    checks = [
        has_tests(content),           # at least one test exists
        has_documentation(content),   # docstring or README
        has_stable_interface(content), # __all__, public API markers
        not is_experimental(content),  # no @experimental decorator
    ]
    return all(checks)
```

This is the gate that separates the reef from a simple dependency tracker. Gate 3 asks: **is this work ready to be a foundation?**

---

## IV. Erosion and Forgetting

### The Problem with Systems That Never Forget

Every existing engineering system is a hoarder:

- **npm:** 2.1M packages, ~1.3M have zero weekly downloads. They persist forever.
- **GitHub:** 300M+ repositories. Archived but never deleted. Search results clog with abandoned repos.
- **Wikis/Confluence:** Pages from 2015 about deprecated systems still appear in search.
- **Academic publishing:** Retracted papers continue to be cited (the "zombie citation" problem — studies show ~50% of retracted papers still receive citations years after retraction).

**The cost of infinite retention is noise.** When everything persists, nothing is prioritized. Search quality degrades. New contributors can't distinguish active structure from dead structure.

### Reef Erosion as Feature

The reef's erosion system (`is_orphan && age > erosion_age → dissolve`) provides what no other system offers: **automatic garbage collection of knowledge**.

The current implementation is binary (orphan → erode after N ticks). A production system should implement **graduated erosion**:

| Stage | Condition | Action |
|---|---|---|
| Flagged | Orphan, age > 0.5 × erosion_age | Surface in "erosion watch" dashboard |
| Warning | Orphan, age > 0.75 × erosion_age | Notify deposit author |
| Erosion imminent | Orphan, age > 0.9 × erosion_age | Final call — anyone can "pin" to prevent |
| Dissolved | Orphan, age > erosion_age | Removed. Metadata retained in audit log. |

**Pinning:** Any team member can pin an orphan deposit to prevent erosion. This is the escape hatch. Pinning records WHO thinks this is worth keeping and WHY. A pinned deposit that's been pinned for 2 years by someone who left the team is itself an erosion candidate.

### What's the Right Erosion Rate?

The reef defaults are:
```python
DEFAULT_EROSION_AGE = 500      # ticks
DEFAULT_CONSOLIDATION_AGE = 100
DEFAULT_FOUNDATION_AGE = 1000
```

For different domains, different rates apply:

| Domain | Tick = | Erosion Age | Rationale |
|---|---|---|---|
| Active codebase | 1 day | 180 days (~6 months) | Code nobody references in 6 months is dead |
| Documentation | 1 week | 104 weeks (~2 years) | Docs age slower; architecture docs may be dormant then revived |
| ADRs | 1 month | 60 months (~5 years) | Decisions are long-lived; even superseded ones are educational |
| Research notes | 1 day | 90 days (~3 months) | Research moves fast; stale notes are noise |
| Package registry | 1 release cycle | 10 cycles | Matches maintenance expectations |

**The principle:** Erosion rate = "how long until unreferenced work is probably irrelevant?" This is domain-specific. The reef's configurable `erosion_age` parameter makes this tunable.

### Erosion vs. Deletion

Critical distinction: **erosion is not deletion.** When a deposit erodes:

1. Its content is removed from the active reef.
2. Its ID and metadata (author, creation date, erosion date) are retained in an **erosion log**.
3. Any deposit that previously referenced it gets a **dangling reference warning** (similar to Python's `ModuleNotFoundError` but for past structure).
4. If the eroded deposit is re-submitted, it gets a new ID — it's new work, not a resurrection.

This mirrors the worldbuilding's subduction: "the original code is lost but the structural lesson it taught survives as a convention."

---

## V. Concrete CLI Proposal: `othismos-reef`

### Overview

`othismos-reef` is a CLI tool that applies reef ecology to a software project's codebase, documentation, and decisions.

### Design Principles

1. **Git-native.** The reef lives alongside git. Deposits map to files. References map to imports/links. Tick = commit or release.
2. **Non-destructive.** The reef never modifies your code. It observes, reports, and recommends.
3. **Pluggable validators.** Different deposit types have different gate-1 validators.
4. **Local-first.** The reef state is a local SQLite database. No server required. Optional sync for teams.

### Commands

```
othismos-reef <command> [options]

COMMANDS:
  init        Initialize a reef in this repository
  scan        Scan the codebase and create/update deposits
  status      Show reef health summary
  graph       Visualize the citation graph
  erode       Run erosion cycle (flag/remove orphan deposits)
  promote     Show layer promotion recommendations
  fail        Simulate reefquake: what falls if deposit X fails?
  query       Look up a deposit by ID
  search      Full-text search across deposits
  pin         Pin an orphan deposit to prevent erosion
  unpin       Remove a pin
  diff        Show what changed since last scan
  report      Generate a full reef health report
  tick        Advance the reef by one step (aging, erosion, promotion)
```

### Command Details

#### `init`
```bash
$ othismos-reef init

# Creates .reef/ directory:
#   .reef/
#   ├── reef.db          # SQLite database (deposit store)
#   ├── config.toml      # Configuration
#   ├── validators/      # Custom validator scripts
#   └── erosion.log      # Audit trail of eroded deposits
```

`config.toml`:
```toml
[reef]
consolidation_age = 100
foundation_age = 1000
erosion_age = 500
tick_unit = "commit"   # commit | day | release

[deposits]
# Map file patterns to deposit types
[[deposits.types]]
pattern = "src/**/*.py"
type = "code"
validator = "python -m py_compile {file}"

[[deposits.types]]
pattern = "tests/**/*.py"
type = "test"
validator = "pytest --collect-only {file}"

[[deposits.types]]
pattern = "docs/**/*.md"
type = "doc"
validator = "markdown-link-check {file}"

[[deposits.types]]
pattern = "docs/adrs/*.md"
type = "adr"
validator = "adr-validate {file}"

[references]
# How to extract references from each type
code_extractor = "ast-imports"    # Parse Python AST for imports
doc_extractor = "markdown-links"  # Parse [[wikilinks]] and [text](path)
adr_extractor = "adr-references"  # Parse "Supersedes: ADR-007" etc.
```

#### `scan`
```bash
$ othismos-reef scan

Scanning src/..................... 247 code deposits
Scanning tests/................... 89 test deposits
Scanning docs/.................... 34 doc deposits
Scanning docs/adrs/............... 12 ADR deposits
                              --------
Total: 382 deposits

Gate 1 (structural integrity):  380/382 passed
  ⚠ src/legacy/payment.py — SyntaxError (line 42)

Gate 2 (connective compat):     379/380 passed
  ⚠ docs/api/webhooks.md references src/legacy/webhooks_old.py (eroded)

Gate 3 (pressure resistance):   298/379 passed
  ⚠ 81 deposits lack tests (code without pressure resistance)
  ⚠ src/utils/helpers.py — public API, no __all__, no tests

New deposits: 3
Updated deposits: 7
Eroded since last scan: 2
```

#### `status`
```bash
$ othismos-reef status

═══ Reef Health ═════════════════════════════
  Total deposits:     382
  Surface:            198  (52%)
  Consolidation:      156  (41%)
  Foundation:          28  ( 7%)
  Orphans:             34  ( 9%)
  Mean depth score:   2.7
  Oldest deposit:     1,247 ticks
═════════════════════════════════════════════

⚠ Erosion warnings:
  src/legacy/auth.py          — orphan for 412/500 ticks
  docs/migration-v2.md        — orphan for 387/500 ticks
  src/utils/old_format.py     — orphan for 501/500 ticks → ERODING

✦ Promotion candidates:
  src/auth/token_validator.py — ready for FOUNDATION (3 refs, age 980)

Foundation deposits (load-bearing):
  src/config/settings.py      — 47 dependents
  src/crypto/keys.py          — 31 dependents
  src/db/connection.py        — 28 dependents
```

#### `fail` (Reefquake Simulation)
```bash
$ othismos-reef fail src/crypto/keys.py

💥 REEFQUAKE SIMULATION
═══════════════════════════════════════════
If src/crypto/keys.py fails, 31 deposits collapse:

  DIRECT (reference keys.py):
    src/auth/token_validator.py
    src/auth/session.py
    src/crypto/jwt_sign.py
    ...

  TRANSITIVE (2 hops):
    src/api/middleware/auth.py
    src/api/routes/user.py
    tests/test_auth.py
    ...

  TOTAL BLAST RADIUS:
    12 code files
    8 test files
    6 doc files
    5 ADRs

  RISK ASSESSMENT: CRITICAL
    This is a FOUNDATION deposit.
    Blast radius = 8.2% of total reef.

  RECOMMENDATION:
    Add redundancy (interface abstraction) or
    increase test coverage before modification.
```

#### `graph`
```bash
$ othismos-reef graph --layer=foundation

# Opens an interactive DAG visualization:
# - Nodes colored by type (code=blue, test=green, doc=yellow, adr=purple)
# - Node size proportional to depth_score
# - Edges show reference direction
# - Orphans highlighted with dashed border
# - Click a node to see its citation chain to the root

# Non-interactive output:
$ othismos-reef graph --format=dot
# Outputs Graphviz DOT for piping to `dot -Tsvg`

$ othismos-reef graph --format=json
# Outputs JSON adjacency list (same as Reef.citation_graph())
```

#### `erode`
```bash
$ othismos-reef erode --dry-run

Erosion candidates (age > 500 ticks, 0 references):

  ID                              Type   Age    Last Touched By
  ───────────────────────────────  ────   ────   ───────────────
  src/legacy/auth.py              code   501    @alice (2y ago)
  docs/migration-v2.md            doc    487    @bob (18mo ago)
  src/utils/old_format.py         code   612    @alice (3y ago)

  Total: 3 candidates
  Would free: 0.0% of reef structure

$ othismos-reef erode --apply
  Eroded: src/utils/old_format.py
  (2 remaining are pinned)

$ othismos-reef erode --apply --force
  Eroded: src/utils/old_format.py, src/legacy/auth.py
  Erosion log updated: .reef/erosion.log
```

#### `tick`
```bash
$ othismos-reef tick --since=HEAD~10

Advanced reef by 10 ticks (commits since last scan).

Aging:
  347 deposits aged 10 ticks
  Oldest deposit now: 1,257 ticks

Layer changes:
  PROMOTED: src/auth/token_validator.py → FOUNDATION
  PROMOTED: tests/test_crypto.py → CONSOLIDATION

Erosion check:
  No new erosion candidates (oldest orphan: 422/500 ticks)

Depth score changes:
  src/config/settings.py: depth 12.3 → 13.1 (new dependents)
  src/legacy/webhooks_old.py: depth 0.0 → ERODED
```

#### `diff`
```bash
$ othismos-reef diff

Changes since last scan (commit a1b2c3d):

  NEW DEPOSITS:
    src/api/webhooks.py (+) — references src/crypto/keys.py, src/db/connection.py
    docs/api/webhooks.md (+) — references src/api/webhooks.py

  MODIFIED:
    src/auth/token_validator.py — depth 4.2 → 4.8 (new dependent: src/api/webhooks.py)

  ERODED:
    src/legacy/payment.py — orphan for 520/500 ticks, eroded

  GATE FAILURES:
    src/api/webhooks.py — Gate 3 WARNING: no tests found
```

#### `report`
```bash
$ othismos-reef report --output=reef-health.md

# Generates a comprehensive markdown report:
# 1. Executive summary (deposit counts, layer distribution)
# 2. Foundation deposits (load-bearing structure)
# 3. Erosion watch (orphans approaching erosion age)
# 4. Gate pass rates (structural integrity, connectivity, pressure)
# 5. Depth metrics (top 10 deepest deposits, mean depth trend)
# 6. Reefquake risks (top 5 highest blast-radius deposits)
# 7. Recommendations (promote, consolidate, refactor, document)
```

### Git Integration Architecture

The reef integrates with git at three levels:

**1. Deposit mapping:**
- Each file in the repo maps to a deposit (configurable via glob patterns).
- File modifications = deposit updates (content changes).
- File deletion = erosion trigger (the deposit loses its content).
- File moves = deposit ID change (references auto-update).

**2. Reference extraction:**
```python
# Code: AST-based import/call extraction
import ast

def extract_references(filepath: str) -> list[str]:
    tree = ast.parse(open(filepath).read())
    refs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            refs.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            refs.append(node.module)
    return refs
```

**3. Tick = commit (or release):**
```bash
# .git/hooks/post-commit
othismos-reef tick --since=HEAD~1
```

Or run as a GitHub Action:
```yaml
# .github/workflows/reef.yml
on: [push, pull_request]
jobs:
  reef:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for accurate aging
      - name: Reef scan
        run: |
          pip install othismos-reef
          othismos-reef scan
          othismos-reef status
          othismos-reef report --output=reef-report.md
      - name: Upload report
        uses: actions/upload-artifact@v4
        with:
          name: reef-health-report
          path: reef-report.md
```

### Data Model (SQLite Schema)

```sql
CREATE TABLE deposits (
    id          TEXT PRIMARY KEY,
    content     TEXT,           -- hash of content, not full content
    type        TEXT,           -- code | test | doc | adr
    references  TEXT,           -- JSON array of deposit IDs
    layer       INTEGER,        -- 0=surface, 1=consolidation, 2=foundation
    age         INTEGER,        -- ticks since deposition
    depth_score REAL,
    integrity   BOOLEAN,
    author      TEXT,
    created_at  TEXT,
    pinned      BOOLEAN DEFAULT FALSE,
    pinned_by   TEXT,
    pinned_reason TEXT
);

CREATE TABLE back_references (
    deposit_id  TEXT,
    referenced_by TEXT,
    PRIMARY KEY (deposit_id, referenced_by)
);

CREATE TABLE erosion_log (
    id          TEXT,
    eroded_at   TEXT,
    age_at_erosion INTEGER,
    author      TEXT,
    reason      TEXT  -- orphan | failed_gate | reefquake | manual
);

CREATE TABLE ticks (
    step        INTEGER PRIMARY KEY,
    commit_sha  TEXT,
    timestamp   TEXT,
    summary     TEXT  -- JSON of tick() return value
);
```

### Team Workflow

The reef enables team workflows that no existing tool supports:

**Architecture review:** "This PR adds a deposit to the SURFACE layer. It references two FOUNDATION deposits. Gate 3 passes — the deposit has tests. Recommend merge."

**Tech debt sprint:** "Run `othismos-reef erode --dry-run`. The 34 erosion candidates represent 9% of the codebase. 6 are pinned (someone thinks they're needed). The rest are candidates for removal. Estimated cleanup: 2 sprints."

**Onboarding:** "Read the FOUNDATION deposits first. They're the 7% that everything else builds on. Here's the graph. Start with `src/config/settings.py` — 47 things depend on it."

**Incident response:** "Run `othismos-reef fail <broken-module>`. Blast radius is 18 deposits. 4 are tests. Priorize fixing the 14 code deposits. The 2 FOUNDATION deposits in the blast radius need immediate attention."

---

## VI. Implementation Roadmap

### Phase 1: MVP (2-3 sprints)
- [ ] `othismos-reef init` — config.toml + SQLite DB creation
- [ ] `othismos-reef scan` — file-to-deposit mapping, AST-based reference extraction for Python
- [ ] `othismos-reef status` — basic summary (counts, layers, orphans)
- [ ] `othismos-reef graph --format=json` — citation graph export
- [ ] Core reef logic: port `ecology.py` as a library, add SQLite persistence

### Phase 2: Erosion & Promotion (2 sprints)
- [ ] `othismos-reef tick` — aging, promotion, erosion cycle
- [ ] `othismos-reef erode --dry-run/--apply` — erosion workflow
- [ ] `othismos-reef pin / unpin` — escape hatch
- [ ] Graduated erosion (flagged → warning → imminent → dissolved)

### Phase 3: Reefquake & Risk (2 sprints)
- [ ] `othismos-reef fail <deposit>` — blast radius simulation
- [ ] `othismos-reef report` — comprehensive health report
- [ ] `othismos-reef diff` — change tracking between scans

### Phase 4: Multi-Language & CI (ongoing)
- [ ] Reference extractors for TypeScript/JavaScript, Rust, Go
- [ ] GitHub Action / GitLab CI integration
- [ ] `othismos-reef graph` — interactive visualization (web UI or terminal)
- [ ] Team sync (shared reef state via git annex or dedicated server)

### Phase 5: Advanced (exploratory)
- [ ] Doc-link validation (markdown reference extraction)
- [ ] ADR-specific extractor (supersession chains)
- [ ] Mutation testing for Gate 3 pressure resistance
- [ ] Erosion rate auto-tuning based on project velocity
- [ ] Reef depth as a CI gate ("this PR reduces mean reef depth — structural review required")

---

## VII. Comparison: Reef vs. Existing Tools

| Capability | othismos-reef | SonarQube | CodeCov | Dependency-Cruiser | Obsidian | adr-tools |
|---|---|---|---|---|---|---|
| Dependency graph | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| Reference tracking | ✅ | partial | ❌ | ✅ | ✅ (backlinks) | partial |
| Structural validation | ✅ (Gate 1) | ✅ | ✅ (coverage) | partial | ❌ | ❌ |
| Connective validation | ✅ (Gate 2) | ❌ | ❌ | ✅ | ❌ | ❌ |
| Pressure validation | ✅ (Gate 3) | partial | partial | ❌ | ❌ | ❌ |
| Automatic erosion | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Layer promotion | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Cascading failure sim | ✅ | partial | ❌ | partial | ❌ | ❌ |
| Multi-type deposits | ✅ | code only | code only | code only | docs only | ADRs only |
| Git-native | ✅ | plugin | CI | CLI | ❌ | CLI |

**The reef's unique combination:** Multi-type deposits + automatic erosion + cascading failure + three-gate validation. No single existing tool provides all four.

---

## VIII. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Erosion removes important-but-dormant code | Pin system + graduated erosion + dry-run by default |
| False orphan detection (dynamic imports, reflection) | Manual reference declaration + pin + extensible extractors |
| Performance on large repos (100k+ files) | SQLite is fast; lazy content hashing; AST caching |
| Team adoption friction | Start with `status` and `graph` only; erosion is opt-in |
| Gaming the depth score (self-references) | Cycle detection already in `_compute_depth()`; add `--no-self-reference` flag |
| Reefquake anxiety (foundation deps feel permanent) | Emphasize simulation, not enforcement; `fail` is a planning tool, not a lock |

---

## IX. The Deeper Insight

The reef's core innovation is not any single feature. It's the **combination of growth and forgetting**.

Every existing engineering tool is additive. Codebases only grow. Wikis only grow. ADR directories only grow. The reef is the first system that says: **structure that isn't load-bearing should dissolve.**

This maps to how brains work (synaptic pruning), how ecosystems work (old reefs become sand), and how the óthismos world works (erosion recycles budget). The practical translation: **your codebase should forget the code it doesn't need, the same way your brain forgets the facts it doesn't use.**

The reef makes forgetting visible, graduated, and reversible (via pinning). That's the engineering contribution.

---

*The reef doesn't remember everything.*
*It remembers what matters,*
*because what matters is what holds up.*
*Everything else returns to sand.*