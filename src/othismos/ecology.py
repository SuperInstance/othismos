"""
Reef ecology — deposit management, layering, and structural memory.

From worldbuilding/05_THE_REEFS_MEMORY.md:
The reef is the accumulated substrate of all verified work.
Deposits pass three gates: structural integrity, connective compatibility,
and pressure resistance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence

import numpy as np


class ReefLayer(IntEnum):
    """The three depths of the reef."""

    SURFACE = 0       # Active zone — new deposits, turbulent, high óthismos
    CONSOLIDATION = 1  # Middle — survived first challenge, settling
    FOUNDATION = 2     # Deep — ancient, load-bearing, change with caution

    @property
    def description(self) -> str:
        return {
            ReefLayer.SURFACE: "Active zone. New deposits land here. High turbulence.",
            ReefLayer.CONSOLIDATION: "Middle zone. Deposits that survived their first challenges.",
            ReefLayer.FOUNDATION: "Deep zone. Ancient, load-bearing, structural. Change with extreme caution.",
        }[self]


@dataclass
class Deposit:
    """A unit of work in the reef.

    Attributes:
        id: Unique identifier
        content: The actual work (code, spec, test, document)
        references: IDs of deposits this one builds on
        referenced_by: IDs of deposits that build on this one
        layer: Current reef layer
        depth_score: Citation depth (recursive reference count)
        structural_integrity: Has this deposit passed validation?
        age: Steps since deposition
    """

    id: str
    content: str
    references: list[str] = field(default_factory=list)
    referenced_by: set[str] = field(default_factory=set)
    layer: ReefLayer = ReefLayer.SURFACE
    depth_score: float = 0.0
    structural_integrity: bool = False
    age: int = 0

    @property
    def is_orphan(self) -> bool:
        """Nothing references this deposit — candidate for erosion.
        Note: having references TO others doesn't protect you.
        Only being referenced BY others keeps you in the reef.
        """
        return not self.referenced_by

    @property
    def reference_count(self) -> int:
        return len(self.referenced_by)


class Reefquake(Exception):
    """Raised when a foundational deposit fails, triggering structural collapse."""

    def __init__(self, failed_deposit_id: str, affected: list[str]):
        self.failed_id = failed_deposit_id
        self.affected = affected
        super().__init__(
            f"Reefquake: deposit '{failed_deposit_id}' failed, "
            f"affecting {len(affected)} dependent deposits: {affected[:5]}..."
        )


class Reef:
    """The accumulated substrate of verified deposits.

    Manages the three gates, layering, erosion, and structural queries.
    """

    # Default layer promotion thresholds (in age steps)
    DEFAULT_CONSOLIDATION_AGE = 100
    DEFAULT_FOUNDATION_AGE = 1000

    # Default erosion: orphan deposits older than this dissolve
    DEFAULT_EROSION_AGE = 500

    # Minimum references for connective compatibility
    MIN_CONNECTIONS = 1

    def __init__(
        self,
        consolidation_age: int | None = None,
        foundation_age: int | None = None,
        erosion_age: int | None = None,
    ):
        self._deposits: dict[str, Deposit] = {}
        self._step = 0
        self.consolidation_age = consolidation_age if consolidation_age is not None else self.DEFAULT_CONSOLIDATION_AGE
        self.foundation_age = foundation_age if foundation_age is not None else self.DEFAULT_FOUNDATION_AGE
        self.erosion_age = erosion_age if erosion_age is not None else self.DEFAULT_EROSION_AGE

    def submit(
        self,
        deposit_id: str,
        content: str,
        references: list[str] | None = None,
        validate: callable | None = None,
    ) -> tuple[bool, str]:
        """Submit a new deposit through the three gates.

        Returns (accepted, reason).
        """
        references = references or []

        # Gate 1: Structural integrity
        if validate is not None:
            try:
                if not validate(content):
                    return False, "Gate 1 REJECTED: structural integrity failed — content invalid"
            except Exception as e:
                return False, f"Gate 1 ERROR: validation raised: {e}"
        else:
            # Default: non-empty content
            if not content or not content.strip():
                return False, "Gate 1 REJECTED: empty content"

        # Gate 2: Connective compatibility
        missing_refs = [r for r in references if r not in self._deposits]
        if missing_refs:
            return False, f"Gate 2 REJECTED: references to non-existent deposits: {missing_refs}"

        # Gate 3: Pressure resistance (simplified — real implementation
        # would test whether the deposit supports additional structure)
        # For now, deposits with more references are considered more pressure-resistant

        # Accept the deposit
        deposit = Deposit(
            id=deposit_id,
            content=content,
            references=list(references),
            structural_integrity=True,
        )

        # Update back-references
        for ref_id in references:
            if ref_id in self._deposits:
                self._deposits[ref_id].referenced_by.add(deposit_id)

        # Compute initial depth score
        deposit.depth_score = self._compute_depth(deposit_id)

        self._deposits[deposit_id] = deposit
        return True, f"ACCEPTED: deposit '{deposit_id}' entered the surface layer"

    def tick(self) -> dict:
        """Advance the reef by one step. Handles aging, layering, erosion.

        Returns a summary of what happened.
        """
        self._step += 1
        eroded: list[str] = []
        promoted: list[tuple[str, ReefLayer]] = []

        for dep_id, deposit in list(self._deposits.items()):
            deposit.age += 1

            # Layer promotion
            old_layer = deposit.layer
            if deposit.age >= self.foundation_age and deposit.reference_count >= 3:
                deposit.layer = ReefLayer.FOUNDATION
            elif deposit.age >= self.consolidation_age and deposit.reference_count >= 1:
                deposit.layer = ReefLayer.CONSOLIDATION
            else:
                deposit.layer = ReefLayer.SURFACE

            if deposit.layer != old_layer:
                promoted.append((dep_id, deposit.layer))

            # Erosion: orphan deposits that are old enough dissolve
            if deposit.is_orphan and deposit.age > self.erosion_age:
                eroded.append(dep_id)
                del self._deposits[dep_id]

        return {
            "step": self._step,
            "total_deposits": len(self._deposits),
            "eroded": eroded,
            "promoted": [(d, l.name) for d, l in promoted],
        }

    def fail_deposit(self, deposit_id: str) -> None:
        """Mark a deposit as structurally failed. Triggers a reefquake.

        Everything that transitively depends on the failed deposit is affected.
        Raises Reefquake exception with the list of affected deposits.
        """
        if deposit_id not in self._deposits:
            raise KeyError(f"Unknown deposit: {deposit_id}")

        # Find all transitive dependents
        affected: list[str] = []
        queue = [deposit_id]
        visited: set[str] = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            if current in self._deposits:
                affected.append(current)
                dependents = set(self._deposits[current].referenced_by)
                queue.extend(dependents)

        # Remove all affected deposits
        for dep_id in affected:
            if dep_id in self._deposits:
                del self._deposits[dep_id]

        raise Reefquake(deposit_id, affected)

    def query(self, deposit_id: str) -> Deposit | None:
        """Retrieve a deposit by ID."""
        return self._deposits.get(deposit_id)

    def search(self, query: str, limit: int = 10) -> list[Deposit]:
        """Simple text search across deposit contents."""
        results = []
        query_lower = query.lower()
        for deposit in self._deposits.values():
            if query_lower in deposit.content.lower():
                results.append(deposit)
        results.sort(key=lambda d: d.depth_score, reverse=True)
        return results[:limit]

    def citation_graph(self) -> dict[str, list[str]]:
        """Return the full citation graph as adjacency list."""
        return {
            dep_id: list(deposit.references)
            for dep_id, deposit in self._deposits.items()
        }

    def depth_distribution(self) -> dict[str, int]:
        """Count deposits by layer."""
        counts = {layer.name: 0 for layer in ReefLayer}
        for deposit in self._deposits.values():
            counts[deposit.layer.name] += 1
        return counts

    def _compute_depth(self, deposit_id: str, memo: dict | None = None) -> float:
        """Compute citation depth recursively.

        depth = 1 + sum of depths of referencing deposits / branching factor.
        """
        if memo is None:
            memo = {}
        if deposit_id in memo:
            return memo[deposit_id]

        deposit = self._deposits.get(deposit_id)
        if deposit is None:
            return 0.0

        if not deposit.referenced_by:
            memo[deposit_id] = 1.0
            return 1.0

        # Prevent infinite recursion on cycles
        memo[deposit_id] = 1.0  # placeholder
        child_depths = sum(self._compute_depth(child, memo) for child in deposit.referenced_by)
        depth = 1.0 + child_depths / max(len(deposit.referenced_by), 1)
        memo[deposit_id] = depth
        return depth

    @property
    def total_deposits(self) -> int:
        return len(self._deposits)

    @property
    def step(self) -> int:
        return self._step

    def summary(self) -> dict:
        """High-level reef statistics."""
        layers = self.depth_distribution()
        orphans = sum(1 for d in self._deposits.values() if d.is_orphan)
        mean_depth = float(np.mean([d.depth_score for d in self._deposits.values()])) if self._deposits else 0.0

        return {
            "total_deposits": self.total_deposits,
            "step": self._step,
            "layers": layers,
            "orphans": orphans,
            "mean_depth_score": mean_depth,
            "oldest_deposit_age": max((d.age for d in self._deposits.values()), default=0),
        }
