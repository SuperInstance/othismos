"""Tests for the reef ecology module."""

import pytest

from othismos.ecology import (
    ReefLayer,
    Deposit,
    Reef,
    Reefquake,
)


class TestReefSubmission:
    def test_submit_simple(self):
        reef = Reef()
        accepted, msg = reef.submit("d1", "Hello world")
        assert accepted
        assert "ACCEPTED" in msg
        assert reef.total_deposits == 1

    def test_submit_empty_rejected(self):
        reef = Reef()
        accepted, msg = reef.submit("d1", "")
        assert not accepted
        assert "REJECTED" in msg
        assert reef.total_deposits == 0

    def test_submit_with_validation(self):
        reef = Reef()
        validate = lambda c: len(c) > 5
        accepted, _ = reef.submit("d1", "short", validate=validate)
        assert not accepted
        accepted, _ = reef.submit("d1", "long enough", validate=validate)
        assert accepted

    def test_submit_with_references(self):
        reef = Reef()
        reef.submit("base", "Foundation deposit")
        accepted, msg = reef.submit("d2", "Depends on base", references=["base"])
        assert accepted

    def test_submit_missing_reference_rejected(self):
        reef = Reef()
        accepted, msg = reef.submit("d1", "Refers to nothing", references=["nonexistent"])
        assert not accepted
        assert "Gate 2" in msg

    def test_back_reference_updates(self):
        reef = Reef()
        reef.submit("base", "Foundation")
        reef.submit("child", "Child", references=["base"])
        base = reef.query("base")
        assert "child" in base.referenced_by


class TestReefLayering:
    def test_initial_surface_layer(self):
        reef = Reef()
        reef.submit("d1", "New deposit")
        deposit = reef.query("d1")
        assert deposit.layer == ReefLayer.SURFACE

    def test_promotion_to_consolidation(self):
        reef = Reef(consolidation_age=5, foundation_age=100, erosion_age=1000)
        reef.submit("d1", "Will age")
        reef.submit("d2", "References d1", references=["d1"])
        for _ in range(6):
            reef.tick()
        d1 = reef.query("d1")
        assert d1.layer == ReefLayer.CONSOLIDATION

    def test_promotion_to_foundation(self):
        reef = Reef(foundation_age=3, consolidation_age=1, erosion_age=1000)
        reef.submit("base", "Foundation deposit")
        reef.submit("c1", "Child 1", references=["base"])
        reef.submit("c2", "Child 2", references=["base"])
        reef.submit("c3", "Child 3", references=["base"])
        for _ in range(4):
            reef.tick()
        base = reef.query("base")
        assert base.layer == ReefLayer.FOUNDATION


class TestReefErosion:
    def test_orphan_erodes(self):
        reef = Reef(erosion_age=5, consolidation_age=100, foundation_age=1000)
        reef.submit("lonely", "No one references me")
        for _ in range(6):
            reef.tick()
        assert reef.query("lonely") is None
        assert reef.total_deposits == 0

    def test_referenced_deposit_surives(self):
        reef = Reef(erosion_age=5, consolidation_age=100, foundation_age=1000)
        reef.submit("base", "Important")
        reef.submit("child", "Uses base", references=["base"])
        for _ in range(6):
            reef.tick()
        assert reef.query("base") is not None  # has referencer
        # child is orphan (nothing references it) → erodes
        assert reef.query("child") is None


class TestReefquake:
    def test_reefquake_removes_dependents(self):
        reef = Reef()
        reef.submit("foundation", "Critical")
        reef.submit("child1", "Depends on foundation", references=["foundation"])
        reef.submit("child2", "Also depends", references=["foundation"])
        reef.submit("grandchild", "Depends on child1", references=["child1"])

        with pytest.raises(Reefquake) as exc_info:
            reef.fail_deposit("foundation")

        assert exc_info.value.failed_id == "foundation"
        assert "child1" in exc_info.value.affected
        assert "child2" in exc_info.value.affected
        assert "grandchild" in exc_info.value.affected
        assert reef.total_deposits == 0


class TestReefQueries:
    def test_search(self):
        reef = Reef()
        reef.submit("d1", "machine learning optimization")
        reef.submit("d2", "quantum computing algorithm")
        reef.submit("d3", "deep learning gradient")

        results = reef.search("learning")
        ids = {r.id for r in results}
        assert "d1" in ids
        assert "d3" in ids
        assert "d2" not in ids

    def test_citation_graph(self):
        reef = Reef()
        reef.submit("a", "Root")
        reef.submit("b", "Child of a", references=["a"])
        reef.submit("c", "Child of b", references=["b"])

        graph = reef.citation_graph()
        assert graph["c"] == ["b"]
        assert graph["b"] == ["a"]
        assert graph["a"] == []

    def test_depth_distribution(self):
        reef = Reef()
        reef.submit("d1", "one")
        reef.submit("d2", "two")
        dist = reef.depth_distribution()
        assert dist["SURFACE"] == 2
        assert dist["CONSOLIDATION"] == 0

    def test_summary(self):
        reef = Reef()
        reef.submit("a", "Root")
        reef.submit("b", "Child", references=["a"])
        summary = reef.summary()
        assert summary["total_deposits"] == 2
        assert summary["step"] == 0
        assert "layers" in summary
