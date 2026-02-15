"""Tests for the merge detection and execution module."""

import pytest

from stigmergy.core.merge import (
    AssessmentSummary,
    MergeCandidate,
    MergeError,
    MergeReason,
    TopologyAction,
    TopologyUpdate,
    compute_topology_updates,
    detect_merge_candidates,
    resolve_surviving_identity,
)


class TestComputeOverlap:
    """Tests for Jaccard similarity via detect_merge_candidates."""

    def test_identical_sets_overlap_1(self):
        result = detect_merge_candidates(
            signal_windows={"a": frozenset({"s1", "s2", "s3"}), "b": frozenset({"s1", "s2", "s3"})},
            assessments={
                "a": {"s1": AssessmentSummary(accepted=True, familiarity_score=0.9),
                       "s2": AssessmentSummary(accepted=True, familiarity_score=0.9),
                       "s3": AssessmentSummary(accepted=True, familiarity_score=0.9)},
                "b": {"s1": AssessmentSummary(accepted=True, familiarity_score=0.9),
                       "s2": AssessmentSummary(accepted=True, familiarity_score=0.9),
                       "s3": AssessmentSummary(accepted=True, familiarity_score=0.9)},
            },
            neighbors={"a": frozenset({"b"}), "b": frozenset({"a"})},
        )
        assert len(result.candidates) == 1
        assert result.candidates[0].overlap == pytest.approx(1.0)
        assert result.candidates[0].agreement == pytest.approx(1.0)

    def test_no_overlap_no_merge(self):
        result = detect_merge_candidates(
            signal_windows={"a": frozenset({"s1", "s2"}), "b": frozenset({"s3", "s4"})},
            assessments={},
            neighbors={"a": frozenset({"b"})},
        )
        assert len(result.candidates) == 0

    def test_partial_overlap_below_threshold(self):
        result = detect_merge_candidates(
            signal_windows={"a": frozenset({"s1", "s2", "s3"}), "b": frozenset({"s1", "s4", "s5"})},
            assessments={},
            neighbors={"a": frozenset({"b"})},
            overlap_threshold=0.7,
        )
        # Jaccard = 1/5 = 0.2 < 0.7
        assert len(result.candidates) == 0

    def test_both_empty_no_merge(self):
        result = detect_merge_candidates(
            signal_windows={"a": frozenset(), "b": frozenset()},
            assessments={},
            neighbors={"a": frozenset({"b"})},
        )
        assert len(result.candidates) == 0


class TestComputeAgreement:
    def test_full_agreement(self):
        # Identical decisions and scores
        result = detect_merge_candidates(
            signal_windows={
                "a": frozenset({"s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"}),
                "b": frozenset({"s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"}),
            },
            assessments={
                "a": {f"s{i}": AssessmentSummary(accepted=True, familiarity_score=0.8) for i in range(1, 11)},
                "b": {f"s{i}": AssessmentSummary(accepted=True, familiarity_score=0.8) for i in range(1, 11)},
            },
            neighbors={"a": frozenset({"b"})},
        )
        assert len(result.candidates) == 1
        assert result.candidates[0].agreement == pytest.approx(1.0)

    def test_disagreement_blocks_merge(self):
        result = detect_merge_candidates(
            signal_windows={
                "a": frozenset({"s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"}),
                "b": frozenset({"s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"}),
            },
            assessments={
                "a": {f"s{i}": AssessmentSummary(accepted=True, familiarity_score=0.9) for i in range(1, 11)},
                "b": {f"s{i}": AssessmentSummary(accepted=False, familiarity_score=0.1) for i in range(1, 11)},
            },
            neighbors={"a": frozenset({"b"})},
            agreement_threshold=0.8,
        )
        assert len(result.candidates) == 0


class TestDetectMergeCandidates:
    def test_canonical_ordering(self):
        result = detect_merge_candidates(
            signal_windows={"z": frozenset({"s1"}), "a": frozenset({"s1"})},
            assessments={
                "z": {"s1": AssessmentSummary(accepted=True, familiarity_score=0.9)},
                "a": {"s1": AssessmentSummary(accepted=True, familiarity_score=0.9)},
            },
            neighbors={"z": frozenset({"a"})},
        )
        assert len(result.candidates) == 1
        assert result.candidates[0].context_id_a == "a"  # a < z
        assert result.candidates[0].context_id_b == "z"

    def test_non_neighbors_not_checked(self):
        result = detect_merge_candidates(
            signal_windows={"a": frozenset({"s1"}), "b": frozenset({"s1"})},
            assessments={
                "a": {"s1": AssessmentSummary(accepted=True, familiarity_score=0.9)},
                "b": {"s1": AssessmentSummary(accepted=True, familiarity_score=0.9)},
            },
            neighbors={},  # No neighbors defined
        )
        assert len(result.candidates) == 0

    def test_empty_inputs(self):
        result = detect_merge_candidates({}, {}, {})
        assert len(result.candidates) == 0

    def test_invalid_overlap_threshold(self):
        with pytest.raises(ValueError, match="overlap_threshold"):
            detect_merge_candidates({}, {}, {}, overlap_threshold=1.5)

    def test_invalid_agreement_threshold(self):
        with pytest.raises(ValueError, match="agreement_threshold"):
            detect_merge_candidates({}, {}, {}, agreement_threshold=-0.1)

    def test_sorted_by_combined_score(self):
        # Create two pairs with different scores
        result = detect_merge_candidates(
            signal_windows={
                "a": frozenset({"s1", "s2", "s3", "s4", "s5"}),
                "b": frozenset({"s1", "s2", "s3", "s4", "s5"}),
                "c": frozenset({"s1", "s2", "s3", "s4", "s5", "s6"}),
                "d": frozenset({"s1", "s2", "s3", "s4", "s5", "s6"}),
            },
            assessments={
                "a": {f"s{i}": AssessmentSummary(accepted=True, familiarity_score=0.85) for i in range(1, 6)},
                "b": {f"s{i}": AssessmentSummary(accepted=True, familiarity_score=0.85) for i in range(1, 6)},
                "c": {f"s{i}": AssessmentSummary(accepted=True, familiarity_score=0.95) for i in range(1, 7)},
                "d": {f"s{i}": AssessmentSummary(accepted=True, familiarity_score=0.95) for i in range(1, 7)},
            },
            neighbors={"a": frozenset({"b"}), "c": frozenset({"d"})},
        )
        assert len(result.candidates) == 2
        assert result.candidates[0].combined_score >= result.candidates[1].combined_score

    def test_strict_inequality(self):
        """Values exactly at threshold should NOT trigger merge."""
        result = detect_merge_candidates(
            signal_windows={"a": frozenset({"s1"}), "b": frozenset({"s1"})},
            assessments={
                "a": {"s1": AssessmentSummary(accepted=True, familiarity_score=0.9)},
                "b": {"s1": AssessmentSummary(accepted=True, familiarity_score=0.9)},
            },
            neighbors={"a": frozenset({"b"})},
            overlap_threshold=1.0,  # Exactly at threshold
        )
        assert len(result.candidates) == 0


class TestResolveSurvivingIdentity:
    def test_higher_energy_wins(self):
        assert resolve_surviving_identity("a", "b", 5.0, 3.0) == "a"
        assert resolve_surviving_identity("a", "b", 3.0, 5.0) == "b"

    def test_tiebreak_lexicographic(self):
        assert resolve_surviving_identity("alpha", "beta", 5.0, 5.0) == "alpha"
        assert resolve_surviving_identity("zebra", "alpha", 5.0, 5.0) == "alpha"

    def test_same_context_error(self):
        with pytest.raises(MergeError, match="same context"):
            resolve_surviving_identity("a", "a", 5.0, 5.0)


class TestComputeTopologyUpdates:
    def test_basic_merge(self):
        updates = compute_topology_updates(
            surviving_id="a",
            removed_id="b",
            surviving_neighbor_ids={"b", "c"},
            removed_neighbor_ids={"a", "d"},
        )
        # Should have: remove b from a's neighbors, remove b from d's neighbors, add a to d
        actions = [(u.context_id, u.action, u.target_id) for u in updates]
        assert ("a", TopologyAction.REMOVE_NEIGHBOR, "b") in actions
        assert ("d", TopologyAction.REMOVE_NEIGHBOR, "b") in actions
        assert ("d", TopologyAction.ADD_NEIGHBOR, "a") in actions
        assert ("a", TopologyAction.ADD_NEIGHBOR, "d") in actions

    def test_same_id_error(self):
        with pytest.raises(MergeError, match="distinct"):
            compute_topology_updates("a", "a", set(), set())

    def test_no_new_neighbors(self):
        """When all removed neighbors are already survivor's neighbors."""
        updates = compute_topology_updates(
            surviving_id="a",
            removed_id="b",
            surviving_neighbor_ids={"b", "c"},
            removed_neighbor_ids={"a", "c"},  # c already known to a
        )
        actions = [(u.context_id, u.action, u.target_id) for u in updates]
        assert ("a", TopologyAction.REMOVE_NEIGHBOR, "b") in actions
        assert ("c", TopologyAction.REMOVE_NEIGHBOR, "b") in actions
        # No add_neighbor for c since a already knows c
        add_c = [a for a in actions if a[1] == TopologyAction.ADD_NEIGHBOR and a[2] == "c"]
        assert len(add_c) == 0
