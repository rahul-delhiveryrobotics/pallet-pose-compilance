"""Unit tests for the authoritative SOP-PAL-03 triage table (task 9.1, R14).

These example-based tests pin the shape and honesty invariants of the
eight-rule triage table: all rule ids 1..8 present exactly once, valid
classifications, non-empty justifications, list-typed assumptions/hidden
regions, and hidden regions justifying every not_verifiable rule. The
property-based coverage of Property 14 lives in task 9.2 (out of scope here).
"""

import pytest

from pallet_pose_compliance.sop import triage
from pallet_pose_compliance.sop.triage import (
    ALLOWED_CLASSIFICATIONS,
    RULE_IDS,
    TRIAGE_TABLE,
    TriageEntry,
    all_entries,
    classifications,
    get_entry,
    pose_dependent_rule_ids,
    rule_names,
    verify_table_complete,
)

# The classifications the triage encodes, used to keep the reusable exports in
# lock-step with the authoritative table (task 9.3/9.7/9.10 consume these).
EXPECTED_CLASSIFICATIONS = {
    1: "partially_verifiable",
    2: "verifiable",
    3: "partially_verifiable",
    4: "not_verifiable",
    5: "verifiable",
    6: "not_verifiable",
    7: "partially_verifiable",
    8: "not_verifiable",
}


def test_all_eight_rule_ids_present_exactly_once():
    """Rule ids 1..8 each appear exactly once (R14.4, Property 14 shape)."""
    keys = sorted(TRIAGE_TABLE.keys())
    assert keys == list(range(1, 9))
    assert len(TRIAGE_TABLE) == 8

    entries = all_entries()
    assert [e.rule_id for e in entries] == list(range(1, 9))
    # No duplicate rule ids among the entries.
    assert len({e.rule_id for e in entries}) == 8


def test_verify_table_complete_passes_for_authoritative_table():
    """The invariant helper accepts the authoritative table."""
    assert verify_table_complete() is True
    assert verify_table_complete(TRIAGE_TABLE) is True


def test_each_entry_has_valid_classification():
    """Every classification is in the allowed set (R14.1)."""
    for rule_id in RULE_IDS:
        entry = get_entry(rule_id)
        assert entry.classification in ALLOWED_CLASSIFICATIONS


def test_classifications_match_expected_single_side_triage():
    """The encoded classifications match the documented single-side reasoning."""
    assert dict(classifications()) == EXPECTED_CLASSIFICATIONS
    for rule_id, expected in EXPECTED_CLASSIFICATIONS.items():
        assert get_entry(rule_id).classification == expected


def test_each_entry_has_non_empty_justification():
    """Every rule records a justification for its classification (R14.2)."""
    for entry in all_entries():
        assert isinstance(entry.justification, str)
        assert entry.justification.strip(), f"rule {entry.rule_id} justification empty"


def test_each_entry_has_list_assumptions_and_hidden_regions():
    """Assumptions and hidden regions are (tuple) list-like collections (R14.3)."""
    for entry in all_entries():
        assert isinstance(entry.assumptions, tuple)
        assert isinstance(entry.hidden_regions, tuple)
        # Every assumption/region is a non-empty string.
        for a in entry.assumptions:
            assert isinstance(a, str) and a.strip()
        for h in entry.hidden_regions:
            assert isinstance(h, str) and h.strip()


def test_not_verifiable_rules_document_hidden_regions():
    """not_verifiable rules name at least one hidden region justifying this (R14.3)."""
    not_verifiable = [e for e in all_entries() if e.classification == "not_verifiable"]
    # Rules 4, 6, 8 are the not-verifiable ones under the single-side view.
    assert {e.rule_id for e in not_verifiable} == {4, 6, 8}
    for entry in not_verifiable:
        assert entry.hidden_regions, (
            f"rule {entry.rule_id} is not_verifiable but names no hidden region"
        )


def test_rule_names_are_present_and_reusable():
    """Every rule exposes a non-empty human-readable name for reuse."""
    names = rule_names()
    assert sorted(names.keys()) == list(range(1, 9))
    for rule_id, name in names.items():
        assert isinstance(name, str) and name.strip()


def test_pose_dependent_rule_ids():
    """Overhang, column tilt, and centroid are the pose-dependent rules (R15.4)."""
    assert pose_dependent_rule_ids() == frozenset({1, 3, 7})


def test_get_entry_rejects_unknown_rule():
    """Requesting a rule outside 1..8 raises KeyError."""
    with pytest.raises(KeyError):
        get_entry(0)
    with pytest.raises(KeyError):
        get_entry(9)


def test_triage_table_is_read_only():
    """The authoritative table cannot be mutated in place."""
    with pytest.raises(TypeError):
        TRIAGE_TABLE[1] = get_entry(2)  # type: ignore[index]


def test_triage_entry_rejects_not_verifiable_without_hidden_region():
    """A not_verifiable entry with no hidden region is rejected (R14.3)."""
    with pytest.raises(ValueError):
        TriageEntry(
            rule_id=4,
            rule_name="x",
            classification="not_verifiable",
            justification="because",
            assumptions=(),
            hidden_regions=(),
        )


def test_triage_entry_rejects_empty_justification():
    """A rule with an empty justification is rejected (R14.2)."""
    with pytest.raises(ValueError):
        TriageEntry(
            rule_id=2,
            rule_name="height",
            classification="verifiable",
            justification="   ",
        )


def test_verify_table_complete_rejects_incomplete_table():
    """A table missing a rule id is rejected."""
    incomplete = {rid: get_entry(rid) for rid in range(1, 8)}
    with pytest.raises(ValueError):
        verify_table_complete(incomplete)


def test_module_exports_match_package_exports():
    """The sop package re-exports the triage table symbols for downstream reuse."""
    from pallet_pose_compliance import sop

    assert sop.TRIAGE_TABLE is triage.TRIAGE_TABLE
    assert sop.get_entry is triage.get_entry
    assert sop.all_entries is triage.all_entries
