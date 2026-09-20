"""Eight-rule triage table and implemented SOP-PAL-03 checks (R14-R16)."""

from .checks import (
    analyze,
    check_centroid_offset,
    check_column_tilt,
    check_load_height,
    check_overhang,
    check_wrap_presence,
)
from .config import (
    DEFAULT_SOP_CONFIG_PATH,
    SopConfig,
    Threshold,
    load_sop_config,
)
from .triage import (
    ALLOWED_CLASSIFICATIONS,
    RULE_IDS,
    TRIAGE_TABLE,
    Classification,
    TriageEntry,
    all_entries,
    classifications,
    get_entry,
    pose_dependent_rule_ids,
    rule_names,
    verify_table_complete,
)

__all__ = [
    "ALLOWED_CLASSIFICATIONS",
    "RULE_IDS",
    "TRIAGE_TABLE",
    "Classification",
    "TriageEntry",
    "all_entries",
    "classifications",
    "get_entry",
    "pose_dependent_rule_ids",
    "rule_names",
    "verify_table_complete",
    # config loader (task 9.3, R18)
    "DEFAULT_SOP_CONFIG_PATH",
    "SopConfig",
    "Threshold",
    "load_sop_config",
    # verifiable-subset checks (task 9.3, R15)
    "analyze",
    "check_overhang",
    "check_load_height",
    "check_column_tilt",
    "check_wrap_presence",
    "check_centroid_offset",
]
