from fpl_oracle.data.features import FEATURE_COLUMNS
from scripts.generate_architecture_feature_table import sync_architecture_doc


def test_feature_count_is_66():
    """
    F-016: FEATURE_COLUMNS must have exactly 66 features.
    """
    assert len(FEATURE_COLUMNS) == 66


def test_architecture_doc_feature_table_and_wording_parity():
    """
    F-016 & F-012 doc: Architecture documentation must be in sync with FEATURE_COLUMNS,
    have no file:/// links, no unprovable claims, and honest Dixon-Coles wording.
    """
    assert sync_architecture_doc(check_only=True) is True
