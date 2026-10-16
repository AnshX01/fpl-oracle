"""
Tests for Config Consolidation and Data-Source Resilience (Phase 7: WI-7.1 & WI-7.4).

Acceptance Criteria:
1. WI-7.1: Strict Pydantic schema validation on all YAML configurations;
   all thresholds carry explicit citations to backtest or rules.
2. WI-7.4: Data-quality auditor calculates missingness, schema drift, duplicated player IDs,
   and price anomalies (< £3.5m or > £16.0m) and exports audit report.
"""

import json

import pandas as pd

from fpl_oracle.config import validate_configuration_schemas
from fpl_oracle.data.quality import DataQualityAuditor, data_quality_auditor


def test_wi71_all_configs_pass_strict_schema_validation():
    """WI-7.1: All committed YAML configuration files pass Pydantic schema validation."""
    results = validate_configuration_schemas()
    assert results.get("settings.yaml") is True
    assert results.get("flags.yaml") is True
    assert results.get("rules.yaml") is True
    assert results.get("news_sources.yaml") is True


def test_wi74_data_quality_clean_dataset_passes():
    """WI-7.4: A valid player dataset passes data quality audit without critical anomalies."""
    df = pd.DataFrame(
        [
            {"element": 1, "web_name": "Raya", "position": "GKP", "value": 55, "expected_points": 4.5},
            {"element": 2, "web_name": "Saliba", "position": "DEF", "value": 60, "expected_points": 5.0},
            {"element": 3, "web_name": "Saka", "position": "MID", "value": 100, "expected_points": 7.5},
            {"element": 4, "web_name": "Haaland", "position": "FWD", "value": 150, "expected_points": 9.2},
        ]
    )

    report = data_quality_auditor.audit_player_dataset(
        df=df,
        dataset_name="clean_test_dataset",
        expected_columns=["element", "web_name", "position", "value", "expected_points"],
    )

    assert report.total_rows == 4
    assert report.has_critical_anomalies is False
    assert report.quality_verdict == "PASS"
    assert len(report.duplicated_element_ids) == 0
    assert len(report.price_anomalies) == 0
    assert len(report.schema_drift.missing_columns) == 0


def test_wi74_data_quality_detects_anomalies_and_drift():
    """
    WI-7.4: Auditor correctly catches:
    1. Duplicated player IDs
    2. Missing columns (schema drift)
    3. Price anomalies (< £3.5m and > £16.0m)
    4. Missing-value rates exceeding 10%
    """
    df = pd.DataFrame(
        [
            {"element": 1, "web_name": "P1", "value": 50, "expected_points": 4.0},
            {"element": 1, "web_name": "P1_Dupe", "value": 50, "expected_points": 4.0},  # Duplicated ID!
            {"element": 2, "web_name": "CheapAnomaly", "value": 20, "expected_points": 2.0},  # £2.0m < £3.5m minimum!
            {"element": 3, "web_name": "ExpensiveAnomaly", "value": 180, "expected_points": 10.0},  # £18.0m > £16.0m max!
            {"element": 4, "web_name": "MissingXp", "value": 60, "expected_points": None},  # 1/5 missing = 20% > 10%
            {"element": 5, "web_name": "NormalP", "value": 65, "expected_points": 5.0},
        ]
    )

    report = data_quality_auditor.audit_player_dataset(
        df=df,
        dataset_name="anomalous_dataset",
        expected_columns=["element", "web_name", "value", "expected_points", "required_but_missing_col"],
    )

    assert report.total_rows == 6
    assert report.has_critical_anomalies is True
    assert "FAIL" in report.quality_verdict

    # Check duplicated IDs
    assert 1 in report.duplicated_element_ids

    # Check price anomalies
    anomaly_names = [a.web_name for a in report.price_anomalies]
    assert "CheapAnomaly" in anomaly_names
    assert "ExpensiveAnomaly" in anomaly_names

    # Check schema drift
    assert "required_but_missing_col" in report.schema_drift.missing_columns

    # Check high missingness column
    assert "expected_points" in report.high_missingness_columns
    assert report.missing_value_rates["expected_points"] > 0.10


def test_wi74_data_quality_report_persistence(tmp_path, monkeypatch):
    """WI-7.4: DataQualityAuditor serializes quality reports to disk."""
    monkeypatch.setattr("fpl_oracle.data.quality.REPORTS_DIR", tmp_path)

    df = pd.DataFrame([{"element": 10, "web_name": "TestP", "value": 50}])
    auditor = DataQualityAuditor()
    report = auditor.audit_player_dataset(df, dataset_name="persist_test")

    out_file = auditor.save_report(report, filename="test_dq.json")
    assert out_file.exists()

    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert saved["dataset_name"] == "persist_test"
    assert saved["total_rows"] == 1
    assert saved["quality_verdict"] == "PASS"
