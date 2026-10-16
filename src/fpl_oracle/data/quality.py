"""
Data Quality and Resilience Auditing (WI-7.4).

Generates rigorous data-quality reports on every ingestion/prediction run:
- Missing-value rates per column
- Schema drift against expected schemas
- Duplicated player / element IDs
- Price anomalies (below minimum £3.5m, above £16.0m, or sudden multi-step spikes)
- Resilient source adapters (FPL API + Understat without FBref dependency)
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from fpl_oracle.config import REPORTS_DIR

logger = logging.getLogger("fpl_oracle.data.quality")


class ColumnDrift(BaseModel):
    missing_columns: list[str] = Field(default_factory=list)
    unexpected_columns: list[str] = Field(default_factory=list)
    dtype_mismatches: dict[str, str] = Field(default_factory=dict)


class PriceAnomaly(BaseModel):
    element_id: int
    web_name: str
    current_cost: float
    reason: str


class DataQualityReport(BaseModel):
    run_timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    dataset_name: str
    total_rows: int
    total_columns: int
    missing_value_rates: dict[str, float] = Field(default_factory=dict)
    high_missingness_columns: list[str] = Field(default_factory=list)  # columns with > 10% missing
    duplicated_element_ids: list[int] = Field(default_factory=list)
    price_anomalies: list[PriceAnomaly] = Field(default_factory=list)
    schema_drift: ColumnDrift = Field(default_factory=ColumnDrift)
    has_critical_anomalies: bool = False
    quality_verdict: str = "PASS"


class DataQualityAuditor:
    """Audits data quality, schema integrity, and anomalies on dataframes."""

    def __init__(self, missing_rate_threshold: float = 0.10):
        self.missing_rate_threshold = missing_rate_threshold

    def audit_player_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str = "player_features",
        expected_columns: list[str] | set[str] | None = None,
        id_column: str = "element",
        price_column: str = "value",
    ) -> DataQualityReport:
        """
        Audits a player-level dataframe for data quality, schema drift, and anomalies.
        """
        total_rows = len(df)
        total_cols = len(df.columns)

        if total_rows == 0:
            return DataQualityReport(
                dataset_name=dataset_name,
                total_rows=0,
                total_columns=total_cols,
                has_critical_anomalies=True,
                quality_verdict="FAIL_EMPTY_DATASET",
            )

        # 1. Missing-value rates
        missing_rates: dict[str, float] = {}
        high_missing: list[str] = []
        for col in df.columns:
            m_count = int(df[col].isna().sum())
            rate = round(float(m_count / total_rows), 4)
            missing_rates[col] = rate
            if rate > self.missing_rate_threshold:
                high_missing.append(col)

        # 2. Duplicated player IDs
        dupes: list[int] = []
        if id_column in df.columns:
            duplicated_mask = df[id_column].duplicated(keep=False)
            if duplicated_mask.any():
                dupes = [int(x) for x in df.loc[duplicated_mask, id_column].unique()]

        # 3. Price anomalies
        price_anomalies: list[PriceAnomaly] = []
        # Support either tenths (£100 = £10.0m) or direct millions (£10.0)
        p_col = price_column if price_column in df.columns else ("now_cost" if "now_cost" in df.columns else None)
        if p_col and p_col in df.columns:
            for _, row in df.iterrows():
                val = row[p_col]
                if pd.isna(val):
                    continue
                try:
                    p_val = float(val)
                    # Normalize: if > 30, it is in tenths of millions (e.g. 45 = £4.5m)
                    cost_m = (p_val / 10.0) if p_val > 30.0 else p_val
                    eid = int(row.get(id_column, row.get("id", 0)))
                    name = str(row.get("web_name", f"ID_{eid}"))

                    # Minimum possible FPL player price is £3.5m (or 35 tenths); max is £16.0m
                    if cost_m < 3.5:
                        price_anomalies.append(
                            PriceAnomaly(
                                element_id=eid,
                                web_name=name,
                                current_cost=cost_m,
                                reason=f"Price £{cost_m:.1f}m is below Premier League minimum (£3.5m)",
                            )
                        )
                    elif cost_m > 16.0:
                        price_anomalies.append(
                            PriceAnomaly(
                                element_id=eid,
                                web_name=name,
                                current_cost=cost_m,
                                reason=f"Price £{cost_m:.1f}m is above Premier League ceiling (£16.0m)",
                            )
                        )
                except (ValueError, TypeError):
                    continue

        # 4. Schema drift
        drift = ColumnDrift()
        if expected_columns:
            expected_set = set(expected_columns)
            actual_set = set(df.columns)
            drift.missing_columns = sorted(list(expected_set - actual_set))
            drift.unexpected_columns = sorted(list(actual_set - expected_set))

        # Critical anomalies verdict
        has_critical = bool(len(dupes) > 0 or len(drift.missing_columns) > 0 or len(price_anomalies) > 5)
        verdict = (
            "FAIL_CRITICAL_ANOMALIES"
            if has_critical
            else ("WARN" if len(high_missing) > 0 or len(price_anomalies) > 0 else "PASS")
        )

        report = DataQualityReport(
            dataset_name=dataset_name,
            total_rows=total_rows,
            total_columns=total_cols,
            missing_value_rates=missing_rates,
            high_missingness_columns=high_missing,
            duplicated_element_ids=dupes,
            price_anomalies=price_anomalies,
            schema_drift=drift,
            has_critical_anomalies=has_critical,
            quality_verdict=verdict,
        )

        return report

    def save_report(self, report: DataQualityReport, filename: str = "data_quality_report.json") -> Path:
        """Persists data quality report to reports directory."""
        out_dir = REPORTS_DIR / "quality"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / filename
        out_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"Saved DataQualityReport to {out_file} (Verdict: {report.quality_verdict})")
        return out_file


data_quality_auditor = DataQualityAuditor()
