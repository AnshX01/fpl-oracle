"""Forward-only league probability evidence. Counterfactual plans have no observed label."""

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path


def freeze_forecast(payload, directory: Path, now=None):
    now = now or datetime.now(UTC)
    deadline = datetime.fromisoformat(payload["deadline_utc"].replace("Z", "+00:00"))
    if deadline.tzinfo is None or now >= deadline:
        raise ValueError("Forecast must be recorded before the actual deadline")
    if payload.get("source_kind") != "official_forward" or not payload.get("snapshot_id"):
        raise ValueError("Forward provenance missing")
    if not payload.get("observed_manager_ids") or len(payload["observed_manager_ids"]) < 2:
        raise ValueError("At least owner and one observed rival required")
    ids = payload["observed_manager_ids"]
    if len(ids) != len(set(ids)) or payload.get("owner_manager_id") not in ids:
        raise ValueError("Manager set must be unique and include owner")
    probability = float(payload["probability"])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid probability")
    record = dict(payload, frozen_at=now.isoformat(), schema_version=1, calibrated=False)
    identifier = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{identifier}.json"
    if not path.exists():
        path.write_text(json.dumps(record, indent=2))
    return dict(path=str(path), forecast_id=identifier)


def score_forecast(record, finalized_scores):
    if finalized_scores.get("source_kind") != "official_finalized_manager_history":
        raise ValueError("Finalized official history required")
    if finalized_scores.get("gameweek") != record["target_gameweek"]:
        raise ValueError("Outcome is for another gameweek")
    if record.get("scope") != "actual_observed_manager_outcome":
        return dict(status="not_scoreable", reason="Counterfactual plan has no actual observed outcome")
    if record.get("schema_version") != 1 or record.get("source_kind") != "official_forward":
        raise ValueError("Unsupported forecast provenance")
    frozen = datetime.fromisoformat(record["frozen_at"].replace("Z", "+00:00"))
    deadline = datetime.fromisoformat(record["deadline_utc"].replace("Z", "+00:00"))
    if frozen.tzinfo is None or deadline.tzinfo is None or frozen >= deadline:
        raise ValueError("Not pre-deadline evidence")
    scores = finalized_scores["scores"]
    ids = record["observed_manager_ids"]
    owner = record["owner_manager_id"]
    if not set(ids).issubset(scores):
        raise ValueError("Missing outcome for observed rival set")
    # Forecast event is strict first place; ties are not silently counted as wins.
    outcome = int(all(scores[owner] > scores[r] for r in ids if r != owner))
    p = float(record["probability"])
    return dict(
        status="scored",
        probability=p,
        outcome=outcome,
        brier=(p - outcome) ** 2,
        baseline_brier=(float(record["persistence_baseline_probability"]) - outcome) ** 2
        if "persistence_baseline_probability" in record
        else None,
        target_gameweek=record["target_gameweek"],
        source_kind="official_forward_scored",
    )


def validation_summary(scored):
    eligible = [
        row for row in scored if row.get("source_kind") == "official_forward_scored" and row.get("status") == "scored"
    ]
    bins = []
    for index in range(5):
        rows = [
            r
            for r in eligible
            if index / 5 <= r["probability"] <= (index + 1) / 5 and (index == 4 or r["probability"] < (index + 1) / 5)
        ]
        if rows:
            bins.append(
                dict(
                    lower=index / 5,
                    upper=(index + 1) / 5,
                    count=len(rows),
                    forecast_mean=sum(r["probability"] for r in rows) / len(rows),
                    observed_rate=sum(r["outcome"] for r in rows) / len(rows),
                )
            )
    paired = [r for r in eligible if r.get("baseline_brier") is not None]
    return dict(
        status="awaiting_forward_evidence" if not eligible else "descriptive_forward_evidence",
        count=len(eligible),
        distinct_gameweeks=len({r["target_gameweek"] for r in eligible}),
        brier=sum(r["brier"] for r in eligible) / len(eligible) if eligible else None,
        reliability_bins=bins,
        paired_baseline_count=len(paired),
        paired_brier_gain=sum(r["baseline_brier"] - r["brier"] for r in paired) / len(paired) if paired else None,
        calibrated=False,
        promotion_allowed=False,
        reason="Metrics alone do not prove calibration. Require independent temporal calibration/test blocks and validated baselines before promotion.",
    )
