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
    return dict(
        status="awaiting_forward_evidence" if not eligible else "descriptive_forward_evidence",
        count=len(eligible),
        distinct_gameweeks=len({r["target_gameweek"] for r in eligible}),
        brier=sum(r["brier"] for r in eligible) / len(eligible) if eligible else None,
        reliability_bins=bins,
        calibrated=False,
        promotion_allowed=False,
        reason="Metrics alone do not prove calibration. Require independent temporal calibration/test blocks and validated baselines before promotion.",
    )
