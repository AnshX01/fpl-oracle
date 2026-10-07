"""Automatic pre-deadline observed-manager forecast and finalized outcome scoring.

The forecast assumes prior published picks are held. Real future manager decisions
are part of its eventual forecast error, never substituted counterfactual labels.
"""

import asyncio
import json
from datetime import UTC, datetime

import numpy as np

from fpl_oracle.league.validation import freeze_forecast, score_forecast, validation_summary


async def capture_forward(target_gw, directory=None, samples=512):
    from fpl_oracle.api.fpl_client import fpl_client
    from fpl_oracle.config import BASE_DIR
    from fpl_oracle.data.store import data_store
    from fpl_oracle.league.montecarlo import monte_carlo_simulator
    from fpl_oracle.league.rivals import rival_analyzer
    from fpl_oracle.league.standings import league_standings_manager
    from fpl_oracle.ml.holdout import compute_manifest_sha256
    from fpl_oracle.ml.model_registry import model_registry
    from fpl_oracle.server.analysis import analysis_service

    directory = directory or BASE_DIR / "data" / "league_forward"
    profile = data_store.get_profile()
    if not profile.manager_id or not profile.target_league_id:
        return dict(status="unconfigured")
    # One immutable forecast per owner/league/GW, independent of legacy player freezes.
    directory.mkdir(parents=True, exist_ok=True)
    index = directory / f"gw{target_gw}-{profile.manager_id}-{profile.target_league_id}.index.json"
    if index.exists():
        return json.loads(index.read_text())
    boot, stale = await fpl_client.get_bootstrap_static(force_refresh=True)
    fixtures, fs = await fpl_client.get_fixtures(force_refresh=True)
    event = next((e for e in boot.events if e.id == target_gw), None)
    if stale or fs or not event or not event.deadline_time:
        raise ValueError("Fresh official deadline/fixtures required")
    deadline = datetime.fromisoformat(event.deadline_time.replace("Z", "+00:00"))
    if datetime.now(UTC) >= deadline:
        raise ValueError("Forward capture is too late")
    standings = await league_standings_manager.get_league_standings(profile.target_league_id)
    if standings.get("coverage", {}).get("partial"):
        raise ValueError("Partial standings not accepted")
    owner = next((r for r in standings.get("standings", []) if r.get("entry") == profile.manager_id), None)
    if not owner:
        raise ValueError("Owner absent from official league")
    analysis = await rival_analyzer.analyze_rivals(standings["standings"], profile.manager_id, target_gw, boot)
    rivals = analysis.get("rival_squads", [])
    if not rivals:
        raise ValueError("No observed rival picks")
    observed = min(r["observed_gameweek"] for r in rivals)
    picks, ps = await fpl_client.get_manager_picks(profile.manager_id, observed, force_refresh=True)
    if ps:
        raise ValueError("Stale owner picks")
    rosters = [
        dict(
            id=profile.manager_id,
            points=owner["total"],
            starters=[p.element for p in picks.picks if p.position <= 11],
            captain=next(p.element for p in picks.picks if p.is_captain),
        )
    ]
    for rival in rivals:
        verified, rival_stale = await fpl_client.get_manager_picks(rival["entry_id"], observed, force_refresh=True)
        if rival_stale:
            raise ValueError("Fresh observed rival picks required")
        rosters.append(
            dict(
                id=rival["entry_id"],
                points=rival["total_points"],
                starters=[p.element for p in verified.picks if p.position <= 11],
                captain=next(p.element for p in verified.picks if p.is_captain),
            )
        )
    if any(len(r["starters"]) != 11 for r in rosters):
        raise ValueError("Observed XI incomplete")
    model_version = model_registry.get_active_version()
    manifest = compute_manifest_sha256()
    if manifest == "unmanifested":
        raise ValueError("Model provenance missing")
    projections = await analysis_service.projections(target_gw, 1, boot, fixtures)
    frame = projections[target_gw]
    pool = {int(r["element"]): r for r in frame.to_dict("records")}
    if any(not set(r["starters"]).issubset(pool) for r in rosters):
        raise ValueError("Projection missing observed player")

    def simulate():
        rng = np.random.default_rng(42)
        wins = 0
        for _ in range(samples):
            draw = monte_carlo_simulator.draw_gameweek(pool, rng)
            totals = [r["points"] + sum(draw[e] for e in r["starters"]) + draw[r["captain"]] for r in rosters]
            wins += int(all(totals[0] > v for v in totals[1:]))
        return wins / samples

    probability = await asyncio.to_thread(simulate)
    if model_registry.get_active_version() != model_version or compute_manifest_sha256() != manifest:
        raise ValueError("Model changed during forward inference")
    payload = dict(
        deadline_utc=event.deadline_time,
        source_kind="official_forward",
        snapshot_id=frame.attrs.get("snapshot_id"),
        model_version=model_version,
        manifest_sha256=manifest,
        owner_manager_id=profile.manager_id,
        league_id=profile.target_league_id,
        observed_manager_ids=[r["id"] for r in rosters],
        observed_gameweek=observed,
        target_gameweek=target_gw,
        probability=probability,
        scope="actual_observed_manager_outcome",
        event_definition="strict cumulative leader among this fixed observed manager set after target GW; ties do not win",
        assumptions="Hold prior published XI/captain; hidden transfers/chips/lineup decisions unmodeled and count in actual forecast error",
        samples=samples,
        persistence_baseline_probability=float(all(rosters[0]["points"] > r["points"] for r in rosters[1:])),
        rosters=rosters,
    )
    saved = freeze_forecast(payload, directory)
    index.write_text(json.dumps(saved, indent=2))
    return saved


async def score_pending(directory=None):
    from fpl_oracle.api.fpl_client import fpl_client
    from fpl_oracle.config import BASE_DIR

    directory = directory or BASE_DIR / "data" / "league_forward"
    if not directory.exists():
        return validation_summary([])
    boot, stale = await fpl_client.get_bootstrap_static(force_refresh=True)
    if stale:
        raise ValueError("Fresh finalization state required")
    scored = []
    for path in directory.glob("*.json"):
        if path.name.endswith(".index.json") or path.name.endswith(".score.json"):
            continue
        record = json.loads(path.read_text())
        gw = record["target_gameweek"]
        destination = path.with_suffix(".score.json")
        if destination.exists():
            scored.append(json.loads(destination.read_text()))
            continue
        event = next((e for e in boot.events if e.id == gw), None)
        if not event or not event.finished or not event.data_checked:
            continue
        fixtures, fs = await fpl_client.get_fixtures(event=gw, force_refresh=True)
        if fs or not fixtures or not all(f.finished for f in fixtures):
            continue
        scores = {}
        for manager in record["observed_manager_ids"]:
            history, hs = await fpl_client.get_manager_history(manager)
            rows = [r for r in history.current if r.event <= gw]
            if hs or not any(r.event == gw for r in rows):
                break
            scores[manager] = next(r.total_points for r in rows if r.event == gw)
        else:
            result = score_forecast(
                record, dict(source_kind="official_finalized_manager_history", gameweek=gw, scores=scores)
            )
            result.update(forecast_id=path.stem, scored_at=datetime.now(UTC).isoformat(), scores=scores)
            destination.write_text(json.dumps(result, indent=2))
            scored.append(result)
    summary = validation_summary(scored)
    (directory / "validation_summary.txt").write_text(json.dumps(summary, indent=2))
    return summary
