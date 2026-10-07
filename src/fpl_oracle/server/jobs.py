"""
Background job scheduler using APScheduler.
Coordinates high-cadence live gameweek stats polling, periodic news ingestion,
hourly price snapshot logging, post-GW automated retrain triggers with rollback,
and pre-deadline briefing refresh alerts.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.api.game_state import GameweekPhase, game_state_manager
from fpl_oracle.briefing.weekly import weekly_briefing_generator
from fpl_oracle.data.store import data_store
from fpl_oracle.ml.model_registry import model_registry
from fpl_oracle.news.ingest import news_ingestion
from fpl_oracle.optimise.price_change import price_change_predictor

logger = logging.getLogger("fpl_oracle.jobs")

scheduler = AsyncIOScheduler()


class JobExecutionTracker:
    def __init__(self):
        self.jobs_state: dict[str, dict[str, Any]] = {
            "cadence_refresh": {
                "name": "Live / Cadence Data Refresh",
                "schedule": "60s (live) / 10m (normal)",
                "status": "IDLE",
                "last_run": None,
                "next_run": None,
                "run_count": 0,
                "duration_seconds": 0.0,
                "details": "Awaiting initial execution",
            },
            "news_refresh": {
                "name": "News & RSS Feed Ingestion",
                "schedule": "Every 20 minutes",
                "status": "IDLE",
                "last_run": None,
                "next_run": None,
                "run_count": 0,
                "duration_seconds": 0.0,
                "details": "Awaiting initial execution",
            },
            "price_snapshot": {
                "name": "Hourly Price Change Snapshot",
                "schedule": "Hourly",
                "status": "IDLE",
                "last_run": None,
                "next_run": None,
                "run_count": 0,
                "duration_seconds": 0.0,
                "details": "Awaiting initial execution",
            },
            "history_refresh": {
                "name": "Finalized Current-Season History Refresh",
                "schedule": "Hourly",
                "status": "IDLE",
                "last_run": None,
                "next_run": None,
                "run_count": 0,
                "duration_seconds": 0.0,
                "details": "Awaiting initial execution",
            },
            "retrain_trigger": {
                "name": "Finalized-GW Candidate Retrain",
                "schedule": "Every 15 minutes",
                "status": "IDLE",
                "last_run": None,
                "next_run": None,
                "run_count": 0,
                "duration_seconds": 0.0,
                "details": "Awaiting initial execution",
            },
            "deadline_alert": {
                "name": "Pre-Deadline Multi-Stage Alert & Briefing",
                "schedule": "Every 5 minutes (-24h, -3h, -1h)",
                "status": "IDLE",
                "last_run": None,
                "next_run": None,
                "run_count": 0,
                "duration_seconds": 0.0,
                "details": "Awaiting initial execution",
            },
            "holdout_forward": {
                "name": "Forward Holdout Pre-Deadline Freeze & Post-GW Scoring",
                "schedule": "Every 15 minutes",
                "status": "IDLE",
                "last_run": None,
                "next_run": None,
                "run_count": 0,
                "duration_seconds": 0.0,
                "details": "Awaiting initial execution",
            },
        }
        self.triggered_deadline_alerts: set[tuple[int, str]] = set()
        self.retrained_gameweeks: set[int] = set()

    def update_job_start(self, job_id: str):
        if job_id in self.jobs_state:
            self.jobs_state[job_id]["status"] = "RUNNING"
            self.jobs_state[job_id]["last_run"] = datetime.now(UTC).isoformat()

    def update_job_finish(self, job_id: str, success: bool, duration: float, details: str):
        if job_id in self.jobs_state:
            self.jobs_state[job_id]["status"] = "SUCCESS" if success else "FAILED"
            self.jobs_state[job_id]["duration_seconds"] = round(duration, 2)
            self.jobs_state[job_id]["run_count"] += 1
            self.jobs_state[job_id]["details"] = details


tracker = JobExecutionTracker()


# ====================================================================
# JOB 1: Live Cadence Data Refresh
# 60s during live matches, 10m otherwise
# ====================================================================
async def cadence_refresh_job():
    job_id = "cadence_refresh"
    job_name = tracker.jobs_state[job_id]["name"]
    start_t = datetime.now(UTC)
    run_id = data_store.record_job_start(job_id, job_name)
    tracker.update_job_start(job_id)

    try:
        game_state = await game_state_manager.get_game_state()
        is_live = game_state.phase in (GameweekPhase.LIVE, GameweekPhase.BONUS_PENDING)

        # Force refresh bootstrap and fixtures
        await fpl_client.get_bootstrap_static(force_refresh=True)
        await fpl_client.get_fixtures(force_refresh=True)

        curr, nxt = await fpl_client.get_current_and_next_gw()
        if is_live and curr:
            await fpl_client.get_live_gameweek(curr)

        dur = (datetime.now(UTC) - start_t).total_seconds()
        details = f"Refreshed bootstrap & fixtures (Phase: {game_state.phase.value}, Live: {is_live}, GW: {curr})"
        data_store.record_job_finish(run_id, "SUCCESS", dur, details)
        tracker.update_job_finish(job_id, True, dur, details)
        logger.info(f"[Scheduler] {job_name} complete in {dur:.2f}s: {details}")
    except Exception as e:
        dur = (datetime.now(UTC) - start_t).total_seconds()
        err_msg = f"Error during refresh: {str(e)}"
        data_store.record_job_finish(run_id, "FAILED", dur, err_msg)
        tracker.update_job_finish(job_id, False, dur, err_msg)
        logger.warning(f"[Scheduler] {job_name} failed: {e}")


# ====================================================================
# JOB 2: News & RSS Feed Ingestion
# Runs every 20 minutes
# ====================================================================
async def news_refresh_job():
    job_id = "news_refresh"
    job_name = tracker.jobs_state[job_id]["name"]
    start_t = datetime.now(UTC)
    run_id = data_store.record_job_start(job_id, job_name)
    tracker.update_job_start(job_id)

    try:
        articles = await news_ingestion.fetch_rss_articles()
        news_status = news_ingestion.get_news_status()

        dur = (datetime.now(UTC) - start_t).total_seconds()
        details = (
            f"Ingested {len(articles)} articles from {news_status['active_feeds']} active feeds "
            f"({news_status['dead_feeds_count']} dead feeds dropped)"
        )
        data_store.record_job_finish(run_id, "SUCCESS", dur, details)
        tracker.update_job_finish(job_id, True, dur, details)
        logger.info(f"[Scheduler] {job_name} complete in {dur:.2f}s: {details}")
    except Exception as e:
        dur = (datetime.now(UTC) - start_t).total_seconds()
        err_msg = f"News ingestion error: {str(e)}"
        data_store.record_job_finish(run_id, "FAILED", dur, err_msg)
        tracker.update_job_finish(job_id, False, dur, err_msg)
        logger.warning(f"[Scheduler] {job_name} failed: {e}")


# ====================================================================
# JOB 3: Hourly Price Change Snapshot
# Runs hourly
# ====================================================================
async def price_snapshot_job():
    job_id = "price_snapshot"
    job_name = tracker.jobs_state[job_id]["name"]
    start_t = datetime.now(UTC)
    run_id = data_store.record_job_start(job_id, job_name)
    tracker.update_job_start(job_id)

    try:
        boot, _ = await fpl_client.get_bootstrap_static()
        predictions = price_change_predictor.analyze_price_changes(boot)

        # Store snapshots of top 40 volatile players
        top_volatile = predictions[:40]
        data_store.save_price_snapshots(top_volatile)

        rises = sum(1 for p in predictions if p["direction"] in ("RISE_IMMINENT", "LIKELY_RISE"))
        falls = sum(1 for p in predictions if p["direction"] in ("FALL_IMMINENT", "LIKELY_FALL"))

        dur = (datetime.now(UTC) - start_t).total_seconds()
        details = f"Logged price snapshot for {len(top_volatile)} players (Rising: {rises}, Falling: {falls})"
        data_store.record_job_finish(run_id, "SUCCESS", dur, details)
        tracker.update_job_finish(job_id, True, dur, details)
        logger.info(f"[Scheduler] {job_name} complete in {dur:.2f}s: {details}")
    except Exception as e:
        dur = (datetime.now(UTC) - start_t).total_seconds()
        err_msg = f"Price snapshot error: {str(e)}"
        data_store.record_job_finish(run_id, "FAILED", dur, err_msg)
        tracker.update_job_finish(job_id, False, dur, err_msg)
        logger.warning(f"[Scheduler] {job_name} failed: {e}")


# ====================================================================
# JOB 4: Retrain Trigger on Event Status Bonus Finalized
# Checks every 15 minutes
# ====================================================================
async def retrain_trigger_job():
    job_id = "retrain_trigger"
    job_name = tracker.jobs_state[job_id]["name"]
    start_t = datetime.now(UTC)
    run_id = data_store.record_job_start(job_id, job_name)
    tracker.update_job_start(job_id)

    try:
        from fpl_oracle.server.retraining import retrain_finalized_gameweeks

        details = await retrain_finalized_gameweeks()

        dur = (datetime.now(UTC) - start_t).total_seconds()
        data_store.record_job_finish(run_id, "SUCCESS", dur, details)
        tracker.update_job_finish(job_id, True, dur, details)
        logger.info(f"[Scheduler] {job_name}: {details}")
    except Exception as e:
        dur = (datetime.now(UTC) - start_t).total_seconds()
        err_msg = f"Retrain trigger error: {str(e)}"
        data_store.record_job_finish(run_id, "FAILED", dur, err_msg)
        tracker.update_job_finish(job_id, False, dur, err_msg)
        logger.warning(f"[Scheduler] {job_name} failed: {e}")


# ====================================================================
# JOB 5: Pre-Deadline Alert & Briefing Regeneration
# Checks every 5 minutes for -24h, -3h, -1h thresholds
# ====================================================================
async def deadline_alert_job():
    job_id = "deadline_alert"
    job_name = tracker.jobs_state[job_id]["name"]
    start_t = datetime.now(UTC)
    run_id = data_store.record_job_start(job_id, job_name)
    tracker.update_job_start(job_id)

    try:
        game_state = await game_state_manager.get_game_state()
        _, next_gw = await fpl_client.get_current_and_next_gw()
        target_gw = next_gw or 6
        sec = game_state.seconds_to_deadline

        # Define alert windows: (window_tag, min_sec, max_sec)
        windows = [
            ("24h_window", 82800, 86400),  # 23h to 24h
            ("3h_window", 7200, 10800),  # 2h to 3h
            ("1h_window", 0, 3600),  # Final 60 minutes
        ]

        triggered_window = None
        for tag, low, high in windows:
            if low <= sec <= high:
                key = (target_gw, tag)
                if key not in tracker.triggered_deadline_alerts:
                    triggered_window = tag
                    tracker.triggered_deadline_alerts.add(key)
                    break

        if triggered_window:
            logger.info(
                f"[Scheduler] Pre-deadline alert triggered: GW{target_gw} ({triggered_window}). Regenerating briefing..."
            )
            # Refresh live data & regenerate briefing
            await fpl_client.get_bootstrap_static(force_refresh=True)
            await news_ingestion.fetch_rss_articles()
            await weekly_briefing_generator.generate_briefing()
            details = f"Alert {triggered_window} executed for GW{target_gw}: Refreshed briefing & team news."
        else:
            details = (
                f"GW{target_gw} deadline countdown: {sec / 3600.0:.1f} hours remaining. No alert threshold reached."
            )

        dur = (datetime.now(UTC) - start_t).total_seconds()
        data_store.record_job_finish(run_id, "SUCCESS", dur, details)
        tracker.update_job_finish(job_id, True, dur, details)
    except Exception as e:
        dur = (datetime.now(UTC) - start_t).total_seconds()
        err_msg = f"Deadline alert error: {str(e)}"
        data_store.record_job_finish(run_id, "FAILED", dur, err_msg)
        tracker.update_job_finish(job_id, False, dur, err_msg)
        logger.warning(f"[Scheduler] {job_name} failed: {e}")


# ====================================================================
# JOB 6: Forward Holdout Pre-Deadline Freeze & Post-GW Scoring (Requirement G6)
# Checks every 15 minutes to freeze predictions or score finished GW
# ====================================================================
async def history_refresh_job():
    from fpl_oracle.data.historical import historical_manager

    job_id = "history_refresh"
    start = datetime.now(UTC)
    run_id = data_store.record_job_start(job_id, tracker.jobs_state[job_id]["name"])
    tracker.update_job_start(job_id)
    try:
        result = await historical_manager.refresh_current_season()
        duration = (datetime.now(UTC) - start).total_seconds()
        success = bool(result.get("complete"))
        details = str(result)
        data_store.record_job_finish(run_id, "SUCCESS" if success else "FAILED", duration, details)
        tracker.update_job_finish(job_id, success, duration, details)
        return result
    except Exception as error:
        duration = (datetime.now(UTC) - start).total_seconds()
        data_store.record_job_finish(run_id, "FAILED", duration, str(error))
        tracker.update_job_finish(job_id, False, duration, str(error))
        raise


async def holdout_forward_job():
    job_id = "holdout_forward"
    job_name = tracker.jobs_state[job_id]["name"]
    start_t = datetime.now(UTC)
    run_id = data_store.record_job_start(job_id, job_name)
    tracker.update_job_start(job_id)

    try:
        from fpl_oracle.ml.holdout import ensure_holdout_log_initialized, freeze_predictions, score_frozen_predictions

        ensure_holdout_log_initialized()

        game_state = await game_state_manager.get_game_state()
        curr_gw, next_gw = await fpl_client.get_current_and_next_gw()
        target_freeze_gw = next_gw

        details = ""
        # 1. Pre-deadline freeze if within 24h
        if (
            target_freeze_gw is not None
            and game_state.seconds_to_deadline is not None
            and 0 < game_state.seconds_to_deadline <= 86400
            and not game_state.stale
        ):
            payload = await freeze_predictions(target_gw=target_freeze_gw)
            details = f"Froze GW{target_freeze_gw} predictions ({payload['player_count']} players)."
        else:
            details = f"Awaiting valid pre-deadline window for GW{target_freeze_gw}."

        # 2. Check if previous gameweek finished and can be scored
        if curr_gw and curr_gw >= 6:
            from fpl_oracle.ml.holdout import fetch_finalized_actuals

            actuals = await fetch_finalized_actuals(curr_gw)
            scored = score_frozen_predictions(gw=curr_gw, actual_points_map=actuals) if actuals is not None else None
            if scored:
                details += f" Scored finished GW{curr_gw} holdout (MAE={scored['ml_mae']})."

        dur = (datetime.now(UTC) - start_t).total_seconds()
        data_store.record_job_finish(run_id, "SUCCESS", dur, details)
        tracker.update_job_finish(job_id, True, dur, details)
    except Exception as e:
        dur = (datetime.now(UTC) - start_t).total_seconds()
        err_msg = f"Holdout forward error: {str(e)}"
        data_store.record_job_finish(run_id, "FAILED", dur, err_msg)
        tracker.update_job_finish(job_id, False, dur, err_msg)
        logger.warning(f"[Scheduler] {job_name} failed: {e}")


def get_jobs_status() -> dict[str, Any]:
    """Return execution status of all scheduler background jobs."""
    jobs_summary = []
    for j_id, j_info in tracker.jobs_state.items():
        job_obj = scheduler.get_job(j_id)
        next_run_iso = None
        if job_obj and job_obj.next_run_time:
            next_run_iso = job_obj.next_run_time.isoformat()

        jobs_summary.append(
            {
                "id": j_id,
                "name": j_info["name"],
                "schedule": j_info["schedule"],
                "status": j_info["status"],
                "last_run": j_info["last_run"],
                "next_run": next_run_iso,
                "run_count": j_info["run_count"],
                "duration_seconds": j_info["duration_seconds"],
                "details": j_info["details"],
            }
        )

    active_model = model_registry.get_active_version()
    news_stat = news_ingestion.get_news_status()

    return {
        "scheduler_running": scheduler.running,
        "total_jobs": len(jobs_summary),
        "jobs": jobs_summary,
        "active_model": active_model,
        "news_status": news_stat,
        "recent_runs": data_store.get_job_runs(limit=15),
    }


async def run_job_on_demand(job_id: str) -> dict[str, Any]:
    """Manually trigger any background job immediately."""
    job_map: dict[str, Callable[[], Any]] = {
        "cadence_refresh": cadence_refresh_job,
        "news_refresh": news_refresh_job,
        "price_snapshot": price_snapshot_job,
        "retrain_trigger": retrain_trigger_job,
        "deadline_alert": deadline_alert_job,
        "holdout_forward": holdout_forward_job,
        "history_refresh": history_refresh_job,
    }

    if job_id not in job_map:
        return {"success": False, "error": f"Unknown job_id: '{job_id}'"}

    func = job_map[job_id]
    await func()
    state = tracker.jobs_state.get(job_id, {})
    return {
        "success": state.get("status") == "SUCCESS",
        "job_id": job_id,
        "status": state.get("status"),
        "duration_seconds": state.get("duration_seconds"),
        "details": state.get("details"),
    }


def start_scheduler():
    """Register and start all background jobs."""
    if not scheduler.running:
        # Job 1: Cadence data refresh every 10 minutes
        scheduler.add_job(cadence_refresh_job, "interval", minutes=10, id="cadence_refresh", replace_existing=True)

        # Job 2: News & RSS feed ingestion every 20 minutes
        scheduler.add_job(news_refresh_job, "interval", minutes=20, id="news_refresh", replace_existing=True)

        # Job 3: Hourly price snapshot
        scheduler.add_job(price_snapshot_job, "interval", hours=1, id="price_snapshot", replace_existing=True)

        # Job 4: Post-GW retrain check every 15 minutes
        scheduler.add_job(retrain_trigger_job, "interval", minutes=15, id="retrain_trigger", replace_existing=True)

        # Job 5: Pre-deadline alert & briefing check every 5 minutes
        scheduler.add_job(deadline_alert_job, "interval", minutes=5, id="deadline_alert", replace_existing=True)

        scheduler.add_job(history_refresh_job, "interval", hours=1, id="history_refresh", replace_existing=True)

        # Job 6: Forward Holdout freeze and scoring every 15 minutes
        scheduler.add_job(holdout_forward_job, "interval", minutes=15, id="holdout_forward", replace_existing=True)

        try:
            scheduler.start()
            logger.info("[Scheduler] All 7 automated background jobs registered and scheduler started.")
        except RuntimeError:
            logger.warning(
                "[Scheduler] No running asyncio event loop. Scheduler jobs registered; starting deferred until event loop starts."
            )


def stop_scheduler():
    """Gracefully terminate background scheduler."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("[Scheduler] Background scheduler stopped.")
