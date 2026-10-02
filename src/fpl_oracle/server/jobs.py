"""
Background job scheduler using APScheduler.
Periodically refreshes live gameweek stats and checks event-status.
"""

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from fpl_oracle.api.fpl_client import fpl_client

logger = logging.getLogger("fpl_oracle.jobs")

scheduler = AsyncIOScheduler()

async def refresh_fpl_data_job():
    logger.info("[Scheduler] Refreshing FPL bootstrap and event status...")
    try:
        await fpl_client.get_bootstrap_static(force_refresh=True)
        curr, nxt = await fpl_client.get_current_and_next_gw()
        if curr:
            await fpl_client.get_live_gameweek(curr)
        logger.info(f"[Scheduler] Background refresh complete (Current GW: {curr}, Next: {nxt}).")
    except Exception as e:
        logger.warning(f"[Scheduler] Background refresh error: {e}")

def start_scheduler():
    # Run refresh every 10 minutes
    scheduler.add_job(refresh_fpl_data_job, "interval", minutes=10, id="fpl_refresh")
    scheduler.start()
    logger.info("Background job scheduler started.")

def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown()
