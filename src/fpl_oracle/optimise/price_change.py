"""
Price Change Prediction & Urgency Engine.
Uses official API price change metrics, hourly rate, and transfer momentum
to predict imminent rises and falls tonight. Incorporates Wildcard transfer
isolation, transfer velocity, dynamic ownership-based target thresholds,
and time-to-cutoff towards the 01:30 UK nightly price change window.
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from fpl_oracle.api.models import BootstrapStatic


def calculate_hours_to_cutoff(now_dt: datetime | None = None) -> float:
    """
    Compute hours remaining until the official nightly FPL price change window at 01:30 UK time.
    Uses Europe/London timezone to properly handle GMT/BST daylight saving transitions.
    """
    tz_london = ZoneInfo("Europe/London")
    if now_dt is None:
        now_dt = datetime.now(UTC)
    elif now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=UTC)

    now_london = now_dt.astimezone(tz_london)
    # Target 01:30:00 London time
    cutoff_today = now_london.replace(hour=1, minute=30, second=0, microsecond=0)
    if now_london < cutoff_today:
        target_cutoff = cutoff_today
    else:
        target_cutoff = cutoff_today + timedelta(days=1)

    diff_seconds = (target_cutoff - now_london).total_seconds()
    hours = diff_seconds / 3600.0
    return float(np.clip(hours, 0.05, 24.0))


def estimate_wildcard_factor(gw: int | None) -> float:
    """
    Estimate proportion of transfers made under active Wildcards / Free Hits.
    The official FPL price change algorithm discounts/excludes transfers made by managers
    on active Wildcards. During peak Wildcard windows (e.g. GW8-10 autumn international break,
    GW19-20 expiry window, GW30-32 spring chip run), Wildcard usage rises to 18-25%.
    """
    if gw is None or gw <= 0:
        return 0.10
    if 8 <= gw <= 10:
        return 0.22
    elif gw in (7, 11):
        return 0.15
    elif 19 <= gw <= 20:
        return 0.20
    elif 30 <= gw <= 32:
        return 0.22
    elif gw in (29, 33):
        return 0.15
    else:
        return 0.10


def calculate_rise_threshold(ownership_pct: float, total_players: int = 10_500_000) -> float:
    """
    Dynamic price rise target threshold T_rise.
    Based on empirical FPL microstructure: popular players require more transfers to rise,
    scaling with sqrt(ownership) and bounded between 25,000 and 125,000.
    """
    p = max(0.1, min(100.0, float(ownership_pct)))
    frac_ownership = p / 100.0
    k_rise = 0.007
    raw_target = k_rise * total_players * np.sqrt(frac_ownership)
    return float(np.clip(raw_target, 25_000.0, 125_000.0))


def calculate_fall_threshold(ownership_pct: float, status: str = "a", total_players: int = 10_500_000) -> float:
    """
    Dynamic price fall target threshold T_fall.
    Slightly asymmetric compared to rises. Red-flagged, suspended, or locked players
    require significantly higher sales volume or have locked price status.
    """
    p = max(0.1, min(100.0, float(ownership_pct)))
    frac_ownership = p / 100.0
    k_fall = 0.008
    raw_target = k_fall * total_players * (frac_ownership**0.6)
    flag_multiplier = 3.0 if status in ("i", "s", "u") else 1.0
    target = raw_target * flag_multiplier
    return float(np.clip(target, 30_000.0 * flag_multiplier, 140_000.0 * flag_multiplier))


class PriceChangePredictor:
    def __init__(self):
        pass

    def analyze_price_changes(
        self,
        bootstrap: BootstrapStatic,
        now_dt: datetime | None = None,
        wildcard_factor: float | None = None,
    ) -> list[dict[str, Any]]:
        """
        Analyze all elements for price change likelihood tonight.
        Returns sorted list of players near price change thresholds.
        Incorporates Wildcard isolation, transfer velocity, dynamic ownership targets,
        and hours remaining until 01:30 UK cutoff.
        """
        predictions: list[dict[str, Any]] = []

        # Derive current gameweek for dynamic wildcard estimation
        curr_gw = None
        for ev in getattr(bootstrap, "events", []):
            if getattr(ev, "is_current", False):
                curr_gw = ev.id
                break
            elif getattr(ev, "is_next", False) and curr_gw is None:
                curr_gw = max(1, ev.id - 1)

        wc_factor = wildcard_factor if wildcard_factor is not None else estimate_wildcard_factor(curr_gw)
        wc_factor = float(np.clip(wc_factor, 0.0, 0.9))

        total_players = getattr(bootstrap, "total_players", None) or 10_500_000
        hours_to_cutoff = calculate_hours_to_cutoff(now_dt)

        for elem in bootstrap.elements:
            # 1. API native projections if available
            api_prob = None
            if elem.price_change_projections and len(elem.price_change_projections) > 0:
                try:
                    p_val = float(elem.price_change_projections[0].get("projected_percent", 0.0))
                    api_prob = p_val
                except Exception:
                    api_prob = None

            # 2. Raw transfer metrics and ownership
            raw_in = elem.transfers_in_event or 0
            raw_out = elem.transfers_out_event or 0
            raw_net = raw_in - raw_out
            ownership = float(elem.selected_by_percent or 1.0)

            # Hourly transfer velocity
            hourly_rate = float(elem.price_change_hourly_rate or 0.0)

            # 3. Wildcard transfer isolation
            # The official algorithm discounts transfers made by managers with active Wildcards
            effective_net = raw_net * (1.0 - wc_factor)
            effective_velocity = hourly_rate * (1.0 - wc_factor)

            # 4. Projected net transfers at 01:30 UK cutoff
            projected_additional = effective_velocity * hours_to_cutoff
            projected_net_cutoff = effective_net + projected_additional

            # 5. Dynamic target thresholds
            status = getattr(elem, "status", "a")
            t_rise = calculate_rise_threshold(ownership, total_players)
            t_fall = calculate_fall_threshold(ownership, status, total_players)

            # 6. Microstructure score & progress
            if projected_net_cutoff >= 0:
                target_threshold = t_rise
                progress_pct = (projected_net_cutoff / max(1.0, t_rise)) * 100.0
                microstructure_score = float(np.clip(progress_pct, 0.0, 100.0))
            else:
                target_threshold = t_fall
                progress_pct = (-projected_net_cutoff / max(1.0, t_fall)) * 100.0
                microstructure_score = float(-np.clip(progress_pct, 0.0, 100.0))

            # Composite urgency score (-100 to +100)
            if api_prob is not None:
                score = 0.6 * microstructure_score + 0.4 * (api_prob * 10.0)
            else:
                score = microstructure_score

            score = float(np.clip(score, -100.0, 100.0))

            direction = "STABLE"
            urgency = "Can wait"
            is_calibrated = bool(api_prob is not None or hourly_rate != 0)
            if score >= 80.0:
                direction = "RISE_IMMINENT"
                urgency = (
                    "Strong rising transfer momentum; price change projected tonight (unconfirmed)"
                    if is_calibrated
                    else "Strong rising transfer momentum; price change unconfirmed"
                )
            elif score >= 50.0:
                direction = "LIKELY_RISE"
                urgency = "Rising momentum"
            elif score <= -80.0:
                direction = "FALL_IMMINENT"
                urgency = (
                    "Strong falling transfer momentum; price change projected tonight (unconfirmed)"
                    if is_calibrated
                    else "Strong falling transfer momentum; price change unconfirmed"
                )
            elif score <= -50.0:
                direction = "LIKELY_FALL"
                urgency = "Falling momentum"

            predictions.append(
                {
                    "element": elem.id,
                    "web_name": elem.web_name,
                    "team": elem.team,
                    "now_cost": elem.now_cost / 10.0,
                    "net_transfers_event": raw_net,
                    "net_transfers_effective": round(effective_net, 1),
                    "selected_by_percent": ownership,
                    "hourly_rate": hourly_rate,
                    "effective_hourly_rate": round(effective_velocity, 2),
                    "wildcard_factor": round(wc_factor, 3),
                    "hours_to_cutoff": round(hours_to_cutoff, 2),
                    "projected_net_at_cutoff": round(projected_net_cutoff, 1),
                    "target_threshold": round(target_threshold, 1),
                    "progress_pct": round(progress_pct, 1),
                    "urgency_score": round(score, 1),
                    "direction": direction,
                    "urgency_message": urgency,
                    "calibrated_price_forecast": is_calibrated,
                }
            )

        # Sort by urgency
        predictions.sort(key=lambda x: abs(x["urgency_score"]), reverse=True)
        return predictions


price_change_predictor = PriceChangePredictor()
