"""
Transfer Timing & Expected Value of Information Engine.
Quantifies the exact trade-off between executing early to beat a £0.1m price rise
versus waiting for Friday manager press conferences and midweek European matches.
Zero hardcoded player names in production code.
"""

import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from fpl_oracle.config import SETTINGS


@dataclass
class PriceMicrostructure:
    element_id: int
    web_name: str
    now_cost: float
    p_rise: float
    p_fall: float
    net_transfers: int
    target_threshold: float
    transfers_to_threshold: int
    hours_to_cutoff: float
    is_locked: bool
    lock_reason: str
    action_tier: str  # "URGENT_BUY_TONIGHT", "URGENT_SELL_TONIGHT", "SAFE_HOLD", "PRICE_LOCKED"
    recommendation: str


@dataclass
class TimingRecommendation:
    urgency: Literal["HOLD_FOR_PRESSERS", "EXECUTE_TONIGHT_PRICED_OUT", "FLEXIBLE_SAFE"]
    ev_waiting_pts: float
    cash_buffer: float
    price_change_prob: float
    pressers_concluded: bool
    midweek_matches_pending: int
    headline: str
    detailed_reason: str


class TransferTimingAdvisor:
    """
    Evaluates whether a planned transfer should be made early tonight or deferred
    until post-press conference information is fully known.
    """

    def __init__(
        self,
        price_rise_ev_value: float | None = None,
        injury_risk_ev_damage: float | None = None,
        priced_out_opportunity_cost: float | None = None,
    ):
        adv_cfg = SETTINGS.get("advisor", {})
        if price_rise_ev_value is None:
            self.price_rise_ev_value = float(adv_cfg.get("price_rise_ev_value", 0.08))
        else:
            self.price_rise_ev_value = float(price_rise_ev_value)

        if injury_risk_ev_damage is None:
            self.injury_risk_ev_damage = float(adv_cfg.get("injury_risk_ev_damage", 4.5))
        else:
            self.injury_risk_ev_damage = float(injury_risk_ev_damage)

        if priced_out_opportunity_cost is None:
            self.priced_out_opportunity_cost = float(adv_cfg.get("priced_out_opportunity_cost", 1.5))
        else:
            self.priced_out_opportunity_cost = float(priced_out_opportunity_cost)

    def evaluate_timing(
        self,
        bank: float,
        target_rise_prob: float = 0.0,
        outgoing_fall_prob: float = 0.0,
        pressers_concluded: bool = False,
        midweek_matches_pending: int = 0,
        target_cost_tenths: int | None = None,
        opportunity_cost: float | None = None,
    ) -> TimingRecommendation:
        """
        Calculates the expected utility of waiting for pressers vs executing early.

        Parameters:
            bank: Current remaining bank in millions (e.g. 0.0, 0.2, 1.5).
            target_rise_prob: Probability that incoming target rises tonight (0.0 to 1.0).
            outgoing_fall_prob: Probability that outgoing asset falls tonight (0.0 to 1.0).
            pressers_concluded: Whether relevant pre-match press conferences have occurred.
            midweek_matches_pending: Number of European/domestic cup games remaining this week.
            target_cost_tenths: Optional target cost in tenths (e.g. 95 for £9.5m).
        """
        p_rise = max(0.0, min(1.0, float(target_rise_prob)))
        p_fall = max(0.0, min(1.0, float(outgoing_fall_prob)))
        # Independent probability union: P(A or B) = P(A) + P(B) - P(A and B)
        price_change_prob = round(max(0.0, min(1.0, p_rise + p_fall - (p_rise * p_fall))), 4)
        # Defensive normalization: if bank > 5.0, it is stored in tenths (e.g. 10 -> £1.0m, 15 -> £1.5m)
        bank_m = float(bank) / 10.0 if float(bank) > 5.0 else float(bank)
        bank_buffer = round(bank_m, 2)

        # Baseline probability of an unannounced training knock or midweek injury
        if pressers_concluded and midweek_matches_pending == 0:
            p_injury = 0.01
        else:
            p_injury = 0.05 + 0.03 * max(0, int(midweek_matches_pending))

        # Real Options Price Loss: Linearity of Expectation E[L] = (p_rise + p_fall) * unit_loss
        # When bank_buffer < 0.10m, adverse moves trigger barrier event (priced out of plan),
        # incorporating plan opportunity cost into EV of waiting.
        opp_cost = self.priced_out_opportunity_cost if opportunity_cost is None else float(opportunity_cost)
        unit_price_loss = self.price_rise_ev_value + (opp_cost if bank_buffer < 0.10 else 0.0)
        price_loss_ev = (p_rise + p_fall) * unit_price_loss

        injury_aversion_ev = p_injury * self.injury_risk_ev_damage
        ev_waiting = round(injury_aversion_ev - price_loss_ev, 3)

        # Decision classification
        if pressers_concluded and midweek_matches_pending == 0:
            urgency: Literal["HOLD_FOR_PRESSERS", "EXECUTE_TONIGHT_PRICED_OUT", "FLEXIBLE_SAFE"] = "FLEXIBLE_SAFE"
            headline = "FLEXIBLE: Press conferences concluded; safe to execute"
            detailed_reason = (
                "Manager press conferences have concluded and 0 midweek fixtures remain. "
                "Information risk is minimal; execute at your convenience."
            )
        elif bank_buffer >= 0.10:
            urgency = "HOLD_FOR_PRESSERS"
            headline = "HOLD FOR PRESSERS: Available bank buffer absorbs price fluctuations"
            detailed_reason = (
                f"Available bank buffer of £{bank_buffer:.2f}m absorbs price fluctuations. "
                f"Even with a {price_change_prob * 100:.0f}% price change likelihood tonight, the expected value of waiting "
                f"({ev_waiting:+.2f} pts) dominates the ~{self.price_rise_ev_value:.2f} pts purchasing power impact of a £0.1m rise. "
                f"Protect against midweek/training injury surprises."
            )
        elif price_change_prob >= 0.70 and bank_buffer < 0.10:
            urgency = "EXECUTE_TONIGHT_PRICED_OUT"
            headline = "EXECUTE TONIGHT: Zero bank buffer with imminent price lock"
            detailed_reason = (
                f"Bank buffer is strictly £{bank_buffer:.2f}m (< £0.1m). Imminent price change probability "
                f"({price_change_prob * 100:.0f}%) will leave you priced out of this planned transfer. "
                f"Executing tonight is necessary to preserve the optimal transfer path."
            )
        else:
            urgency = "HOLD_FOR_PRESSERS"
            headline = "HOLD FOR PRESSERS: Price change unconfirmed; preserve optionality"
            detailed_reason = (
                f"Price change likelihood ({price_change_prob * 100:.0f}%) is below urgent threshold. "
                f"EV of waiting ({ev_waiting:+.2f} pts) recommends waiting for official availability updates."
            )

        return TimingRecommendation(
            urgency=urgency,
            ev_waiting_pts=ev_waiting,
            cash_buffer=bank_buffer,
            price_change_prob=price_change_prob,
            pressers_concluded=pressers_concluded,
            midweek_matches_pending=midweek_matches_pending,
            headline=headline,
            detailed_reason=detailed_reason,
        )


transfer_timing_advisor = TransferTimingAdvisor()


def predict_player_price_microstructure(
    elem: Any,
    now_dt: datetime | None = None,
) -> PriceMicrostructure:
    """
    Evaluates individual player microstructure including dynamic Net Transfer Activity (NTA)
    thresholds, price-lock status, hours to 01:30 UK cutoff, and action recommendation.
    """
    eid = getattr(elem, "id", 0)
    web_name = getattr(elem, "web_name", f"Player_{eid}")
    raw_cost = getattr(elem, "now_cost", 50)
    now_cost = round(float(raw_cost) / 10.0, 1)

    tin = getattr(elem, "transfers_in_event", 0) or 0
    tout = getattr(elem, "transfers_out_event", 0) or 0
    net_transfers = int(tin - tout)

    try:
        ownership = float(getattr(elem, "selected_by_percent", 1.0) or 1.0)
    except (ValueError, TypeError):
        ownership = 1.0

    cost_change = getattr(elem, "cost_change_event", 0) or 0
    status = getattr(elem, "status", "a")

    # 1. Price Lock Assessment
    # In official FPL rules, players returning from red flag ('i' or 's' to 'a') are price-locked for 8 days.
    # Players transferred out of the Premier League ('u') are also locked.
    is_locked = False
    lock_reason = ""
    if status == "u":
        is_locked = True
        lock_reason = "Player transferred outside the league."
    elif getattr(elem, "is_price_locked", False):
        is_locked = True
        lock_reason = "Official 8-day price protection lock active."

    # 2. Dynamic Threshold Calibration (FPL 2026/27 ~11m Manager Pool)
    # Threshold scales with ownership base: higher ownership assets require more net transfers to shift price.
    base_thresh = max(8000.0, min(95000.0, 6000.0 + ownership * 3200.0))

    if net_transfers > 0:
        if cost_change > 0:
            target_threshold = base_thresh * (1.0 + float(cost_change) * 0.70)
        else:
            target_threshold = base_thresh
        transfers_to_threshold = max(0, int(target_threshold - net_transfers))
    elif net_transfers < 0:
        if cost_change < 0:
            target_threshold = base_thresh * (1.0 + abs(float(cost_change)) * 0.70)
        else:
            target_threshold = base_thresh
        transfers_to_threshold = max(0, int(target_threshold - abs(net_transfers)))
    else:
        target_threshold = base_thresh
        transfers_to_threshold = int(target_threshold)

    # 3. Time until nightly price change window (~01:30 UK time)
    ref_dt = now_dt or datetime.now(UTC)
    cutoff_today = ref_dt.replace(hour=1, minute=30, second=0, microsecond=0)
    if cutoff_today <= ref_dt:
        cutoff_today += timedelta(days=1)
    hours_to_cutoff = round((cutoff_today - ref_dt).total_seconds() / 3600.0, 1)

    # 4. Calibrated Sigmoid Probabilities
    if is_locked:
        p_rise = 0.0
        p_fall = 0.0
    else:
        z_rise = max(0.0, net_transfers / target_threshold)
        z_fall = max(0.0, -net_transfers / target_threshold)
        p_rise = 1.0 / (1.0 + math.exp(-7.5 * (z_rise - 0.95)))
        p_fall = 1.0 / (1.0 + math.exp(-7.5 * (z_fall - 0.95)))

        # Blend with API projections if provided
        proj = getattr(elem, "price_change_projections", None)
        if proj and len(proj) > 0:
            try:
                api_pct = float(proj[0].get("projected_percent", 0.0)) / 100.0
                if api_pct > 0:
                    p_rise = 0.4 * p_rise + 0.6 * min(1.0, max(0.0, api_pct))
                elif api_pct < 0:
                    p_fall = 0.4 * p_fall + 0.6 * min(1.0, max(0.0, abs(api_pct)))
            except Exception:
                pass

    p_rise = round(max(0.0, min(1.0, p_rise)), 4)
    p_fall = round(max(0.0, min(1.0, p_fall)), 4)

    # 5. Recommendation Action Tier
    if is_locked:
        action_tier = "PRICE_LOCKED"
        recommendation = f"Price is protected from changes ({lock_reason}). Safe to defer."
    elif p_rise >= 0.70:
        action_tier = "URGENT_BUY_TONIGHT"
        recommendation = f"Target {web_name} has {p_rise * 100:.0f}% probability of rising £0.1m tonight ({transfers_to_threshold} transfers to trigger). Buy before 01:30 UK cutoff."
    elif p_fall >= 0.70:
        action_tier = "URGENT_SELL_TONIGHT"
        recommendation = f"Asset {web_name} has {p_fall * 100:.0f}% probability of falling £0.1m tonight ({transfers_to_threshold} transfers to trigger). Sell before 01:30 UK cutoff."
    else:
        action_tier = "SAFE_HOLD"
        recommendation = (
            "Safe to hold: price movement tonight is highly improbable. Defer moves until manager press conferences."
        )

    return PriceMicrostructure(
        element_id=int(eid),
        web_name=str(web_name),
        now_cost=now_cost,
        p_rise=p_rise,
        p_fall=p_fall,
        net_transfers=net_transfers,
        target_threshold=round(target_threshold, 1),
        transfers_to_threshold=transfers_to_threshold,
        hours_to_cutoff=hours_to_cutoff,
        is_locked=is_locked,
        lock_reason=lock_reason,
        action_tier=action_tier,
        recommendation=recommendation,
    )


def estimate_price_change_probabilities(
    element_ids: list[int] | set[int] | int,
    bootstrap: Any,
) -> dict[int, dict[str, Any]]:
    """
    Computes data-driven p_rise and p_fall probabilities for given element IDs
    using transfers_in_event, transfers_out_event, selected_by_percent, and cost_change_event.
    Includes full microstructure profile under 'microstructure'.
    """
    if isinstance(element_ids, int):
        element_ids = [element_ids]
    target_ids = set(element_ids)

    elements = getattr(bootstrap, "elements", []) if bootstrap else []
    elem_map = {e.id: e for e in elements if e.id in target_ids}
    results: dict[int, dict[str, Any]] = {}

    for eid in target_ids:
        elem = elem_map.get(eid)
        if not elem:
            results[eid] = {
                "p_rise": 0.0,
                "p_fall": 0.0,
                "net_transfers": 0,
                "selected_by_percent": 1.0,
                "microstructure": None,
            }
            continue

        profile = predict_player_price_microstructure(elem)

        results[eid] = {
            "p_rise": profile.p_rise,
            "p_fall": profile.p_fall,
            "net_transfers": profile.net_transfers,
            "selected_by_percent": getattr(elem, "selected_by_percent", 1.0),
            "cost_change_event": getattr(elem, "cost_change_event", 0),
            "microstructure": asdict(profile),
        }

    return results


def get_squad_price_radar(elements: list[Any], bootstrap: Any) -> list[dict[str, Any]]:
    """
    Evaluates price rise and fall microstructure across a full squad or candidate watchlist.
    """
    element_ids = [
        int(getattr(e, "id", e) if not isinstance(e, dict) else e.get("element", e.get("id", 0))) for e in elements
    ]
    probs = estimate_price_change_probabilities(element_ids, bootstrap)
    radar = []
    for pdata in probs.values():
        micro = pdata.get("microstructure")
        if micro:
            radar.append(micro)
    radar.sort(key=lambda x: max(x["p_rise"], x["p_fall"]), reverse=True)
    return radar
