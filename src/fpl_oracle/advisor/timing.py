"""
Transfer Timing & Expected Value of Information Engine (Hall of Fame Feature 2).
Quantifies the exact trade-off between executing early to beat a £0.1m price rise
versus waiting for Friday manager press conferences and midweek European matches.
Zero hardcoded player names in production code.
"""

from dataclasses import dataclass
from typing import Literal

from fpl_oracle.config import SETTINGS


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

    def evaluate_timing(
        self,
        bank: float,
        target_rise_prob: float = 0.0,
        outgoing_fall_prob: float = 0.0,
        pressers_concluded: bool = False,
        midweek_matches_pending: int = 0,
        target_cost_tenths: int | None = None,
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
        price_change_prob = max(0.0, min(1.0, max(float(target_rise_prob), float(outgoing_fall_prob))))
        bank_buffer = round(float(bank), 2)

        # Baseline probability of an unannounced training knock or midweek injury
        if pressers_concluded and midweek_matches_pending == 0:
            p_injury = 0.01
        else:
            p_injury = 0.05 + 0.03 * max(0, int(midweek_matches_pending))

        # Expected Value of waiting: P(injury averted) * Damage - Price Rise Cost
        price_loss_ev = price_change_prob * self.price_rise_ev_value
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
                f"(+{ev_waiting:.2f} pts) dominates the ~0.08 pts purchasing power impact of a £0.1m rise. "
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
                f"EV of waiting (+{ev_waiting:.2f} pts) recommends waiting for official availability updates."
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
