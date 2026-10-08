"""
Price Change Prediction & Urgency Engine.
Uses official API price change metrics, hourly rate, and transfer momentum
to predict imminent rises and falls tonight.
"""

from typing import Any

import numpy as np

from fpl_oracle.api.models import BootstrapStatic


class PriceChangePredictor:
    def __init__(self):
        pass

    def analyze_price_changes(self, bootstrap: BootstrapStatic) -> list[dict[str, Any]]:
        """
        Analyze all elements for price change likelihood tonight.
        Returns sorted list of players near price change thresholds.
        """
        predictions = []

        for elem in bootstrap.elements:
            # 1. API native projections if available
            api_prob = None
            if elem.price_change_projections and len(elem.price_change_projections) > 0:
                try:
                    p_val = float(elem.price_change_projections[0].get("projected_percent", 0.0))
                    api_prob = p_val
                except Exception:
                    api_prob = None

            # 2. Heuristic momentum from transfers_in_event vs transfers_out_event
            net_transfers = (elem.transfers_in_event or 0) - (elem.transfers_out_event or 0)
            ownership = float(elem.selected_by_percent or 1.0)

            # Hourly rate from 2026/27 API
            hourly_rate = elem.price_change_hourly_rate or 0

            # Composite urgency score (-100 to +100)
            # Threshold in FPL typically corresponds to ~100k net transfers or ~10% net ownership delta
            score = 0.0
            if api_prob is not None:
                score = api_prob * 10.0
            else:
                score = (net_transfers / max(5000, ownership * 10000)) * 50.0

            if hourly_rate > 0 or hourly_rate < 0:
                score += hourly_rate * 0.5

            score = float(np.clip(score, -100.0, 100.0))

            direction = "STABLE"
            urgency = "Can wait"
            if score >= 80.0:
                direction = "RISE_IMMINENT"
                urgency = "Strong rising transfer momentum; price change unconfirmed"
            elif score >= 50.0:
                direction = "LIKELY_RISE"
                urgency = "Rising momentum"
            elif score <= -80.0:
                direction = "FALL_IMMINENT"
                urgency = "Strong falling transfer momentum; price change unconfirmed"
            elif score <= -50.0:
                direction = "LIKELY_FALL"
                urgency = "Falling momentum"

            predictions.append(
                {
                    "element": elem.id,
                    "web_name": elem.web_name,
                    "team": elem.team,
                    "now_cost": elem.now_cost / 10.0,
                    "net_transfers_event": net_transfers,
                    "selected_by_percent": ownership,
                    "hourly_rate": hourly_rate,
                    "urgency_score": round(score, 1),
                    "direction": direction,
                    "urgency_message": urgency,
                    "calibrated_price_forecast": False,
                }
            )

        # Sort by urgency
        predictions.sort(key=lambda x: abs(float(str(x["urgency_score"]))), reverse=True)
        return predictions


price_change_predictor = PriceChangePredictor()
