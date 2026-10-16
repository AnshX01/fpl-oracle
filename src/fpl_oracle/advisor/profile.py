"""
Manager Profile and Context Management (Phase 8: WI-8.5).
Stores and manages manager entry context, mini-league standings, risk appetite,
objective weights, and hard preferences (e.g., never sell X, avoid club Y).
Differentiates between persistent profile preferences and transient session overrides.
Zero hardcoded player names in production code.
"""

import copy
import logging
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from fpl_oracle.config import BASE_DIR, CONFIG_DIR

logger = logging.getLogger("fpl_oracle.advisor.profile")

DEFAULT_PROFILE_PATH = CONFIG_DIR / "manager_profile.yaml"


class MiniLeagueContext(BaseModel):
    league_id: int
    name: str = ""
    rank: int = 1
    total_points: int = 0
    leader_points: int = 0
    gap_to_leader: int = 0
    target_rank: int = 1


class HardPreferences(BaseModel):
    never_sell: list[int] = Field(default_factory=list)
    avoid_clubs: list[int] = Field(default_factory=list)
    min_captain_pool: list[int] = Field(default_factory=list)
    custom_constraints: dict[str, Any] = Field(default_factory=dict)


class ManagerProfile(BaseModel):
    entry_id: int = 1085141
    manager_name: str = ""
    team_name: str = ""
    mini_leagues: list[MiniLeagueContext] = Field(default_factory=list)
    objective_weight: Literal["points", "league_win", "overall_rank"] = "points"
    risk_appetite: Literal["template-safe", "balanced", "differential-hungry"] = "balanced"
    hard_preferences: HardPreferences = Field(default_factory=HardPreferences)
    hit_tolerance: Literal["low", "medium", "high"] = "medium"
    check_frequency: str = "daily"


class ManagerProfileService:
    """
    Manages loading, updating, session-overriding, and applying manager profiles (WI-8.5).
    Enforces Principle P5: Knows the manager.
    """

    def __init__(self, profile_path: Path | None = None) -> None:
        self.profile_path = profile_path or DEFAULT_PROFILE_PATH
        self._session_overrides: dict[str, Any] = {}
        self._cached_profile: ManagerProfile | None = None

    def load_profile(self) -> ManagerProfile:
        """Loads manager profile from YAML, falling back to data/profile.json or defaults."""
        if self._cached_profile is not None:
            return self._apply_session_overrides(self._cached_profile)

        data: dict[str, Any] = {}
        if self.profile_path.exists():
            try:
                with open(self.profile_path, encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        data = loaded
            except Exception as e:
                logger.warning(f"Failed to read {self.profile_path}: {e}")

        if not data:
            legacy_path = BASE_DIR / "data" / "profile.json"
            if legacy_path.exists():
                try:
                    import json

                    with open(legacy_path, encoding="utf-8") as f:
                        legacy_data = json.load(f)
                    if isinstance(legacy_data, dict):
                        data = {
                            "entry_id": legacy_data.get("manager_id", 1085141),
                            "risk_appetite": "template-safe" if legacy_data.get("risk_preference") == "conservative" else "balanced",
                            "mini_leagues": [
                                {
                                    "league_id": legacy_data.get("target_league_id", 435956),
                                    "name": "Target Mini-League",
                                    "rank": 2,
                                    "gap_to_leader": 14,
                                }
                            ]
                            if legacy_data.get("target_league_id")
                            else [],
                        }
                except Exception as e:
                    logger.warning(f"Failed to read legacy profile {legacy_path}: {e}")

        try:
            profile = ManagerProfile.model_validate(data) if data else ManagerProfile()
        except Exception as e:
            logger.warning(f"Schema validation failed for profile: {e}. Using defaults.")
            profile = ManagerProfile()

        self._cached_profile = profile
        return self._apply_session_overrides(profile)

    def save_profile(self, profile: ManagerProfile) -> None:
        """Persists profile to configuration path."""
        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        dump_data = profile.model_dump(mode="json")
        with open(self.profile_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(dump_data, f, sort_keys=False)
        self._cached_profile = profile

    def set_session_override(self, key: str, value: Any) -> None:
        """Sets a transient preference for the current conversation without persisting to disk."""
        self._session_overrides[key] = value

    def clear_session_overrides(self) -> None:
        """Clears all session-scoped preference overrides."""
        self._session_overrides.clear()

    def persist_preference(self, key: str, value: Any) -> ManagerProfile:
        """Persists a preference to the YAML config when explicitly requested by user."""
        current = self.load_profile()
        current_dict = current.model_dump()
        if key in current_dict:
            current_dict[key] = value
        elif hasattr(current.hard_preferences, key):
            current_dict.setdefault("hard_preferences", {})[key] = value
        else:
            current_dict.setdefault("hard_preferences", {}).setdefault("custom_constraints", {})[key] = value

        updated = ManagerProfile.model_validate(current_dict)
        self.save_profile(updated)
        return updated

    def _apply_session_overrides(self, profile: ManagerProfile) -> ManagerProfile:
        """Applies active session overrides on top of loaded profile."""
        if not self._session_overrides:
            return profile

        p_copy = copy.deepcopy(profile)
        for k, v in self._session_overrides.items():
            if hasattr(p_copy, k):
                setattr(p_copy, k, v)
            elif hasattr(p_copy.hard_preferences, k):
                setattr(p_copy.hard_preferences, k, v)
            else:
                p_copy.hard_preferences.custom_constraints[k] = v
        return p_copy

    def rank_captains(
        self,
        player_pool_df: pd.DataFrame,
        squad_element_ids: list[int],
        effective_ownership_map: dict[int, float] | None = None,
        profile: ManagerProfile | None = None,
    ) -> list[dict[str, Any]]:
        """
        Ranks captaincy choices from squad_element_ids incorporating risk appetite (WI-8.5).
        - template-safe: rewards high effective ownership (protects rank/lead)
        - differential-hungry: rewards low effective ownership and ceiling P90
        - balanced: pure expected points
        """
        prof = profile or self.load_profile()
        eo_map = effective_ownership_map or {}

        squad_set = set(squad_element_ids)
        candidates_df = player_pool_df[player_pool_df["element"].isin(squad_set)].copy()
        if candidates_df.empty:
            return []

        ranked_candidates: list[dict[str, Any]] = []

        for _, row in candidates_df.iterrows():
            eid = int(row["element"])
            xp = float(row.get("expected_points", 0.0))
            p10 = float(row.get("p10", xp * 0.4))
            p90 = float(row.get("p90", xp * 1.8))
            eo = float(eo_map.get(eid, row.get("selected_by_percent", 20.0)))

            if prof.risk_appetite == "template-safe":
                # Safety bonus for high ownership: S = xP + 0.05 * EO
                score = xp + (0.05 * eo)
                rationale = f"Template Protection (EO: {eo:.1f}%, xP: {xp:.2f})"
            elif prof.risk_appetite == "differential-hungry":
                # Differential bonus for low ownership + high ceiling: S = xP + 0.05 * (100-EO) + 0.35 * (P90-xP)
                diff_bonus = 0.05 * max(0.0, 100.0 - eo)
                ceiling_bonus = 0.35 * max(0.0, p90 - xp)
                score = xp + diff_bonus + ceiling_bonus
                rationale = f"Differential Upside (EO: {eo:.1f}%, Ceiling P90: {p90:.2f})"
            else:
                score = xp
                rationale = f"Balanced Expected Value (xP: {xp:.2f})"

            # If element is in min_captain_pool, add prior boost
            if prof.hard_preferences.min_captain_pool and eid in prof.hard_preferences.min_captain_pool:
                score += 1.0
                rationale += " [Preferred Captain Pool]"

            ranked_candidates.append({
                "element": eid,
                "web_name": str(row.get("web_name", f"ID_{eid}")),
                "position": str(row.get("position", "MID")),
                "expected_points": xp,
                "p10_floor": p10,
                "p90_ceiling": p90,
                "effective_ownership": eo,
                "selection_score": round(score, 3),
                "rationale": rationale,
            })

        ranked_candidates.sort(key=lambda c: c["selection_score"], reverse=True)
        return ranked_candidates

    def apply_hard_constraints_to_optimizer_args(
        self,
        kwargs: dict[str, Any],
        profile: ManagerProfile | None = None,
    ) -> dict[str, Any]:
        """
        Injects profile's hard preferences into MultiPeriodOptimizer arguments.
        - never_sell IDs are added to locked_in_ids
        - avoid_clubs IDs are added to excluded_team_ids
        """
        prof = profile or self.load_profile()
        updated_kwargs = dict(kwargs)

        # 1. Never sell constraints
        if prof.hard_preferences.never_sell:
            existing_locked_in = set(updated_kwargs.get("locked_in_ids") or [])
            existing_locked_in.update(prof.hard_preferences.never_sell)
            updated_kwargs["locked_in_ids"] = list(existing_locked_in)

        # 2. Avoid clubs
        if prof.hard_preferences.avoid_clubs:
            existing_excluded = set(updated_kwargs.get("excluded_team_ids") or [])
            existing_excluded.update(prof.hard_preferences.avoid_clubs)
            updated_kwargs["excluded_team_ids"] = list(existing_excluded)

        return updated_kwargs


manager_profile_service = ManagerProfileService()
