"""
Structured Recommendation Engine and Numeric Grounding Verification (WI-6.3).

Ensures all assistant narratives are strictly grounded in mathematical tool outputs:
1. Emits strict StructuredRecommendation objects (action, verdict, numbers, alternatives).
2. Verifies that every numeric figure in the generated narrative exists in tool outputs (allowing rounding).
3. On numeric grounding failure, regenerates once with targeted error feedback; on second failure,
   falls back to deterministic templated narration.
4. Model names and fallbacks live in config, not code.
5. Clearly labels offline mode.
"""

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from fpl_oracle.config import CONFIG_DIR, _load_yaml

logger = logging.getLogger("fpl_oracle.llm.recommendation")


def get_assistant_config() -> dict[str, Any]:
    """Load assistant model settings and tier from config/flags.yaml."""
    flags_path = CONFIG_DIR / "flags.yaml"
    default_cfg = {
        "model_tier": "flash_lite",
        "primary_model": "gemini-flash-lite-latest",
        "fallback_model": "offline",
        "enforce_numeric_grounding": True,
        "label_offline_mode": True,
    }
    if not flags_path.exists():
        return default_cfg
    try:
        data = _load_yaml(flags_path) or {}
        assistant = data.get("assistant", {})
        merged = dict(default_cfg)
        merged.update({k: v for k, v in assistant.items() if v is not None})
        return merged
    except Exception as e:
        logger.warning(f"Error loading assistant config: {e}")
        return default_cfg


class RecommendedMove(BaseModel):
    element_out: str
    element_out_id: int | None = None
    cost_out: float = 0.0
    element_in: str
    element_in_id: int | None = None
    cost_in: float = 0.0
    position: str = ""
    net_cost: float = 0.0


class StructuredRecommendation(BaseModel):
    """
    Formal, machine-readable recommendation emitted by the mathematical optimization engine.
    Precedes narrative generation to anchor all cited facts and numbers.
    """

    recommendation_id: str
    gameweek: int
    action_type: str = Field(description="ROLL, 1_TRANSFER, 2_TRANSFERS, HIT_MINUS_4, CHIP, CAPTAIN")
    headline_verdict: str
    expected_points_primary: float
    expected_points_baseline: float
    expected_gain_xp: float
    confidence_tier: str = "HIGH"  # HIGH, MEDIUM, LOW
    grounded_numbers: dict[str, float | int] = Field(default_factory=dict)
    moves: list[RecommendedMove] = Field(default_factory=list)
    captain_pick: dict[str, Any] = Field(default_factory=dict)
    vice_captain_pick: dict[str, Any] = Field(default_factory=dict)
    alternatives: list[dict[str, Any]] = Field(default_factory=list)
    decision_triggers: list[str] = Field(default_factory=list)
    offline_mode: bool = False

    def collect_all_grounded_numbers(self) -> set[float]:
        """Aggregate all legitimate numbers that may appear in the narrative."""
        nums: set[float] = {
            float(self.gameweek),
            round(float(self.expected_points_primary), 2),
            round(float(self.expected_points_baseline), 2),
            round(float(self.expected_gain_xp), 2),
            round(float(self.expected_points_primary), 1),
            round(float(self.expected_points_baseline), 1),
            round(float(self.expected_gain_xp), 1),
            round(float(self.expected_points_primary), 0),
            round(float(self.expected_points_baseline), 0),
            round(float(self.expected_gain_xp), 0),
        }

        # Dedicated numbers dict
        for val in self.grounded_numbers.values():
            try:
                fv = float(val)
                nums.add(round(fv, 2))
                nums.add(round(fv, 1))
                nums.add(round(fv, 0))
            except (ValueError, TypeError):
                continue

        # Moves costs and IDs
        for m in self.moves:
            for cost in (m.cost_out, m.cost_in, m.net_cost):
                nums.add(round(float(cost), 2))
                nums.add(round(float(cost), 1))
            if m.element_out_id:
                nums.add(float(m.element_out_id))
            if m.element_in_id:
                nums.add(float(m.element_in_id))

        # Captain metrics
        for cap in (self.captain_pick, self.vice_captain_pick):
            for k in ("expected_points", "cost", "p10_floor", "p90_ceiling", "p10", "p90", "defcon_pts"):
                if k in cap and cap[k] is not None:
                    try:
                        fv = float(cap[k])
                        nums.add(round(fv, 2))
                        nums.add(round(fv, 1))
                        nums.add(round(fv, 0))
                    except (ValueError, TypeError):
                        pass

        # Alternatives
        for alt in self.alternatives:
            for k in ("net_xp", "gain", "expected_gain", "points", "gameweek"):
                if k in alt and alt[k] is not None:
                    try:
                        fv = float(alt[k])
                        nums.add(round(fv, 2))
                        nums.add(round(fv, 1))
                        nums.add(round(fv, 0))
                    except (ValueError, TypeError):
                        pass

        return nums


def extract_numbers_from_text(text: str) -> list[float]:
    """
    Extracts numeric tokens from narrative text, filtering out dates, years (2026/2027),
    markdown numbering (e.g. '1.'), and standard punctuation.
    """
    # Remove dates like 2026-10-15 or years 2026, 2027
    cleaned = re.sub(r"\b202[4-9]\b", " ", text)
    # Remove markdown enumeration lines like "1. ", "2. "
    cleaned = re.sub(r"^\s*\d+\.\s+", " ", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"[£$€]", " ", cleaned)

    # Match floating point or integer numbers (supports £9.5m, 14.8pts, etc.)
    tokens = re.findall(r"(\d+(?:\.\d+)?)", cleaned)
    results = []
    for t in tokens:
        t_clean = t.strip()
        if not t_clean or t_clean in (".", "-"):
            continue
        try:
            results.append(float(t_clean))
        except ValueError:
            continue
    return results


def verify_numeric_grounding(
    narration: str,
    grounded_numbers: set[float] | list[float],
    tolerance: float = 0.15,
) -> tuple[bool, list[float]]:
    """
    Checks that every number in narration appears in the grounded numbers (allowing rounding).
    Returns (is_grounded, list_of_ungrounded_numbers).
    """
    found_numbers = extract_numbers_from_text(narration)
    allowed = set(grounded_numbers)

    ungrounded: list[float] = []
    for num in found_numbers:
        # Check exact or fuzzy tolerance match
        matched = False
        for target in allowed:
            if abs(num - target) <= tolerance:
                matched = True
                break
            # Also allow integer vs rounded comparison
            if abs(round(num) - round(target)) <= 0.01:
                matched = True
                break
        if not matched:
            ungrounded.append(num)

    return (len(ungrounded) == 0, ungrounded)


def generate_templated_narration(rec: StructuredRecommendation) -> str:
    """
    Deterministic expert narration generated strictly from StructuredRecommendation.
    Used for offline mode and guaranteed hallucination-free fallback.
    """
    lines: list[str] = []

    if rec.offline_mode:
        lines.append(
            "[OFFLINE EXPERT MODE: Running purely on live local data, projections, and mathematical optimizations without external LLM keys.]\n"
        )

    lines.append(f"### Gameweek {rec.gameweek} Recommendation & Mathematical Working")
    lines.append(f"**Verdict:** {rec.headline_verdict}")
    lines.append(
        f"**Expected Points:** {rec.expected_points_primary:.1f} xP "
        f"(+{rec.expected_gain_xp:.1f} vs baseline {rec.expected_points_baseline:.1f} xP; Confidence: {rec.confidence_tier})\n"
    )

    if rec.moves:
        lines.append("**Recommended Moves:**")
        for m in rec.moves:
            cost_str = f"£{m.cost_out:.1f}m -> £{m.cost_in:.1f}m"
            lines.append(f"- **OUT:** {m.element_out} -> **IN:** {m.element_in} ({cost_str})")
        lines.append("")

    if rec.captain_pick:
        cap_name = rec.captain_pick.get("web_name", "Primary Captain")
        cap_xp = rec.captain_pick.get("expected_points", 0.0)
        p10 = rec.captain_pick.get("p10_floor", rec.captain_pick.get("p10", 0.0))
        p90 = rec.captain_pick.get("p90_ceiling", rec.captain_pick.get("p90", 0.0))
        lines.append(f"**Captaincy:** {cap_name} ({cap_xp:.1f} xP | P10 Floor: {p10:.1f} | P90 Ceiling: {p90:.1f})\n")

    if rec.alternatives:
        lines.append("**Evaluated Alternatives:**")
        for alt in rec.alternatives:
            lines.append(
                f"- {alt.get('summary', alt.get('name', 'Plan'))}: {alt.get('net_xp', alt.get('gain', ''))} xP"
            )
        lines.append("")

    if rec.decision_triggers:
        lines.append("**Decision Sensitivity & Triggers:**")
        for trig in rec.decision_triggers:
            lines.append(f"- {trig}")

    return "\n".join(lines).strip()


async def narrate_recommendation_with_grounding(
    rec: StructuredRecommendation,
    chat_callable: Any = None,
    max_retries: int = 1,
) -> tuple[str, bool, str]:
    """
    Narrates a StructuredRecommendation with strict numeric grounding verification.
    If LLM narration contains ungrounded numbers, retries once with explicit feedback;
    if still ungrounded, falls back immediately to deterministic templated narration.

    Returns: (final_text, is_grounded, source_type: 'llm' | 'llm_retry' | 'templated_fallback')
    """
    _cfg = get_assistant_config()
    allowed_numbers = rec.collect_all_grounded_numbers()

    # If offline mode or no external callable, return templated narration immediately
    if rec.offline_mode or chat_callable is None:
        rec.offline_mode = True
        templated = generate_templated_narration(rec)
        return templated, True, "templated_offline"

    # Format structured context for LLM narration
    grounding_prompt = (
        f"You are the FPL Oracle expert narrator. Convert this exact recommendation into a concise, professional briefing.\n"
        f"STRICT GROUNDING RULE: You may ONLY cite numbers that exist in this JSON. Never invent or adjust any numeric values.\n\n"
        f"JSON RECOMMENDATION:\n{rec.model_dump_json(indent=2)}\n\n"
        f"Narrate the recommendation now:"
    )

    try:
        raw_narrative = await chat_callable(grounding_prompt)
        is_grounded, ungrounded = verify_numeric_grounding(raw_narrative, allowed_numbers)

        if is_grounded:
            return raw_narrative, True, "llm"

        logger.warning(
            "[GroundingCheck] LLM narration contained ungrounded numbers: %s. Retrying once with error feedback...",
            ungrounded,
        )

        # Retry once with explicit feedback
        if max_retries > 0:
            retry_prompt = (
                f"{grounding_prompt}\n\n"
                f"ERROR CORRECTION: Your previous draft was rejected because it hallucinated numbers {ungrounded} "
                f"which were NOT in the provided recommendation object. "
                f"Rewrite the narrative citing ONLY the numbers explicitly given in the JSON:"
            )
            retry_narrative = await chat_callable(retry_prompt)
            retry_grounded, retry_ungrounded = verify_numeric_grounding(retry_narrative, allowed_numbers)
            if retry_grounded:
                return retry_narrative, True, "llm_retry"

            logger.warning(
                "[GroundingCheck] Second draft also contained ungrounded numbers: %s. Falling back to deterministic template.",
                retry_ungrounded,
            )

        # Fallback to deterministic template
        templated = generate_templated_narration(rec)
        return templated, True, "templated_fallback"

    except Exception as e:
        logger.error(f"[GroundingCheck] Error in LLM chat callable: {e}. Using templated narration.")
        templated = generate_templated_narration(rec)
        return templated, True, "templated_fallback"
