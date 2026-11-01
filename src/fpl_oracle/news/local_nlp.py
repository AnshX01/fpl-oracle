"""
Local Calibrated Press Conference NLP Engine.
Zero-token, offline probabilistic availability extractor for press conference transcripts,
medical updates, and club media releases.

Calibrates managerial statements into 5 standardized probability bands:
- 0.00: Confirmed Out / Ruled Out / Surgery / Long-term layoff
- 0.25: Major Doubt / Missed training / Very unlikely
- 0.50: 50-50 / Late fitness test / Touch and go / Decision tomorrow
- 0.75: Minor knock / Back in training / In contention / Likely to travel
- 1.00: Confirmed Fit / 100% ready / Available to start
"""

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fpl_oracle.news.extract import TextExtractor, surname_token
from fpl_oracle.news.models import EvidenceCategory, PlayerEvidence


@dataclass(frozen=True)
class CalibratedBand:
    name: str
    p_play: float
    p_starts: float
    category: EvidenceCategory
    confidence: float


BAND_RULED_OUT = CalibratedBand(
    name="RULED_OUT",
    p_play=0.0,
    p_starts=0.0,
    category=EvidenceCategory.RULED_OUT,
    confidence=0.98,
)

BAND_MAJOR_DOUBT = CalibratedBand(
    name="MAJOR_DOUBT",
    p_play=0.25,
    p_starts=0.15,
    category=EvidenceCategory.DOUBTFUL,
    confidence=0.88,
)

BAND_TOUCH_AND_GO = CalibratedBand(
    name="TOUCH_AND_GO",
    p_play=0.50,
    p_starts=0.40,
    category=EvidenceCategory.DOUBTFUL,
    confidence=0.85,
)

BAND_IN_CONTENTION = CalibratedBand(
    name="IN_CONTENTION",
    p_play=0.75,
    p_starts=0.68,
    category=EvidenceCategory.RETURNED_TO_TRAINING,
    confidence=0.85,
)

BAND_AVAILABLE = CalibratedBand(
    name="AVAILABLE",
    p_play=1.00,
    p_starts=0.92,
    category=EvidenceCategory.AVAILABLE,
    confidence=0.95,
)


# Syntactic patterns for each calibrated band
PATTERNS_RULED_OUT = [
    r"\b(ruled out|out of contention|will miss|sidelined|unavailable|out for (several|a few|months|weeks))\b",
    r"\b(underwent surgery|torn (acl|hamstring|groin|meniscus)|fracture|broken (foot|leg|arm))\b",
    r"\b(no chance of playing|not available for selection|suspended|ineligible)\b",
    r"\b(has not recovered|won't be available|will not feature|not in the squad)\b",
    r"\b(long[- ]term (absentee|injury|layoff))\b",
]

PATTERNS_MAJOR_DOUBT = [
    r"\b(major doubt|big doubt|very unlikely|huge doubt|struggling to make it)\b",
    r"\b(hasn't trained all week|has not trained with the group|did not train)\b",
    r"\b(hard to see him (play|involved|featuring))\b",
    r"\b(still feeling pain|felt a twinge|setback in (training|recovery))\b",
]

PATTERNS_TOUCH_AND_GO = [
    r"\b(touch and go|late (fitness )?test|decision (will be made|tomorrow))\b",
    r"\b(50-50|fifty-fifty|day[- ]by[- ]day|we will assess him)\b",
    r"\b(we'll see (tomorrow|how he responds|in the morning))\b",
    r"\b(could feature|possibility he plays|might make the bench)\b",
]

PATTERNS_IN_CONTENTION = [
    r"\b(in contention|back in training|trained with the squad|trained today)\b",
    r"\b(minor knock|slight knock|precautionary|just a niggle|minor issue)\b",
    r"\b(should be fine|expected to travel|ready to be involved|in the squad)\b",
    r"\b(made good progress|recovering well|ahead of schedule)\b",
]

PATTERNS_AVAILABLE = [
    r"\b(fully fit|100% (fit|ready)|available for selection|ready to start)\b",
    r"\b(passed fit|all clear|no injury issues|completely fine)\b",
    r"\b(trained all week|started in midweek|no concerns)\b",
]


class LocalCalibratedNLPEngine:
    """
    100% offline, deterministic NLP processor classifying manager quotes
    into five mathematically calibrated probability bands.
    """

    def __init__(self):
        self._text_extractor = TextExtractor()

    def classify_statement(self, text: str) -> tuple[CalibratedBand, str, str]:
        """
        Analyzes a textual sentence/clause and returns:
        (CalibratedBand, matched_pattern, matched_quote)
        """
        lower = text.lower()

        # Reject speculative questions or interrogative inquiry headlines
        if "?" in text or re.match(
            r"^(?:which\s+game|will\s+\w+\s+(?:miss|play|start)|could\s+\w+\s+(?:miss|play|start)|why\s+is\s+\w+\s+(?:out|missing))\b",
            text.strip(),
            re.IGNORECASE,
        ):
            return BAND_AVAILABLE, "speculative question", text.strip()

        # Reject historical / narrative background recaps (debuts, season openers, past seasons)
        if re.search(
            r"\b(?:season-opener|came on for (?:his )?debut|made (?:his )?debut|debut in|in their (?:first|second|third) match|earlier this season|last season|former club|signed in the summer|arrived from)\b",
            lower,
        ) and not re.search(
            r"\b(?:this weekend|tonight|tomorrow|saturday|sunday|monday|tuesday|wednesday|thursday|friday|next (?:game|match|fixture|round)|gw|gameweek)\b",
            lower,
        ):
            return BAND_AVAILABLE, "historical narrative recap", text.strip()

        # Check for negation overrides (e.g. "not ruled out", "not a major concern")
        is_negated_doubt = bool(re.search(r"\bnot\s+(ruled out|a doubt|serious|out)\b", lower))
        if is_negated_doubt:
            return BAND_IN_CONTENTION, "not ruled out/not serious", text.strip()

        is_negated_fit = bool(re.search(r"\bnot\s+(fit|ready|available|100%)\b", lower))
        if is_negated_fit:
            return BAND_MAJOR_DOUBT, "not fit/not ready", text.strip()

        # Check for affirmative availability overrides even if a contrastive miss is mentioned
        # (e.g. "while they will miss X for trip to Anfield, he will be available to face United")
        if re.search(
            r"\b(?:will\s+be\s+available|available\s+to\s+face|fit\s+to\s+face|ready\s+to\s+face|cleared\s+to\s+face|available\s+for\s+selection)\b",
            lower,
        ) or re.search(
            r"\b(?:while|although|though)\b.*\b(?:will\s+miss|misses|out\s+for|sidelined)\b.*\b(?:will\s+be\s+available|available\s+to|fit\s+to|ready\s+to)\b",
            lower,
        ):
            return BAND_AVAILABLE, "confirmed available for upcoming match", text.strip()

        # Match in hierarchical order from most severe (ruled out) to fully fit
        for pat in PATTERNS_RULED_OUT:
            match = re.search(pat, lower)
            if match:
                return BAND_RULED_OUT, match.group(0), text.strip()

        for pat in PATTERNS_MAJOR_DOUBT:
            match = re.search(pat, lower)
            if match:
                return BAND_MAJOR_DOUBT, match.group(0), text.strip()

        for pat in PATTERNS_TOUCH_AND_GO:
            match = re.search(pat, lower)
            if match:
                return BAND_TOUCH_AND_GO, match.group(0), text.strip()

        for pat in PATTERNS_IN_CONTENTION:
            match = re.search(pat, lower)
            if match:
                return BAND_IN_CONTENTION, match.group(0), text.strip()

        for pat in PATTERNS_AVAILABLE:
            match = re.search(pat, lower)
            if match:
                return BAND_AVAILABLE, match.group(0), text.strip()

        # Default fallback if sports context exists but non-committal
        if re.search(r"\b(injury|fitness|knock|hamstring|ankle|knee)\b", lower):
            return BAND_TOUCH_AND_GO, "general injury reference", text.strip()

        return BAND_AVAILABLE, "no injury detected", text.strip()

    def extract_calibrated_evidence(
        self,
        article_text: str,
        article_url: str,
        published_dt: datetime | None,
        candidate_roster: list[dict[str, Any]],
        target_gw: int | None = None,
    ) -> list[PlayerEvidence]:
        """
        Scans article text for candidate players, isolates per-sentence attribution,
        and generates calibrated PlayerEvidence objects.
        """
        clean_text = self._text_extractor.clean_html(article_text)
        if not clean_text or not candidate_roster:
            return []

        # Check for cup competition context across article text or URL
        match_context = None
        clean_lower = clean_text.lower()
        url_lower = (article_url or "").lower()
        if (
            re.search(r"\b(?:carabao\s+cup|carabao|efl\s+cup|league\s+cup)\b", clean_lower)
            or "carabao" in url_lower
            or "efl-cup" in url_lower
        ):
            match_context = "Carabao Cup"
        elif re.search(r"\b(?:fa\s+cup)\b", clean_lower) or "fa-cup" in url_lower:
            match_context = "FA Cup"
        elif re.search(r"\b(?:champions\s+league|ucl)\b", clean_lower) or "champions-league" in url_lower:
            match_context = "Champions League"
        elif re.search(r"\b(?:europa\s+league)\b", clean_lower) or "europa-league" in url_lower:
            match_context = "Europa League"
        elif re.search(r"\b(?:conference\s+league)\b", clean_lower) or "conference-league" in url_lower:
            match_context = "Conference League"
        elif re.search(r"\b(?:nations\s+league)\b", clean_lower) or "nations-league" in url_lower:
            match_context = "Nations League"

        # Split into sentences for strict clause attribution
        sentences = re.split(r"(?<=[.!?])\s+", clean_text)
        evidences: list[PlayerEvidence] = []
        now = datetime.now(UTC)

        for player in candidate_roster:
            pid = int(player.get("id") or player.get("element", 0))
            web_name = player.get("web_name", "")
            s_token = surname_token(web_name)
            if len(s_token) < 3:
                continue

            # Look for player name or surname in sentences
            player_pat = re.compile(rf"\b{re.escape(s_token)}\b", re.IGNORECASE)

            for sentence in sentences:
                if not player_pat.search(sentence):
                    continue

                # Clause segmentation & multi-subject disambiguation
                delims = list(
                    re.finditer(
                        r"\b(?:but|while|whereas|when|after|because|as|since|although|though|until|amid|amidst|following|with)\b|;",
                        sentence,
                        re.IGNORECASE,
                    )
                )
                target_text = sentence
                if delims:
                    cuts = [0] + [d.start() for d in delims] + [len(sentence)]
                    clauses = [sentence[cuts[i] : cuts[i + 1]].strip() for i in range(len(cuts) - 1)]
                    p_clause_idx = next(
                        (i for i, c in enumerate(clauses) if player_pat.search(c)),
                        None,
                    )
                    if p_clause_idx is not None:
                        other_clause_injury = False
                        for i, c in enumerate(clauses):
                            if i != p_clause_idx:
                                m_subj_inj = re.search(
                                    r"\b(?:after|when|as|because|since|with)\s+([A-Z][a-z]+)\s+(?:picked up|suffered|sustained|underwent|had|got)\s+(?:an?\s+)?(?:injury|knock|blow|tear|surgery|problem)\b",
                                    c,
                                )
                                if m_subj_inj and m_subj_inj.group(1).lower() != s_token:
                                    other_clause_injury = True
                                    break
                        if other_clause_injury:
                            target_text = clauses[p_clause_idx]

                # Check if sentence/clause indicates the player was a replacement/substitute for someone else
                is_substitute_feature = bool(
                    re.search(
                        r"\b(?:came on (?:for|as)|replaced|deputised for|filled in for)\b",
                        target_text,
                        re.IGNORECASE,
                    )
                )

                # Sentence references player: classify the statement
                band, pattern, quote = self.classify_statement(target_text)

                if is_substitute_feature and band in (BAND_RULED_OUT, BAND_MAJOR_DOUBT, BAND_TOUCH_AND_GO):
                    band = BAND_AVAILABLE
                    pattern = "substitute appearance"

                # Skip trivial or non-actionable sentences
                if band == BAND_AVAILABLE and pattern in (
                    "no injury detected",
                    "historical narrative recap",
                    "speculative question",
                    "substitute appearance",
                ):
                    continue

                ev = PlayerEvidence(
                    player_id=pid,
                    player_name=web_name,
                    team_name=player.get("team_name", ""),
                    team_id=player.get("team"),
                    category=band.category,
                    quote=quote[:300],
                    match_context=match_context,
                    target_gw=target_gw,
                    confidence=band.confidence,
                    source_url=article_url,
                    published_at=published_dt or now,
                    extracted_at=now,
                    ambiguity_notes=f"Calibrated band: {band.name} (p_play={band.p_play:.2f}, p_starts={band.p_starts:.2f})",
                )
                evidences.append(ev)
                break  # Keep first authoritative mention per article

        return evidences


local_nlp_engine = LocalCalibratedNLPEngine()
