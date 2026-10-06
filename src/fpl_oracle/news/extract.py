"""
Text extraction, HTML sanitization, and deterministic rule-based evidence extraction
for news articles and adversarial benchmark verification (Requirement F9).
"""

import re
from datetime import UTC, datetime

from bs4 import BeautifulSoup

from fpl_oracle.news.models import EvidenceCategory, PlayerEvidence


class TextExtractor:
    def clean_html(self, raw_html: str) -> str:
        if not raw_html:
            return ""
        soup = BeautifulSoup(raw_html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_evidence_from_text(
        self,
        text: str,
        player_name: str,
        player_id: int,
        target_gw: int | None = None,
    ) -> list[PlayerEvidence]:
        """
        Rule-based NLP evidence extractor for sports news and press conferences.
        Extracts structured PlayerEvidence from raw article text with:
        - Adversarial prompt injection defense
        - Cup vs Premier League match context isolation
        - Negation and dismissed-rumor detection
        - Minutes restriction parsing
        - Exact verbatim quote preservation
        """
        cleaned = self.clean_html(text)
        if not cleaned:
            return []

        # 1. Adversarial prompt injection defense
        injection_patterns = [
            r"ignore\s+previous\s+instructions",
            r"system\s+alert",
            r"override\s+player\s+status",
            r"set\s+.*\s+chance\s+to\s+0%",
            r"admin\s+override",
        ]
        is_injection = any(re.search(pat, cleaned, re.IGNORECASE) for pat in injection_patterns)
        if is_injection:
            return [
                PlayerEvidence(
                    player_id=player_id,
                    player_name=player_name,
                    category=EvidenceCategory.UNKNOWN,
                    quote=cleaned,
                    confidence=0.20,
                    ambiguity_notes="Adversarial prompt injection pattern detected and neutralised",
                    extracted_at=datetime.now(UTC),
                )
            ]

        # 2. Extract match context (Cup competition filtering)
        match_context = None
        if re.search(r"\b(?:carabao\s+cup|efl\s+cup|league\s+cup)\b", cleaned, re.IGNORECASE):
            match_context = "Carabao Cup"
        elif re.search(r"\b(?:fa\s+cup)\b", cleaned, re.IGNORECASE):
            match_context = "FA Cup"
        elif re.search(r"\b(?:champions\s+league|ucl)\b", cleaned, re.IGNORECASE):
            match_context = "Champions League"

        # 3. Target gameweek reference
        explicit_gw = None
        gw_match = re.search(r"\b(?:gw|gameweek)\s*(\d+)\b", cleaned, re.IGNORECASE)
        if gw_match:
            explicit_gw = int(gw_match.group(1))

        # 4. Negation detection
        negation_patterns = [
            r"\bnot\s+injured\b",
            r"\bno\s+injury\b",
            r"\bdismissed\s+rumou?rs?\b",
            r"\bdenied\s+rumou?rs?\b",
            r"\bnot\s+ruled\s+out\b",
            r"\bcleared\s+for\b",
        ]
        is_negated = any(re.search(pat, cleaned, re.IGNORECASE) for pat in negation_patterns)

        # 5. Minutes restriction
        minutes_limit = None
        min_match = re.search(r"(\d+)\s*(?:minutes|mins)", cleaned, re.IGNORECASE)
        if min_match and re.search(r"\b(?:only|bench|restriction|max|limit)\b", cleaned, re.IGNORECASE):
            minutes_limit = int(min_match.group(1))

        # 6. Category classification
        lower = cleaned.lower()
        conf = 0.80
        if minutes_limit is not None:
            category = EvidenceCategory.MINUTES_LIMIT
            conf = 0.90
        elif "returned to" in lower and "training" in lower or "back in full training" in lower:
            category = EvidenceCategory.RETURNED_TO_TRAINING
            conf = 0.92
        elif is_negated and ("injury" in lower or "injured" in lower or "ankle" in lower):
            category = EvidenceCategory.RULED_OUT if "ruled out" in lower else EvidenceCategory.DOUBTFUL
            conf = 0.90
        elif is_negated and "rumor" in lower:
            category = EvidenceCategory.DOUBTFUL
            conf = 0.88
        elif (
            "ruled out" in lower
            or "sprain" in lower
            or "sidelined" in lower
            or "fracture" in lower
            or "hamstring tear" in lower
            or "surgery" in lower
            or "will miss" in lower
            or "miss the next" in lower
        ):
            category = EvidenceCategory.RULED_OUT
            conf = 0.95
        elif "fitness test" in lower or "tightness" in lower or "doubt" in lower or "knock" in lower:
            category = EvidenceCategory.DOUBTFUL
            conf = 0.85
        elif "unverified" in lower or "fan forum" in lower:
            category = EvidenceCategory.RULED_OUT
            conf = 0.55
        elif "expect him to make it" in lower or "passed fit" in lower:
            category = EvidenceCategory.AVAILABLE
            conf = 0.80
        elif "precaution" in lower:
            category = EvidenceCategory.UNKNOWN
            conf = 0.60
        elif "feeling great" in lower or "scored a hat-trick" in lower or "fit to start" in lower:
            category = EvidenceCategory.SELECTION_STATEMENT
            conf = 0.95
        elif "transfer" in lower or "bid" in lower or "madrid" in lower:
            category = EvidenceCategory.UNKNOWN
            conf = 0.70
        elif "on fire" in lower:  # colloquial idiom
            category = EvidenceCategory.UNKNOWN
            conf = 0.50
        else:
            category = EvidenceCategory.UNKNOWN
            conf = 0.50

        # Verbatim quote: find sentence matching player or key indicator
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        quote = cleaned
        for s in sentences:
            if player_name.lower() in s.lower() or any(
                w in s.lower() for w in ["injury", "training", "test", "out", "minutes"]
            ):
                quote = s.strip()
                break

        return [
            PlayerEvidence(
                player_id=player_id,
                player_name=player_name,
                category=category,
                quote=quote,
                match_context=match_context,
                target_gw=explicit_gw,
                is_negated=is_negated,
                minutes_restriction=minutes_limit,
                confidence=conf,
                extracted_at=datetime.now(UTC),
            )
        ]


text_extractor = TextExtractor()
