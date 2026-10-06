"""
Strict Pydantic models for football news evidence extraction,
official scout risks, and single availability reconciliation.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvidenceCategory(str, Enum):
    RULED_OUT = "ruled_out"
    DOUBTFUL = "doubtful"
    AVAILABLE = "available"
    RETURNED_TO_TRAINING = "returned_to_training"
    MINUTES_LIMIT = "minutes_limit"
    SELECTION_STATEMENT = "selection_statement"
    INELIGIBLE = "ineligible"
    UNKNOWN = "unknown"


class PlayerEvidence(BaseModel):
    """
    Structured extraction schema for LLM evidence extractor.
    Enforces factual quotes, negation handling, and canonical entity links.
    """
    player_id: int = Field(description="Canonical FPL element ID if resolved, else 0")
    player_name: str = Field(description="Exact player name referenced")
    team_name: str = Field(default="", description="Club name referenced")
    team_id: int | None = Field(default=None, description="FPL team ID if resolved")
    category: EvidenceCategory = Field(default=EvidenceCategory.UNKNOWN, description="Category of reported availability")
    quote: str = Field(default="", description="Exact verbatim manager or medical quote from article")
    quote_start_offset: int | None = Field(default=None, description="Character offset start in cleaned source text")
    quote_end_offset: int | None = Field(default=None, description="Character offset end in cleaned source text")
    target_gw: int | None = Field(default=None, description="Explicit gameweek if mentioned")
    match_context: str | None = Field(default=None, description="Context, e.g. Premier League match vs European/Cup tie")
    is_negated: bool = Field(default=False, description="True if statement is negated (e.g. 'not ruled out', 'not fit')")
    minutes_restriction: int | None = Field(default=None, description="Explicit minutes limit if mentioned by manager")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Extraction certainty score")
    ambiguity_notes: str | None = Field(default=None, description="Reason for ambiguity or conflicting signals")
    source_url: str = Field(default="", description="Canonical URL of source article")
    published_at: datetime | None = Field(default=None, description="Publication timestamp of article")
    extracted_at: datetime | None = Field(default=None, description="Extraction execution timestamp")


class ExtractedNewsPayload(BaseModel):
    """Container for batch article extractions."""
    article_url: str
    article_hash: str
    published_at: str | None = None
    fetched_at: str
    evidences: list[PlayerEvidence] = []
    extraction_model: str = "gemini-2.5-flash-lite"
    raw_response_snippet: str | None = None


class RecommendationMode(str, Enum):
    API_ONLY = "api_only"
    SHADOW = "shadow"
    GATED_ACTIVE = "gated_active"


class ReconciledAvailability(BaseModel):
    """
    Auditable single per-player, per-fixture reconciliation outcome.
    Combines authoritative official baseline with validated text evidence without double counting.
    """
    element_id: int
    web_name: str
    team_id: int
    target_gw: int
    fixture_sub_index: int = 0
    # Official API baseline
    baseline_status: str  # "a", "d", "i", "s", "u"
    baseline_chance_of_playing: float  # 0.0 to 100.0
    official_news_text: str = ""
    # Single reconciled state
    effective_chance_of_playing: float  # 0.0 to 100.0
    p_available: float  # 0.0 to 1.0
    p_start_given_available: float  # 0.0 to 1.0
    expected_minutes_limit: float | None = None
    evidence_category: EvidenceCategory = EvidenceCategory.UNKNOWN
    source_quote: str | None = None
    source_url: str | None = None
    source_published_age_hours: float | None = None
    reconciliation_reason: str = "Official FPL API baseline"
    rejected_signals: list[str] = []
    # Recommendation application mode
    mode: RecommendationMode = RecommendationMode.SHADOW
    is_shadow_override: bool = False
    candidate_xp_delta: float = 0.0
    applied_to_production: bool = False
