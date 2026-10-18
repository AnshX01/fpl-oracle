"""
Configuration loader for FPL Oracle.
Reads YAML configs from config/ and environment variables from project-local .env.
Ensures project .env takes strict precedence over inherited shell environment variables.
Provides typed Pydantic configuration loader and redacted diagnostics.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Detect base project directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables: project .env strictly overrides inherited environment variables
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE, override=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Load config files
CONFIG_DIR = BASE_DIR / "config"
SETTINGS: dict[str, Any] = _load_yaml(CONFIG_DIR / "settings.yaml")
RULES: dict[str, Any] = _load_yaml(CONFIG_DIR / "rules.yaml")
NEWS_SOURCES: dict[str, Any] = _load_yaml(CONFIG_DIR / "news_sources.yaml")


def load_rules_config() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "rules.yaml")


def load_scoring_config() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "scoring.yaml")


def is_flag_enabled(flag_name: str, default: bool = False) -> bool:
    """Check whether a feature flag is enabled in config/flags.yaml."""
    flags_path = CONFIG_DIR / "flags.yaml"
    if not flags_path.exists():
        return default
    try:
        data = _load_yaml(flags_path)
        return bool(data.get("flags", {}).get(flag_name, default))
    except Exception:
        return default


class FPLSettings(BaseModel):
    """Typed runtime settings loaded strictly from project-local .env."""

    # FPL Manager Credentials
    fpl_manager_id: int | None = Field(default=None, description="Official FPL manager / entry ID")
    fpl_target_league_id: int | None = Field(default=None, description="Target mini-league ID for rival analysis")

    # Google Gemini Free-Tier Configuration
    gemini_api_key: str = Field(default="", description="Google AI Studio Gemini API Key")
    gemini_free_tier_confirmed: bool = Field(
        default=False, description="Confirmation that project is non-billable Free Tier"
    )
    gemini_model: str = Field(default="gemini-flash-lite-latest", description="Supported Gemini free model")

    # Recommendation Mode (press, shadow, api_only, gated_active)
    news_recommendation_mode: str = Field(default="press", description="Application mode for candidate news evidence")
    gated_active_opt_in: bool = Field(
        default=False, description="Explicit opt-in to apply candidate news to production"
    )
    news_gate_min_precision: float = Field(
        default=0.85, description="Minimum benchmark precision required to gate news into production"
    )
    news_gate_min_recall: float = Field(
        default=0.80, description="Minimum benchmark recall required to gate news into production"
    )
    news_gate_max_false_ruled_out: float = Field(
        default=0.02, description="Maximum benchmark false ruled-out rate allowed to gate news into production"
    )

    # Optional Providers
    llm_provider: str = Field(default="gemini", description="LLM provider for chat/briefings")
    anthropic_api_key: str = Field(default="", description="Optional Anthropic API Key")
    openai_api_key: str = Field(default="", description="Optional OpenAI API Key")

    # League & Rival Intelligence Settings
    rival_points_window: int = Field(
        default=0, description="Fixed points window below the user for rival selection; 0 = everyone still in reach"
    )
    max_standings_pages: int = Field(default=200, description="High safety limit for mini-league standings pagination")

    # Clean Sheet Intelligence Settings
    default_clean_sheet_probability: float = Field(default=0.30, description="Baseline clean sheet prior probability")
    clean_sheet_home_factor: float = Field(
        default=1.15, description="Home fixture multiplier for clean sheet probability"
    )
    clean_sheet_away_factor: float = Field(
        default=0.85, description="Away fixture multiplier for clean sheet probability"
    )

    # Joint Multi-GW Planner Settings
    planner_horizon: int = Field(default=5, description="Default planning horizon in gameweeks for joint search")
    odds_api_key: str = Field(default="", description="Optional free The Odds API key for earlier upcoming odds")
    planner_min_horizon: int = Field(default=6, description="Fewest gameweeks the main plan looks ahead")
    planner_max_horizon: int = Field(default=12, description="Most gameweeks the main plan looks ahead")

    def get_redacted_status(self) -> dict[str, Any]:
        """Return diagnostic status without exposing secrets or private IDs."""
        mgr_set = self.fpl_manager_id is not None
        lg_set = self.fpl_target_league_id is not None
        gem_key_set = bool(self.gemini_api_key and self.gemini_api_key.strip())

        status_msg = "Awaiting Gemini key & Free-tier confirmation; Official API baseline active"
        if gem_key_set and self.gemini_free_tier_confirmed:
            if self.news_recommendation_mode == "press":
                status_msg = "Press evidence applied under safety rules (can only lower official availability)"
            elif self.news_recommendation_mode == "gated_active" and self.gated_active_opt_in:
                status_msg = "Active in Production (Candidate news overrides applied)"
            else:
                status_msg = "Running in Shadow Mode (Candidate evidence logged; official baseline in production)"

        return {
            "manager_id_configured": mgr_set,
            "manager_id_redacted": f"***{str(self.fpl_manager_id)[-3:]}" if mgr_set else "Not configured in .env",
            "league_id_configured": lg_set,
            "league_id_redacted": f"***{str(self.fpl_target_league_id)[-3:]}" if lg_set else "Not configured in .env",
            "gemini_api_key_configured": gem_key_set,
            "gemini_free_tier_confirmed": self.gemini_free_tier_confirmed,
            "gemini_model": self.gemini_model,
            "recommendation_mode": self.news_recommendation_mode,
            "gated_active_opt_in": self.gated_active_opt_in,
            "llm_extractor_status": status_msg,
        }


def _load_settings_from_env() -> FPLSettings:
    def _parse_int(val: str | None) -> int | None:
        if val and val.strip().isdigit():
            return int(val.strip())
        return None

    def _parse_bool(val: str | None) -> bool:
        if not val:
            return False
        return val.strip().lower() in ("true", "1", "yes")

    def _parse_float(val: str | None, default: float) -> float:
        if val:
            try:
                return float(val.strip())
            except ValueError:
                pass
        return default

    return FPLSettings(
        fpl_manager_id=_parse_int(os.getenv("FPL_MANAGER_ID")),
        fpl_target_league_id=_parse_int(os.getenv("FPL_TARGET_LEAGUE_ID")),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_free_tier_confirmed=_parse_bool(os.getenv("GEMINI_FREE_TIER_CONFIRMED")),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest").strip(),
        news_recommendation_mode=os.getenv("NEWS_RECOMMENDATION_MODE", "press").strip(),
        gated_active_opt_in=_parse_bool(os.getenv("GATED_ACTIVE_OPT_IN")),
        llm_provider=os.getenv("LLM_PROVIDER", "gemini").strip(),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        rival_points_window=_parse_int(os.getenv("RIVAL_POINTS_WINDOW")) or 0,
        odds_api_key=os.getenv("ODDS_API_KEY", "").strip(),
        planner_min_horizon=_parse_int(os.getenv("PLANNER_MIN_HORIZON")) or 6,
        planner_max_horizon=_parse_int(os.getenv("PLANNER_MAX_HORIZON")) or 12,
        max_standings_pages=_parse_int(os.getenv("MAX_STANDINGS_PAGES")) or 200,
        default_clean_sheet_probability=_parse_float(os.getenv("DEFAULT_CLEAN_SHEET_PROBABILITY"), 0.30),
        clean_sheet_home_factor=_parse_float(os.getenv("CLEAN_SHEET_HOME_FACTOR"), 1.15),
        clean_sheet_away_factor=_parse_float(os.getenv("CLEAN_SHEET_AWAY_FACTOR"), 0.85),
    )


app_config = _load_settings_from_env()

# Top-level exports for backwards compatibility
GEMINI_API_KEY = app_config.gemini_api_key
NEWS_RECOMMENDATION_MODE = app_config.news_recommendation_mode
NEWS_GATE_MIN_PRECISION = app_config.news_gate_min_precision
NEWS_GATE_MIN_RECALL = app_config.news_gate_min_recall
NEWS_GATE_MAX_FALSE_RULED_OUT = app_config.news_gate_max_false_ruled_out

ANTHROPIC_API_KEY = app_config.anthropic_api_key
OPENAI_API_KEY = app_config.openai_api_key
FPL_MANAGER_ID = str(app_config.fpl_manager_id) if app_config.fpl_manager_id else ""
FPL_TARGET_LEAGUE_ID = str(app_config.fpl_target_league_id) if app_config.fpl_target_league_id else ""
RIVAL_POINTS_WINDOW = app_config.rival_points_window
MAX_STANDINGS_PAGES = app_config.max_standings_pages
DEFAULT_CLEAN_SHEET_PROBABILITY = app_config.default_clean_sheet_probability
CLEAN_SHEET_HOME_FACTOR = app_config.clean_sheet_home_factor
CLEAN_SHEET_AWAY_FACTOR = app_config.clean_sheet_away_factor
PLANNER_HORIZON = app_config.planner_horizon
ODDS_API_KEY = app_config.odds_api_key
PLANNER_MIN_HORIZON = app_config.planner_min_horizon
PLANNER_MAX_HORIZON = app_config.planner_max_horizon


def planning_horizon(start_gw: int) -> int:
    """Gameweeks the main plan covers: through the current chip set's expiry, but never fewer than
    PLANNER_MIN_HORIZON (no myopic plans near GW19/GW38) or more than PLANNER_MAX_HORIZON (solve time)."""
    to_expiry = (20 if start_gw <= 19 else 39) - start_gw
    return max(1, min(39 - start_gw, PLANNER_MAX_HORIZON, max(PLANNER_MIN_HORIZON, to_expiry)))


# Directory paths
# FPL_ORACLE_DATA_DIR lets tests run against a copy instead of the live data/ folder.
DATA_DIR = Path(os.environ.get("FPL_ORACLE_DATA_DIR") or BASE_DIR / "data")
CACHE_DIR = DATA_DIR / "cache"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
MODELS_DIR = DATA_DIR / "models"
HISTORICAL_DIR = DATA_DIR / "historical"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"
HOLDOUT_DIR = DATA_DIR / "holdout"

for d in [DATA_DIR, CACHE_DIR, SNAPSHOTS_DIR, MODELS_DIR, HISTORICAL_DIR, REPORTS_DIR, LOGS_DIR, HOLDOUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "fpl_oracle.db"
