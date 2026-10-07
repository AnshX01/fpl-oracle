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
SCORING: dict[str, Any] = _load_yaml(CONFIG_DIR / "scoring.yaml")
NEWS_SOURCES: dict[str, Any] = _load_yaml(CONFIG_DIR / "news_sources.yaml")


def load_rules_config() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "rules.yaml")


def load_scoring_config() -> dict[str, Any]:
    return _load_yaml(CONFIG_DIR / "scoring.yaml")


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
    gemini_model: str = Field(default="gemini-2.5-flash-lite", description="Supported Gemini free model")
    gemini_daily_request_limit: int = Field(default=150, description="Max daily requests on free tier")
    gemini_daily_token_limit: int = Field(default=500_000, description="Max daily tokens on free tier")

    # Recommendation Mode (api_only, shadow, gated_active)
    news_recommendation_mode: str = Field(default="shadow", description="Application mode for candidate news evidence")
    gated_active_opt_in: bool = Field(
        default=False, description="Explicit opt-in to apply candidate news to production"
    )

    # Optional Providers
    odds_api_key: str = Field(
        default="", description="Dormant/deprecated (N8): zero-cost guarantee uses continuous Dixon-Coles team ratings"
    )
    llm_provider: str = Field(default="gemini", description="LLM provider for chat/briefings")
    anthropic_api_key: str = Field(default="", description="Optional Anthropic API Key")
    openai_api_key: str = Field(default="", description="Optional OpenAI API Key")
    tavily_api_key: str = Field(
        default="", description="Dormant/deprecated (N8): zero-cost guarantee uses free RSS feeds"
    )
    brave_api_key: str = Field(
        default="", description="Dormant/deprecated (N8): zero-cost guarantee uses free RSS feeds"
    )
    discord_webhook_url: str = Field(default="", description="Optional Discord webhook")
    telegram_bot_token: str = Field(default="", description="Optional Telegram bot token")
    telegram_chat_id: str = Field(default="", description="Optional Telegram chat ID")

    # League & Rival Intelligence Settings
    rival_points_window: int = Field(default=20, description="Points window below user for proximity rival selection")
    max_standings_pages: int = Field(default=200, description="High safety limit for mini-league standings pagination")

    def get_redacted_status(self) -> dict[str, Any]:
        """Return diagnostic status without exposing secrets or private IDs."""
        mgr_set = self.fpl_manager_id is not None
        lg_set = self.fpl_target_league_id is not None
        gem_key_set = bool(self.gemini_api_key and self.gemini_api_key.strip())

        status_msg = "Awaiting Gemini key & Free-tier confirmation; Official API baseline active"
        if gem_key_set and self.gemini_free_tier_confirmed:
            if self.news_recommendation_mode == "gated_active" and self.gated_active_opt_in:
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

    return FPLSettings(
        fpl_manager_id=_parse_int(os.getenv("FPL_MANAGER_ID")),
        fpl_target_league_id=_parse_int(os.getenv("FPL_TARGET_LEAGUE_ID")),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_free_tier_confirmed=_parse_bool(os.getenv("GEMINI_FREE_TIER_CONFIRMED")),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip(),
        gemini_daily_request_limit=_parse_int(os.getenv("GEMINI_DAILY_REQUEST_BUDGET")) or 150,
        gemini_daily_token_limit=_parse_int(os.getenv("GEMINI_DAILY_TOKEN_BUDGET")) or 500_000,
        news_recommendation_mode=os.getenv("NEWS_RECOMMENDATION_MODE", "shadow").strip(),
        gated_active_opt_in=_parse_bool(os.getenv("GATED_ACTIVE_OPT_IN")),
        odds_api_key=os.getenv("ODDS_API_KEY", "").strip(),
        llm_provider=os.getenv("LLM_PROVIDER", "gemini").strip(),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        tavily_api_key=os.getenv("TAVILY_API_KEY", "").strip(),
        brave_api_key=os.getenv("BRAVE_API_KEY", "").strip(),
        discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL", "").strip(),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
        rival_points_window=_parse_int(os.getenv("RIVAL_POINTS_WINDOW")) or 20,
        max_standings_pages=_parse_int(os.getenv("MAX_STANDINGS_PAGES")) or 200,
    )


app_config = _load_settings_from_env()

# Top-level exports for backwards compatibility
GEMINI_API_KEY = app_config.gemini_api_key
GEMINI_FREE_TIER_CONFIRMED = app_config.gemini_free_tier_confirmed
GEMINI_MODEL = app_config.gemini_model
NEWS_RECOMMENDATION_MODE = app_config.news_recommendation_mode
GATED_ACTIVE_OPT_IN = app_config.gated_active_opt_in

ANTHROPIC_API_KEY = app_config.anthropic_api_key
OPENAI_API_KEY = app_config.openai_api_key
FPL_MANAGER_ID = str(app_config.fpl_manager_id) if app_config.fpl_manager_id else ""
FPL_TARGET_LEAGUE_ID = str(app_config.fpl_target_league_id) if app_config.fpl_target_league_id else ""
RIVAL_POINTS_WINDOW = app_config.rival_points_window
MAX_STANDINGS_PAGES = app_config.max_standings_pages
TAVILY_API_KEY = app_config.tavily_api_key
BRAVE_API_KEY = app_config.brave_api_key
ODDS_API_KEY = app_config.odds_api_key
DISCORD_WEBHOOK_URL = app_config.discord_webhook_url
TELEGRAM_BOT_TOKEN = app_config.telegram_bot_token
TELEGRAM_CHAT_ID = app_config.telegram_chat_id

# Directory paths
DATA_DIR = BASE_DIR / "data"
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
