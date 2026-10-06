"""
Configuration loader for FPL Oracle.
Reads YAML configs from config/ and environment variables from .env.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Detect base project directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")


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


# Environment keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
FPL_MANAGER_ID = os.getenv("FPL_MANAGER_ID", "")
FPL_TARGET_LEAGUE_ID = os.getenv("FPL_TARGET_LEAGUE_ID", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Directory paths
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
MODELS_DIR = DATA_DIR / "models"
HISTORICAL_DIR = DATA_DIR / "historical"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

for d in [DATA_DIR, CACHE_DIR, SNAPSHOTS_DIR, MODELS_DIR, HISTORICAL_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "fpl_oracle.db"
