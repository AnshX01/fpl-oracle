"""
SQLite Database store via SQLAlchemy.
Persists API cache, snapshots, user profile, decisions log, and chat sessions.
"""

import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float, DateTime, Boolean, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from fpl_oracle.config import DB_PATH, BASE_DIR

PROFILE_JSON_PATH = BASE_DIR / "data" / "profile.json"

Base = declarative_base()

class APICacheEntry(Base):
    __tablename__ = "api_cache"
    key = Column(String(255), primary_key=True)
    data_json = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    expires_at = Column(DateTime, nullable=False)

class RawSnapshot(Base):
    __tablename__ = "raw_snapshots"
    id = Column(Integer, primary_key=True, autoincrement=True)
    endpoint = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    payload_json = Column(Text, nullable=False)

class UserProfile(Base):
    __tablename__ = "user_profile"
    id = Column(Integer, primary_key=True, default=1)
    manager_id = Column(Integer, nullable=True)
    target_league_id = Column(Integer, nullable=True)
    risk_preference = Column(String(50), default="balanced")
    llm_provider = Column(String(50), default="gemini")
    bank = Column(Float, default=0.0) # in millions, e.g. 1.5
    free_transfers = Column(Integer, default=1)
    manual_squad = Column(Text, nullable=True) # JSON list of element IDs
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

class DecisionRecord(Base):
    __tablename__ = "decision_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    gameweek = Column(Integer, nullable=False)
    decision_type = Column(String(50), nullable=False) # transfer, captain, chip, lineup
    recommendation = Column(Text, nullable=False)
    user_choice = Column(Text, nullable=True)
    expected_points = Column(Float, nullable=True)
    actual_points = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), default="default")
    role = Column(String(20), nullable=False) # user, assistant, system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# Engine and session initialization
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Create tables if they do not exist and ensure schema is up to date."""
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE user_profile ADD COLUMN manual_squad TEXT"))
            conn.commit()
        except Exception:
            pass

init_db()

class DataStore:
    def __init__(self):
        self.session_factory = SessionLocal

    def get_session(self) -> Session:
        return self.session_factory()

    def get_cache(self, key: str) -> Optional[Dict[str, Any]]:
        with self.get_session() as session:
            entry = session.query(APICacheEntry).filter(APICacheEntry.key == key).first()
            if entry:
                now = datetime.now(timezone.utc)
                exp = entry.expires_at
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                if exp > now:
                    try:
                        return json.loads(entry.data_json)
                    except Exception:
                        return None
        return None

    def get_stale_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Fallback to any existing cache when API is unreachable."""
        with self.get_session() as session:
            entry = session.query(APICacheEntry).filter(APICacheEntry.key == key).first()
            if entry:
                try:
                    return json.loads(entry.data_json)
                except Exception:
                    return None
        return None

    def set_cache(self, key: str, data: Any, ttl_seconds: int):
        with self.get_session() as session:
            now = datetime.now(timezone.utc)
            expires_at = datetime.fromtimestamp(now.timestamp() + ttl_seconds, tz=timezone.utc)
            data_str = json.dumps(data)
            entry = session.query(APICacheEntry).filter(APICacheEntry.key == key).first()
            if entry:
                entry.data_json = data_str
                entry.updated_at = now
                entry.expires_at = expires_at
            else:
                entry = APICacheEntry(key=key, data_json=data_str, updated_at=now, expires_at=expires_at)
                session.add(entry)
            session.commit()

    def save_snapshot(self, endpoint: str, data: Any):
        with self.get_session() as session:
            rec = RawSnapshot(
                endpoint=endpoint,
                created_at=datetime.now(timezone.utc),
                payload_json=json.dumps(data)
            )
            session.add(rec)
            session.commit()

    def _sync_profile_json(self, profile: UserProfile):
        """Persist profile state to data/profile.json."""
        try:
            PROFILE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "manager_id": profile.manager_id,
                "target_league_id": profile.target_league_id,
                "risk_preference": profile.risk_preference,
                "llm_provider": profile.llm_provider,
                "bank": profile.bank,
                "free_transfers": profile.free_transfers,
                "manual_squad": json.loads(profile.manual_squad) if profile.manual_squad else None,
                "updated_at": profile.updated_at.isoformat() if profile.updated_at else datetime.now(timezone.utc).isoformat()
            }
            with open(PROFILE_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def get_profile(self) -> UserProfile:
        with self.get_session() as session:
            profile = session.query(UserProfile).filter(UserProfile.id == 1).first()
            if not profile:
                profile = UserProfile(id=1, manager_id=None, target_league_id=None)
                if PROFILE_JSON_PATH.exists():
                    try:
                        with open(PROFILE_JSON_PATH, "r", encoding="utf-8") as f:
                            p_data = json.load(f)
                            profile.manager_id = p_data.get("manager_id")
                            profile.target_league_id = p_data.get("target_league_id")
                            profile.risk_preference = p_data.get("risk_preference", "balanced")
                            profile.llm_provider = p_data.get("llm_provider", "gemini")
                            profile.bank = p_data.get("bank", 0.0)
                            profile.free_transfers = p_data.get("free_transfers", 1)
                            if p_data.get("manual_squad"):
                                profile.manual_squad = json.dumps(p_data["manual_squad"])
                    except Exception:
                        pass
                session.add(profile)
                session.commit()
                session.refresh(profile)

            if not PROFILE_JSON_PATH.exists():
                self._sync_profile_json(profile)

            return UserProfile(
                id=profile.id,
                manager_id=profile.manager_id,
                target_league_id=profile.target_league_id,
                risk_preference=profile.risk_preference,
                llm_provider=profile.llm_provider,
                bank=profile.bank,
                free_transfers=profile.free_transfers,
                manual_squad=profile.manual_squad,
                updated_at=profile.updated_at
            )

    def update_profile(self, **kwargs):
        with self.get_session() as session:
            profile = session.query(UserProfile).filter(UserProfile.id == 1).first()
            if not profile:
                profile = UserProfile(id=1)
                session.add(profile)
            for k, v in kwargs.items():
                if k == "manual_squad" and isinstance(v, list):
                    setattr(profile, "manual_squad", json.dumps(v))
                elif hasattr(profile, k) and v is not None:
                    setattr(profile, k, v)
            profile.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(profile)
            self._sync_profile_json(profile)

    def log_decision(self, gameweek: int, decision_type: str, recommendation: str, expected_points: Optional[float] = None):
        with self.get_session() as session:
            rec = DecisionRecord(
                gameweek=gameweek,
                decision_type=decision_type,
                recommendation=recommendation,
                expected_points=expected_points,
                created_at=datetime.now(timezone.utc)
            )
            session.add(rec)
            session.commit()

    def get_chat_history(self, session_id: str = "default", limit: int = 50) -> List[Dict[str, str]]:
        with self.get_session() as session:
            msgs = (
                session.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
                .limit(limit)
                .all()
            )
            return [{"role": m.role, "content": m.content} for m in msgs]

    def add_chat_message(self, role: str, content: str, session_id: str = "default"):
        with self.get_session() as session:
            msg = ChatMessage(session_id=session_id, role=role, content=content, created_at=datetime.now(timezone.utc))
            session.add(msg)
            session.commit()

data_store = DataStore()
