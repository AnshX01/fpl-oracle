"""
SQLite Database store via SQLAlchemy.
Persists API cache, snapshots, user profile, decisions log, and chat sessions.
"""

import json
import random
import sqlite3
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    event,
    text,
)
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from fpl_oracle.config import DATA_DIR, DB_PATH

PROFILE_JSON_PATH = DATA_DIR / "profile.json"


class Base(DeclarativeBase):
    pass


class APICacheEntry(Base):
    __tablename__ = "api_cache"
    key = Column(String(255), primary_key=True)
    data_json = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC))
    expires_at = Column(DateTime, nullable=False)


class RawSnapshot(Base):
    __tablename__ = "raw_snapshots"
    id = Column(Integer, primary_key=True, autoincrement=True)
    endpoint = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), index=True)
    payload_json = Column(Text, nullable=False)


class UserProfile(Base):
    __tablename__ = "user_profile"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    manager_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    target_league_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    risk_preference: Mapped[str] = mapped_column(String(50), default="balanced")
    llm_provider: Mapped[str] = mapped_column(String(50), default="gemini")
    bank: Mapped[float] = mapped_column(Float, default=0.0)  # in millions, e.g. 1.5
    free_transfers: Mapped[int] = mapped_column(Integer, default=1)
    bank_override_enabled: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ft_override_enabled: Mapped[int | None] = mapped_column(Integer, nullable=True)
    manual_squad: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list of element IDs
    # Manager's own calls the planner must respect: JSON lists of element IDs
    keep_ids: Mapped[str | None] = mapped_column(Text, nullable=True)
    avoid_ids: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), default="default")
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class PriceSnapshotRecord(Base):
    __tablename__ = "price_snapshots"
    id = Column(Integer, primary_key=True, autoincrement=True)
    element_id = Column(Integer, nullable=False, index=True)
    web_name = Column(String(100), nullable=False)
    now_cost = Column(Float, nullable=False)
    net_transfers = Column(Integer, default=0)
    selected_by_percent = Column(Float, default=0.0)
    hourly_rate = Column(Float, default=0.0)
    urgency_score = Column(Float, default=0.0)
    direction = Column(String(50), default="STABLE")
    snapshot_time = Column(DateTime, default=lambda: datetime.now(UTC), index=True)


class JobRunRecord(Base):
    __tablename__ = "job_runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(100), nullable=False, index=True)
    job_name = Column(String(200), nullable=False)
    status = Column(String(50), nullable=False)  # SUCCESS, FAILED, RUNNING
    started_at = Column(DateTime, default=lambda: datetime.now(UTC))
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, default=0.0)
    details = Column(Text, nullable=True)


class ModelVersionRecord(Base):
    __tablename__ = "model_versions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(50), nullable=False, unique=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))
    ml_mae = Column(Float, nullable=False)
    ml_spearman = Column(Float, nullable=False)
    base_mae = Column(Float, nullable=False)
    is_active = Column(Integer, default=0)
    status = Column(String(50), default="production")  # production, archived, rejected_rollback
    notes = Column(Text, nullable=True)


# Engine and session initialization
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False, "timeout": 30.0},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        pass
    finally:
        cursor.close()


def retry_on_db_lock(max_retries: int = 5, base_delay: float = 0.05, max_delay: float = 0.5):
    """
    Decorator for database write transactions.
    Retries on SQLite database locked / busy OperationalError with exponential
    backoff and jitter to ensure extreme concurrency resilience under heavy parallel writes.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (OperationalError, DBAPIError, sqlite3.OperationalError) as e:
                    err_msg = str(e).lower()
                    if "locked" in err_msg or "busy" in err_msg:
                        last_err = e
                        if attempt < max_retries - 1:
                            delay = min(max_delay, base_delay * (2**attempt)) + random.uniform(0.01, 0.05)
                            time.sleep(delay)
                            continue
                    raise
            if last_err:
                raise last_err

        return wrapper

    return decorator


def init_db():
    """Create tables if they do not exist and ensure schema is up to date."""
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        try:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.execute(text("PRAGMA busy_timeout=30000"))
            conn.commit()
        except Exception:
            pass
    # Additive SQLite migration preserves all existing data and NOT NULL columns.
    with engine.begin() as conn:
        names = {r[1] for r in conn.execute(text("PRAGMA table_info(user_profile)"))}
        for name in ("bank_override_enabled", "ft_override_enabled"):
            if name not in names:
                conn.execute(text(f"ALTER TABLE user_profile ADD COLUMN {name} INTEGER"))
        for name in ("keep_ids", "avoid_ids"):
            if name not in names:
                conn.execute(text(f"ALTER TABLE user_profile ADD COLUMN {name} TEXT"))
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE user_profile ADD COLUMN manual_squad TEXT"))
            conn.commit()
        except Exception:
            pass
        for idx_stmt in (
            "CREATE INDEX IF NOT EXISTS ix_raw_snapshots_endpoint_created ON raw_snapshots (endpoint, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS ix_api_cache_expires ON api_cache (expires_at)",
            "CREATE INDEX IF NOT EXISTS ix_chat_messages_session ON chat_messages (session_id, created_at DESC, id DESC)",
            "CREATE INDEX IF NOT EXISTS ix_job_runs_started ON job_runs (started_at DESC)",
        ):
            try:
                conn.execute(text(idx_stmt))
                conn.commit()
            except Exception:
                pass


init_db()


@dataclass
class ProfileData:
    id: int = 1
    manager_id: int | None = None
    target_league_id: int | None = None
    risk_preference: str = "balanced"
    llm_provider: str = "gemini"
    bank: float = 0.0
    free_transfers: int = 1
    bank_override_enabled: bool | None = None
    ft_override_enabled: bool | None = None
    manual_squad: str | None = None
    keep_ids: list[int] = field(default_factory=list)
    avoid_ids: list[int] = field(default_factory=list)
    updated_at: datetime | None = None


class DataStore:
    def __init__(self):
        self.session_factory = SessionLocal

    def get_session(self) -> Session:
        return self.session_factory()

    @contextmanager
    def transaction(self) -> Generator[Session, None, None]:
        """
        Safe transaction context manager with automatic rollback on error.
        Ensures connections returned to pool never carry uncommitted or aborted transaction locks.
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_cache_entry(self, key: str) -> tuple[dict[str, Any], datetime, datetime] | None:
        """Return (data, updated_at, expires_at) if valid cache entry exists."""
        with self.get_session() as session:
            entry = session.query(APICacheEntry).filter(APICacheEntry.key == key).first()
            if entry:
                now = datetime.now(UTC)
                exp = entry.expires_at
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=UTC)
                if exp > now:
                    try:
                        upd = entry.updated_at
                        if upd and upd.tzinfo is None:
                            upd = upd.replace(tzinfo=UTC)
                        upd_dt: datetime = upd if isinstance(upd, datetime) else now
                        exp_dt: datetime = exp if isinstance(exp, datetime) else now
                        return json.loads(str(entry.data_json)), upd_dt, exp_dt
                    except Exception:
                        return None
        return None

    def get_stale_cache_entry(self, key: str) -> tuple[dict[str, Any], datetime] | None:
        """Return (data, updated_at) for any existing cache entry when API is unreachable."""
        with self.get_session() as session:
            entry = session.query(APICacheEntry).filter(APICacheEntry.key == key).first()
            if entry:
                try:
                    upd = entry.updated_at
                    if upd and upd.tzinfo is None:
                        upd = upd.replace(tzinfo=UTC)
                    upd_dt: datetime = upd if isinstance(upd, datetime) else datetime.now(UTC)
                    return json.loads(str(entry.data_json)), upd_dt
                except Exception:
                    return None
        return None

    @retry_on_db_lock()
    def delete_cache_entry(self, key: str):
        """Delete a cache entry from persistent SQLite storage."""
        with self.transaction() as session:
            entry = session.query(APICacheEntry).filter(APICacheEntry.key == key).first()
            if entry:
                session.delete(entry)

    @retry_on_db_lock()
    def set_cache(self, key: str, data: Any, ttl_seconds: int):
        with self.transaction() as session:
            now = datetime.now(UTC)
            expires_at = datetime.fromtimestamp(now.timestamp() + ttl_seconds, tz=UTC)
            data_str = json.dumps(data)
            entry = session.query(APICacheEntry).filter(APICacheEntry.key == key).first()
            if entry:
                entry.data_json = data_str  # type: ignore[assignment]
                entry.updated_at = now  # type: ignore[assignment]
                entry.expires_at = expires_at  # type: ignore[assignment]
            else:
                entry = APICacheEntry(key=key, data_json=data_str, updated_at=now, expires_at=expires_at)
                session.add(entry)

    @retry_on_db_lock()
    def save_snapshot(self, endpoint: str, data: Any):
        with self.transaction() as session:
            rec = RawSnapshot(endpoint=endpoint, created_at=datetime.now(UTC), payload_json=json.dumps(data))
            session.add(rec)

    def get_latest_snapshot(self, endpoint: str) -> tuple[dict[str, Any], datetime] | None:
        """Retrieve most recent raw snapshot as ultimate fallback."""
        with self.get_session() as session:
            rec = (
                session.query(RawSnapshot)
                .filter(RawSnapshot.endpoint == endpoint)
                .order_by(RawSnapshot.created_at.desc())
                .first()
            )
            if rec:
                try:
                    upd = rec.created_at
                    if upd and upd.tzinfo is None:
                        upd = upd.replace(tzinfo=UTC)
                    return json.loads(rec.payload_json), upd or datetime.now(UTC)
                except Exception:
                    return None
        return None

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
                "bank_override_enabled": profile.bank_override_enabled,
                "ft_override_enabled": profile.ft_override_enabled,
                "free_transfers": profile.free_transfers,
                "manual_squad": json.loads(str(profile.manual_squad)) if profile.manual_squad else None,
                "updated_at": profile.updated_at.isoformat() if profile.updated_at else datetime.now(UTC).isoformat(),
            }
            with open(PROFILE_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def get_profile(self) -> ProfileData:
        with self.get_session() as session:
            profile = session.query(UserProfile).filter(UserProfile.id == 1).first()
            if not profile:
                profile = UserProfile(
                    id=1, manager_id=None, target_league_id=None, bank_override_enabled=0, ft_override_enabled=0
                )
                if PROFILE_JSON_PATH.exists():
                    try:
                        with open(PROFILE_JSON_PATH, encoding="utf-8") as f:
                            p_data = json.load(f)
                            profile.manager_id = p_data.get("manager_id")
                            profile.target_league_id = p_data.get("target_league_id")
                            profile.risk_preference = p_data.get("risk_preference", "balanced")
                            profile.llm_provider = p_data.get("llm_provider", "gemini")
                            profile.bank = p_data.get("bank", 0.0)
                            profile.free_transfers = p_data.get("free_transfers", 1)
                            profile.bank_override_enabled = p_data.get(
                                "bank_override_enabled", 1 if "bank" in p_data else 0
                            )
                            profile.ft_override_enabled = p_data.get(
                                "ft_override_enabled", 1 if "free_transfers" in p_data else 0
                            )
                            if p_data.get("manual_squad"):
                                profile.manual_squad = json.dumps(p_data["manual_squad"])  # type: ignore[assignment]
                    except Exception:
                        pass

                # If manager_id or league_id still unconfigured, seed from environment
                from fpl_oracle.config import FPL_MANAGER_ID, FPL_TARGET_LEAGUE_ID

                if profile.manager_id is None and FPL_MANAGER_ID:
                    try:
                        val = int(str(FPL_MANAGER_ID).strip())
                        if val > 0:
                            profile.manager_id = val
                    except (ValueError, TypeError):
                        pass

                if profile.target_league_id is None and FPL_TARGET_LEAGUE_ID:
                    try:
                        val = int(str(FPL_TARGET_LEAGUE_ID).strip())
                        if val > 0:
                            profile.target_league_id = val
                    except (ValueError, TypeError):
                        pass

                session.add(profile)
                session.commit()
                session.refresh(profile)

            # If existing profile has None for IDs, also offer safe initial seeding from env without overwriting existing IDs
            from fpl_oracle.config import FPL_MANAGER_ID, FPL_TARGET_LEAGUE_ID

            changed = False
            if profile.manager_id is None and FPL_MANAGER_ID:
                try:
                    val = int(str(FPL_MANAGER_ID).strip())
                    if val > 0:
                        profile.manager_id = val
                        changed = True
                except (ValueError, TypeError):
                    pass
            if profile.target_league_id is None and FPL_TARGET_LEAGUE_ID:
                try:
                    val = int(str(FPL_TARGET_LEAGUE_ID).strip())
                    if val > 0:
                        profile.target_league_id = val
                        changed = True
                except (ValueError, TypeError):
                    pass
            if changed:
                profile.updated_at = datetime.now(UTC)
                session.commit()
                session.refresh(profile)
                self._sync_profile_json(profile)

            if not PROFILE_JSON_PATH.exists():
                self._sync_profile_json(profile)

            return ProfileData(
                id=int(profile.id) if profile.id is not None else 1,
                manager_id=int(profile.manager_id) if profile.manager_id is not None else None,
                target_league_id=int(profile.target_league_id) if profile.target_league_id is not None else None,
                risk_preference=str(profile.risk_preference or "balanced"),
                llm_provider=str(profile.llm_provider or "gemini"),
                bank_override_enabled=bool(profile.bank_override_enabled)
                if profile.bank_override_enabled is not None
                else False,
                ft_override_enabled=bool(profile.ft_override_enabled)
                if profile.ft_override_enabled is not None
                else False,
                bank=float(profile.bank if profile.bank is not None else 0.0),
                free_transfers=int(profile.free_transfers if profile.free_transfers is not None else 1),
                manual_squad=str(profile.manual_squad) if profile.manual_squad else None,
                keep_ids=sorted(json.loads(profile.keep_ids)) if profile.keep_ids else [],
                avoid_ids=sorted(json.loads(profile.avoid_ids)) if profile.avoid_ids else [],
                updated_at=profile.updated_at if isinstance(profile.updated_at, datetime) else None,
            )

    @retry_on_db_lock()
    def update_profile(self, **kwargs):
        with self.transaction() as session:
            profile = session.query(UserProfile).filter(UserProfile.id == 1).first()
            if not profile:
                profile = UserProfile(id=1, bank_override_enabled=0, ft_override_enabled=0)
                session.add(profile)
            for k, v in kwargs.items():
                if k in ("keep_ids", "avoid_ids"):
                    setattr(profile, k, json.dumps(sorted({int(i) for i in v})) if v else None)
                elif k == "manual_squad":
                    if isinstance(v, list) and len(v) > 0:
                        profile.manual_squad = json.dumps(v)
                    elif v is None or v == [] or v == "":
                        profile.manual_squad = None
                elif hasattr(profile, k):
                    setattr(profile, k, v)
            if "bank" in kwargs:
                profile.bank_override_enabled = int(kwargs["bank"] is not None)
                if kwargs["bank"] is None:
                    profile.bank = 0.0
            if "free_transfers" in kwargs:
                profile.ft_override_enabled = int(kwargs["free_transfers"] is not None)
                if kwargs["free_transfers"] is None:
                    profile.free_transfers = 1
            profile.updated_at = datetime.now(UTC)
            session.flush()
            session.refresh(profile)
            self._sync_profile_json(profile)

    def get_chat_history(self, session_id: str = "default", limit: int = 50) -> list[dict[str, str]]:
        with self.get_session() as session:
            msgs = (
                session.query(ChatMessage)
                .filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
                .limit(limit)
                .all()
            )
            return [{"role": m.role, "content": m.content} for m in reversed(msgs)]

    @retry_on_db_lock()
    def add_chat_message(self, role: str, content: str, session_id: str = "default"):
        with self.transaction() as session:
            msg = ChatMessage(session_id=session_id, role=role, content=content, created_at=datetime.now(UTC))
            session.add(msg)

    @retry_on_db_lock()
    def save_price_snapshots(self, snapshots: list[dict[str, Any]]):
        """Save a batch of player price predictions / urgency records."""
        with self.transaction() as session:
            now = datetime.now(UTC)
            for s in snapshots:
                rec = PriceSnapshotRecord(
                    element_id=int(s["element"]),
                    web_name=str(s["web_name"]),
                    now_cost=float(s["now_cost"]),
                    net_transfers=int(s.get("net_transfers_event", 0)),
                    selected_by_percent=float(s.get("selected_by_percent", 0.0)),
                    hourly_rate=float(s.get("hourly_rate", 0.0)),
                    urgency_score=float(s.get("urgency_score", 0.0)),
                    direction=str(s.get("direction", "STABLE")),
                    snapshot_time=now,
                )
                session.add(rec)

    def get_latest_price_snapshots(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.get_session() as session:
            rows = (
                session.query(PriceSnapshotRecord).order_by(PriceSnapshotRecord.snapshot_time.desc()).limit(limit).all()
            )
            return [
                {
                    "element_id": r.element_id,
                    "web_name": r.web_name,
                    "now_cost": r.now_cost,
                    "net_transfers": r.net_transfers,
                    "selected_by_percent": r.selected_by_percent,
                    "hourly_rate": r.hourly_rate,
                    "urgency_score": r.urgency_score,
                    "direction": r.direction,
                    "snapshot_time": r.snapshot_time.isoformat() if r.snapshot_time else None,
                }
                for r in rows
            ]

    @retry_on_db_lock()
    def record_job_start(self, job_id: str, job_name: str) -> int:
        with self.transaction() as session:
            run = JobRunRecord(job_id=job_id, job_name=job_name, status="RUNNING", started_at=datetime.now(UTC))
            session.add(run)
            session.flush()
            return int(run.id)

    @retry_on_db_lock()
    def record_job_finish(self, run_id: int, status: str, duration_seconds: float, details: str = ""):
        with self.transaction() as session:
            run = session.query(JobRunRecord).filter(JobRunRecord.id == run_id).first()
            if run:
                run.status = status  # type: ignore[assignment]
                run.completed_at = datetime.now(UTC)  # type: ignore[assignment]
                run.duration_seconds = duration_seconds  # type: ignore[assignment]
                run.details = details  # type: ignore[assignment]

    def get_job_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.get_session() as session:
            runs = session.query(JobRunRecord).order_by(JobRunRecord.started_at.desc()).limit(limit).all()
            return [
                {
                    "id": r.id,
                    "job_id": r.job_id,
                    "job_name": r.job_name,
                    "status": r.status,
                    "started_at": r.started_at.isoformat() if r.started_at else None,
                    "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                    "duration_seconds": round(float(r.duration_seconds or 0.0), 2),
                    "details": r.details,
                }
                for r in runs
            ]

    @retry_on_db_lock()
    def save_model_version(
        self,
        version: str,
        ml_mae: float,
        ml_spearman: float,
        base_mae: float,
        status: str = "production",
        is_active: bool = True,
        notes: str = "",
    ):
        with self.transaction() as session:
            if is_active:
                session.query(ModelVersionRecord).update({"is_active": 0})
            existing = session.query(ModelVersionRecord).filter(ModelVersionRecord.version == version).first()
            if existing:
                existing.ml_mae = ml_mae  # type: ignore[assignment]
                existing.ml_spearman = ml_spearman  # type: ignore[assignment]
                existing.base_mae = base_mae  # type: ignore[assignment]
                existing.is_active = 1 if is_active else 0  # type: ignore[assignment]
                existing.status = status  # type: ignore[assignment]
                existing.notes = notes  # type: ignore[assignment]
                existing.created_at = datetime.now(UTC)  # type: ignore[assignment]
            else:
                rec = ModelVersionRecord(
                    version=version,
                    created_at=datetime.now(UTC),
                    ml_mae=ml_mae,
                    ml_spearman=ml_spearman,
                    base_mae=base_mae,
                    is_active=1 if is_active else 0,
                    status=status,
                    notes=notes,
                )
                session.add(rec)

    def get_model_versions(self) -> list[dict[str, Any]]:
        with self.get_session() as session:
            rows = session.query(ModelVersionRecord).order_by(ModelVersionRecord.created_at.desc()).all()
            return [
                {
                    "version": r.version,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "ml_mae": r.ml_mae,
                    "ml_spearman": r.ml_spearman,
                    "base_mae": r.base_mae,
                    "is_active": bool(r.is_active),
                    "status": r.status,
                    "notes": r.notes,
                }
                for r in rows
            ]

    def get_active_model_version(self) -> dict[str, Any] | None:
        with self.get_session() as session:
            r = session.query(ModelVersionRecord).filter(ModelVersionRecord.is_active == 1).first()
            if r:
                return {
                    "version": r.version,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "ml_mae": r.ml_mae,
                    "ml_spearman": r.ml_spearman,
                    "base_mae": r.base_mae,
                    "status": r.status,
                    "notes": r.notes,
                }
            return None


data_store = DataStore()
