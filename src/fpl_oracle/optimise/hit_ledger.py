"""Persistent recommendations and official taken-hit outcomes, kept separate."""

import hashlib
import json
from typing import cast

from sqlalchemy import Column, String, Table, Text

from fpl_oracle.api.fpl_client import fpl_client
from fpl_oracle.data.store import Base, data_store, engine
from fpl_oracle.optimise.hit_policy import cooldown_until, settle_hit


class HitRecord(Base):
    __tablename__ = "hit_outcomes"
    key = Column(String(100), primary_key=True)
    payload = Column(Text, nullable=False)


cast(Table, HitRecord.__table__).create(bind=engine, checkfirst=True)


class HitLedger:
    def __init__(self):
        self._refreshes = {}

    def read(self, manager_id):
        with data_store.get_session() as session:
            rows = session.query(HitRecord).filter(HitRecord.key.like(f"{manager_id}:%")).all()
            return [json.loads(row.payload) for row in rows]

    def write(self, manager_id, record):
        identity = (
            [record["kind"], record["gameweek"]]
            if record["kind"] == "taken"
            else [record["kind"], record["gameweek"], sorted(record["ins"]), sorted(record["outs"])]
        )
        key = f"{manager_id}:" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        with data_store.get_session() as session:
            if record["kind"] == "taken":
                for old in session.query(HitRecord).filter(HitRecord.key.like(f"{manager_id}:%")).all():
                    prior = json.loads(old.payload)
                    if prior.get("kind") == "taken" and prior["gameweek"] == record["gameweek"] and old.key != key:
                        session.delete(old)
            row = session.get(HitRecord, key)
            if row is None:
                session.add(HitRecord(key=key, payload=json.dumps(record)))
            else:
                row.payload = json.dumps(record)
            session.commit()

    def undo_user_confirmation(self, manager_id, gameweek, previous):
        with data_store.get_session() as session:
            for row in session.query(HitRecord).filter(HitRecord.key.like(f"{manager_id}:%")).all():
                rec = json.loads(row.payload)
                if rec.get("kind") == "taken" and rec["gameweek"] == gameweek and rec.get("source") == "user_confirmed":
                    session.delete(row)
            session.commit()
        if previous and previous.get("hit_cost"):
            self.write(
                manager_id,
                dict(
                    kind="taken",
                    gameweek=gameweek,
                    ins=previous["actual_ins"],
                    outs=previous["actual_outs"],
                    hit_cost=previous["hit_cost"],
                    source="user_confirmed",
                    status="pending",
                    expected_gain=None,
                    matched_recommendation=False,
                ),
            )
        self._refreshes.clear()

    async def refresh(self, manager_id):
        if not manager_id:
            records = self.read("manual")
            bootstrap, stale = await fpl_client.get_bootstrap_static()
            for record in records:
                event = next((e for e in bootstrap.events if e.id == record["gameweek"]), None)
                if record.get("kind") != "taken" or not event or not event.finished or not event.data_checked or stale:
                    continue
                live, live_stale = await fpl_client.get_live_gameweek(record["gameweek"])
                if not live_stale:
                    points = {
                        int(e["id"]): float(e["stats"]["total_points"])
                        for e in live.get("elements", [])
                        if "total_points" in e.get("stats", {})
                    }
                    record.update(settle_hit(record["ins"], record["outs"], points, record["hit_cost"]))
                    self.write("manual", record)
            records = self.read("manual")
            return {"status": "manual_confirmed", "records": records, "cooldown_until": cooldown_until(records)}
        import time

        cached = self._refreshes.get(manager_id)
        if cached and time.monotonic() - cached[0] < 300:
            return cached[1]
        history, hs = await fpl_client.get_manager_history(manager_id)
        transfers, ts = await fpl_client.get_manager_transfers(manager_id)
        bootstrap, bs = await fpl_client.get_bootstrap_static()
        existing = self.read(manager_id)
        if hs or ts or bs:
            return {"status": "stale", "records": existing, "cooldown_until": cooldown_until(existing)}
        events = {e.id: e for e in bootstrap.events}
        for week in history.current:
            if not week.event_transfers_cost:
                continue
            moves = [t for t in transfers if t.event == week.event]
            from collections import Counter

            bought = Counter(t.element_in for t in moves)
            sold = Counter(t.element_out for t in moves)
            ins, outs = list((bought - sold).elements()), list((sold - bought).elements())
            record = dict(
                kind="taken",
                gameweek=week.event,
                ins=ins,
                outs=outs,
                hit_cost=week.event_transfers_cost,
                status="pending",
                source="official_verified",
                previously_user_confirmed=any(
                    r.get("kind") == "taken" and r["gameweek"] == week.event and r.get("source") == "user_confirmed"
                    for r in existing
                ),
            )
            event = events.get(week.event)
            if not moves:
                record["status"] = "unknown"
            elif event and event.finished and event.data_checked:
                live, stale = await fpl_client.get_live_gameweek(week.event)
                if not stale:
                    points = {
                        int(e["id"]): float(e["stats"]["total_points"])
                        for e in live.get("elements", [])
                        if "total_points" in e.get("stats", {})
                    }
                    record.update(settle_hit(ins, outs, points, week.event_transfers_cost))
            matched = next(
                (
                    r
                    for r in existing
                    if r.get("kind") == "recommended"
                    and r["gameweek"] == week.event
                    and sorted(r["ins"]) == sorted(ins)
                    and sorted(r["outs"]) == sorted(outs)
                ),
                None,
            )
            record["matched_recommendation"] = bool(matched)
            record["expected_gain"] = matched.get("expected_gain") if matched else None
            self.write(manager_id, record)
        records = self.read(manager_id)
        result = {"status": "fresh", "records": records, "cooldown_until": cooldown_until(records)}
        self._refreshes[manager_id] = (time.monotonic(), result)
        return result

    def recommend(self, manager_id, plan, target_gw):
        if not manager_id or not plan.get("hit_cost"):
            return
        first = plan["trajectory"][0]
        self.write(
            manager_id,
            dict(
                kind="recommended",
                gameweek=target_gw,
                ins=first["transfers_in"],
                outs=first["transfers_out"],
                hit_cost=first["hit_cost"],
                expected_gain=first["expected_player_gain"],
                status="not_verified_taken",
            ),
        )


hit_ledger = HitLedger()
