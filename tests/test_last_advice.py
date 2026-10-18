from datetime import UTC, datetime, timedelta

from fpl_oracle.server import advice_job


def card(deadline):
    return {"decisionCard": {"deadline": deadline.isoformat()}}


def test_last_advice_is_reused_for_the_same_manager_until_the_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(advice_job, "_last_path", lambda: tmp_path / "last_advice.json")
    future = datetime.now(UTC) + timedelta(days=2)
    advice_job.save_last_advice({"manager": 1}, card(future))
    assert advice_job.load_last_advice({"manager": 1})["result"] == card(future)
    assert advice_job.load_last_advice({"manager": 2}) is None


def test_last_advice_expires_at_the_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(advice_job, "_last_path", lambda: tmp_path / "last_advice.json")
    advice_job.save_last_advice({"manager": 1}, card(datetime.now(UTC) - timedelta(minutes=1)))
    assert advice_job.load_last_advice({"manager": 1}) is None
