import asyncio
import io
import logging
from types import SimpleNamespace

from fpl_oracle.server.advice_job import AdvicePublisher, profile_key
from fpl_oracle.utils.logging import SecretRedactingFilter


def test_failure_trace_is_logged_with_type_but_secret_is_scrubbed(monkeypatch):
    from fpl_oracle.data.store import data_store
    from fpl_oracle.server.routes import api

    secret = "test-private-key-never-log"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    monkeypatch.setattr(data_store, "get_profile", lambda: SimpleNamespace(manager_id=1))

    async def broken():
        raise RuntimeError("https://example.invalid/?key=" + secret)

    monkeypatch.setattr(api, "get_squad", broken)
    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.addFilter(SecretRedactingFilter())
    logger = logging.getLogger("fpl_oracle.advice")
    logger.addHandler(handler)
    try:
        async def run():
            publisher = AdvicePublisher()
            publisher.start(profile_key(data_store.get_profile()))
            await publisher.task
            assert publisher.public()["status"] == "failed"
            assert publisher.public()["error_type"] == "RuntimeError"
            assert secret not in str(publisher.public())
        asyncio.run(run())
    finally:
        logger.removeHandler(handler)
    text = output.getvalue()
    assert "type=RuntimeError" in text
    assert "Traceback" in text
    assert secret not in text
    assert "[REDACTED]" in text
