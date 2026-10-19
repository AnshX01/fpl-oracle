import asyncio

from fastapi.testclient import TestClient

from fpl_oracle.server import main


def test_files_outside_the_web_folder_are_never_served(tmp_path, monkeypatch):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("page")
    (web / "app.js").write_text("script")
    (tmp_path / ".env").write_text("SECRET=1")
    monkeypatch.setattr(main, "WEB_DIR", web)

    for path in ("../.env", "../../.env", str(tmp_path / ".env")):
        served = asyncio.run(main.serve_spa_or_static(path))
        assert served.path == web / "index.html"
    assert asyncio.run(main.serve_spa_or_static("app.js")).path == (web / "app.js").resolve()


def test_a_foreign_website_gets_no_cors_permission():
    client = TestClient(main.app)
    foreign = client.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in foreign.headers
    local = client.get("/api/health", headers={"Origin": "http://127.0.0.1:8000"})
    assert local.headers["access-control-allow-origin"] == "http://127.0.0.1:8000"
