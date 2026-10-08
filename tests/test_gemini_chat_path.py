import pytest

from fpl_oracle.llm.gemini import GeminiProvider


@pytest.mark.anyio
async def test_gemini_multiple_calls_and_grounded_followup(monkeypatch):
    payloads = []
    responses = [
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"functionCall": {"name": "get_my_team", "args": {}}},
                            {"functionCall": {"name": "captain_options", "args": {}}},
                        ]
                    }
                }
            ]
        },
        {"candidates": [{"content": {"parts": [{"text": "Saka: 8.2 predicted points."}]}}]},
    ]

    class Response:
        status_code = 200

        def json(self):
            return responses.pop(0)

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, json, headers):
            assert "key=" not in url
            payloads.append(json)
            return Response()

    async def execute(name, args):
        return {"name": "Saka", "expected_points": 8.2}

    monkeypatch.setattr("fpl_oracle.llm.gemini.httpx.AsyncClient", Client)
    monkeypatch.setattr("fpl_oracle.llm.gemini.tool_executor.execute", execute)
    text = await GeminiProvider("test-not-a-real-key").chat(
        [{"role": "user", "content": "captain?"}], "Use fetched data"
    )
    assert "8.2" in text
    assert payloads[0]["tool_config"]["function_calling_config"]["mode"] == "ANY"
    assert payloads[1]["contents"][-1]["role"] == "user"
    assert len(payloads[1]["contents"][-1]["parts"]) == 2


@pytest.mark.anyio
async def test_gemini_empty_answer_fails_instead_of_invented_success(monkeypatch):
    class Response:
        status_code = 200

        def json(self):
            return {"candidates": []}

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr("fpl_oracle.llm.gemini.httpx.AsyncClient", Client)
    with pytest.raises(RuntimeError, match="grounded"):
        await GeminiProvider("test").chat([{"role": "user", "content": "captain?"}], "data")
