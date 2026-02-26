import time

from fastapi.testclient import TestClient

from backend.main import app
from backend.routes import chat as chat_route
from backend.services.session import store


client = TestClient(app)


def _make_session():
    session = store.create({"username": "tester", "role": "Admin", "name": "Tester"})
    session.connected = True
    session.db = object()
    session.llm = object()
    return session


def test_nl_poll_returns_ready_state():
    session = _make_session()
    request_id = "req-ready-1"
    chat_route._set_nl_state(
        session_token=session.token,
        request_id=request_id,
        status="ready",
        question="q",
        results=[{"v": 1}],
        nl_answer="summary ready",
        error=None,
    )

    resp = client.get(
        f"/api/chat/nl-response?request_id={request_id}",
        headers={"Authorization": f"Bearer {session.token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["nl_answer"] == "summary ready"
    store.delete(session.token)


def test_nl_stream_ready_emits_done_event():
    session = _make_session()
    request_id = "req-stream-1"
    chat_route._set_nl_state(
        session_token=session.token,
        request_id=request_id,
        status="ready",
        question="q",
        results=[{"v": 1}],
        nl_answer="streamable summary",
        error=None,
    )

    with client.stream(
        "GET",
        f"/api/chat/nl-stream?request_id={request_id}",
        headers={"Authorization": f"Bearer {session.token}"},
    ) as resp:
        assert resp.status_code == 200
        payload = "".join(resp.iter_text())
    assert "event: token" in payload
    assert "event: done" in payload
    store.delete(session.token)


def test_query_starts_server_side_nl_generation(monkeypatch):
    session = _make_session()

    def _fake_handle_query(**kwargs):
        return {
            "intent": "DATA_QUERY",
            "nl_answer": None,
            "sql": "SELECT 1 AS revenue",
            "results": [{"revenue": 1}],
            "row_count": 1,
            "from_cache": False,
            "error": None,
            "updated_state": {},
            "nl_pending": True,
            "fallback_used": False,
            "timing": {"total": 10.0},
        }

    monkeypatch.setattr(chat_route, "legacy_handle_query", _fake_handle_query)
    monkeypatch.setattr(chat_route, "generate_nl_response", lambda q, r, llm: "background summary")

    resp = client.post(
        "/api/chat/query",
        headers={"Authorization": f"Bearer {session.token}"},
        json={"question": "show revenue"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["nl_pending"] is True
    request_id = body["request_id"]
    assert request_id

    status = None
    for _ in range(20):
        poll = client.get(
            f"/api/chat/nl-response?request_id={request_id}",
            headers={"Authorization": f"Bearer {session.token}"},
        )
        assert poll.status_code == 200
        status = poll.json()
        if status["status"] == "ready":
            break
        time.sleep(0.05)

    assert status is not None
    assert status["status"] == "ready"
    assert status["nl_answer"] == "background summary"
    store.delete(session.token)


def test_query_timing_includes_all_stage_keys(monkeypatch):
    session = _make_session()
    monkeypatch.setattr(chat_route, "detect_intent_simple", lambda _q: "DATA_QUERY")

    def _fake_handle_query(**kwargs):
        return {
            "intent": "DATA_QUERY",
            "nl_answer": None,
            "sql": "SELECT 1 AS v",
            "results": [{"v": 1}],
            "row_count": 1,
            "from_cache": False,
            "error": None,
            "updated_state": {},
            "nl_pending": False,
            "fallback_used": False,
            "timing": {"total": 12.3},
        }

    monkeypatch.setattr(chat_route, "legacy_handle_query", _fake_handle_query)

    resp = client.post(
        "/api/chat/query",
        headers={"Authorization": f"Bearer {session.token}"},
        json={"question": "show revenue"},
    )
    assert resp.status_code == 200
    timing = resp.json().get("timing") or {}
    expected_stages = [
        "start",
        "intent_detection",
        "schema_loading",
        "stored_procedure_guidance",
        "cache_lookup",
        "rag_retrieval",
        "sql_generation",
        "sql_validation",
        "guardrails_applied",
        "db_execution",
        "results_formatting",
        "total",
    ]
    for stage in expected_stages:
        assert stage in timing
    store.delete(session.token)
