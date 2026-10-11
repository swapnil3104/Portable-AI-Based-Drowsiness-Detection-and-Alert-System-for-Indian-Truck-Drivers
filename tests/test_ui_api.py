"""Unit and integration tests for Phase 7 Software UI endpoints and telemetry.

Verifies:
- Project branding: DrishtiDrive AI and 'A Watchful Eye on Every Journey'.
- Camera lifecycle (start, stop, device enumeration).
- Metric reporting: eye state, confidence, closure duration, FPS, latency.
- Three-stage alert panel and manual preview triggers.
- Event log endpoints (retrieval, formatting, clearing).
- Real-time WebSocket telemetry stream responsiveness.
"""
from __future__ import annotations

import json
import time
from fastapi.testclient import TestClient
import pytest

from backend.app import app, state
from backend.audio_backend import MockAudioBackend


@pytest.fixture
def client() -> TestClient:
    """Create FastAPI test client fixture with mock audio for fast tests."""
    mock_audio = MockAudioBackend()
    state.pipeline.alert_manager.audio_backend = mock_audio
    return TestClient(app)


def test_health_and_branding(client: TestClient) -> None:
    """Verify Task T7.1: Branding and tagline in health API."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["project"] == "DrishtiDrive AI"
    assert data["tagline"] == "A Watchful Eye on Every Journey"
    assert "model" in data


def test_camera_device_enumeration(client: TestClient) -> None:
    """Verify Task T7.3: Camera source enumeration."""
    res = client.get("/api/camera/devices")
    assert res.status_code == 200
    devices = res.json()
    assert isinstance(devices, list)
    assert len(devices) >= 1
    assert "name" in devices[0]


def test_camera_lifecycle_endpoints(client: TestClient) -> None:
    """Verify Task T7.3: Camera start and stop controls."""
    # Start camera
    res_start = client.post("/api/camera/start", json={"camera_index": 0, "resolution": "1280x720"})
    assert res_start.status_code == 200
    assert "status" in res_start.json()

    # Verify active state
    health = client.get("/api/health").json()
    assert health["camera_active"] is True

    # Stop camera
    res_stop = client.post("/api/camera/stop")
    assert res_stop.status_code == 200
    assert res_stop.json()["status"] == "stopped"

    # Verify stopped state
    health_after = client.get("/api/health").json()
    assert health_after["camera_active"] is False


def test_settings_retrieval_and_update(client: TestClient) -> None:
    """Verify settings endpoint allows updating alert parameters."""
    res = client.get("/api/settings")
    assert res.status_code == 200
    cfg = res.json()
    assert "alerts" in cfg

    # Update volume
    res_up = client.put("/api/settings", json={"volume": 0.45})
    assert res_up.status_code == 200
    assert res_up.json()["status"] == "updated"
    assert state.pipeline.alert_manager.volume == pytest.approx(0.45, abs=0.01)


def test_event_log_lifecycle(client: TestClient) -> None:
    """Verify Task T7.7: Event log retrieval and clear."""
    # Get logs
    res_logs = client.get("/api/logs?limit=10")
    assert res_logs.status_code == 200
    logs = res_logs.json()
    assert isinstance(logs, list)
    assert len(logs) >= 1

    entry = logs[0]
    assert "timestamp" in entry
    assert "description" in entry
    assert "stage" in entry

    # Clear logs
    res_clear = client.delete("/api/logs")
    assert res_clear.status_code == 200
    assert res_clear.json()["status"] == "cleared"

    res_empty = client.get("/api/logs")
    assert len(res_empty.json()) == 0


def test_alert_preview_triggers(client: TestClient) -> None:
    """Verify Task T7.6: Manual alert test triggers and event logging."""
    for stg in [1, 2, 3]:
        res = client.post("/api/alerts/preview", json={"stage": stg})
        assert res.status_code == 200
        assert res.json()["stage"] == stg

    # Ensure events were logged
    logs = client.get("/api/logs").json()
    stages_logged = [e.get("stage") for e in logs if e.get("event_type") == "MANUAL_ALERT_TEST"]
    assert 1 in stages_logged
    assert 2 in stages_logged
    assert 3 in stages_logged


def test_websocket_telemetry_stream(client: TestClient) -> None:
    """Verify WebSocket /ws/stream delivers real-time telemetry packets."""
    with client.websocket_connect("/ws/stream") as ws:
        packet = ws.receive_json()
        assert "active" in packet
        assert "status" in packet
        assert "telemetry" in packet

        tel = packet["telemetry"]
        # Required Phase 7 metrics
        assert "eye_state" in tel
        assert "confidence" in tel
        assert "closure_duration_sec" in tel
        assert "fps" in tel
        assert "latency_ms" in tel
        assert "stage" in tel
        assert "status_text" in tel
        assert "voice_message" in tel
        assert "audio_status" in tel
