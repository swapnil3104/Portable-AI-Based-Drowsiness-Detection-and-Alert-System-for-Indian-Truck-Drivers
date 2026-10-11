import { useState, useEffect, useRef, useCallback } from 'react';

export function useCameraStream() {
  const [streamData, setStreamData] = useState({
    active: false,
    status: 'READY',
    frame: null,
    telemetry: {
      ear: 0.32,
      mar: 0.15,
      yaw: 2,
      pitch: -1,
      bpm: 18,
      attention_score: 95,
      status_text: 'Normal',
      stage: 1,
      fps: 30,
    },
    boxes: [],
  });

  const [devices, setDevices] = useState([]);
  const [selectedDevice, setSelectedDevice] = useState(0);
  const [sensitivity, setSensitivity] = useState(1.0);
  const [events, setEvents] = useState([]);
  const wsRef = useRef(null);

  // Fetch initial devices, settings, logs
  const fetchData = useCallback(async () => {
    try {
      const devRes = await fetch('/api/camera/devices');
      if (devRes.ok) setDevices(await devRes.json());

      const setRes = await fetch('/api/settings');
      if (setRes.ok) {
        const s = await setRes.json();
        if (s.detection?.sensitivity) setSensitivity(s.detection.sensitivity);
        if (s.camera?.index !== undefined) setSelectedDevice(s.camera.index);
      }

      const logRes = await fetch('/api/logs?limit=50');
      if (logRes.ok) setEvents(await logRes.json());
    } catch (e) {
      // Offline fallback
    }
  }, []);

  useEffect(() => {
    fetchData();
    // Auto-start continuous monitoring on default webcam (Device 0)
    fetch('/api/camera/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera_index: 0 }),
    }).catch(() => {});
  }, [fetchData]);

  // WebSocket connect
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/stream`;

    let reconnectTimer;

    let lastAlertStage = 0;

    const connectWs = () => {
      try {
        const socket = new WebSocket(wsUrl);
        wsRef.current = socket;

        socket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            setStreamData(data);

            const stage = data.telemetry?.stage || 0;
            if (stage > 0 && stage !== lastAlertStage) {
              lastAlertStage = stage;
              fetchData();
            } else if (stage === 0) {
              lastAlertStage = 0;
            }
          } catch (err) {}
        };

        socket.onclose = () => {
          reconnectTimer = setTimeout(connectWs, 2000);
        };
      } catch (err) {
        reconnectTimer = setTimeout(connectWs, 3000);
      }
    };

    connectWs();

    return () => {
      clearTimeout(reconnectTimer);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const startCamera = async (cameraIndex = selectedDevice) => {
    try {
      await fetch('/api/camera/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera_index: cameraIndex }),
      });
      fetchData();
    } catch (e) {}
  };

  const stopCamera = async () => {
    try {
      await fetch('/api/camera/stop', { method: 'POST' });
      fetchData();
    } catch (e) {}
  };

  const updateSensitivity = async (val) => {
    setSensitivity(val);
    try {
      await fetch('/api/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ detection: { sensitivity: val } }),
      });
    } catch (e) {}
  };

  const previewAlert = async (stage) => {
    try {
      await fetch('/api/alerts/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stage }),
      });
      fetchData();
    } catch (e) {}
  };

  const clearLog = async () => {
    try {
      await fetch('/api/logs', { method: 'DELETE' });
      setEvents([]);
    } catch (e) {}
  };

  return {
    streamData,
    devices,
    selectedDevice,
    setSelectedDevice,
    sensitivity,
    updateSensitivity,
    events,
    startCamera,
    stopCamera,
    previewAlert,
    clearLog,
  };
}
