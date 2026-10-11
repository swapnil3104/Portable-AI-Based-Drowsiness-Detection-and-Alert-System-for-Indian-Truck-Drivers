import React, { useState, useEffect } from 'react';
import { Video, Square, Play, ShieldAlert, Eye, EyeOff } from 'lucide-react';
import './LiveMonitoring.css';

export default function LiveMonitoring({
  streamData,
  onStartCamera,
  onStopCamera,
  showLandmarks = true,
}) {
  const [elapsed, setElapsed] = useState(0);

  const isRunning = streamData?.active ?? false;
  const telemetry = streamData?.telemetry ?? {
    ear: 0.35,
    mar: 0.14,
    yaw: 0,
    pitch: 0,
    attention_score: 95,
    fps: 30,
    status_text: 'Normal',
  };

  const isEyesClosed = telemetry.eye_state === 'CLOSED' || (typeof telemetry.closure_duration_sec === 'number' && telemetry.closure_duration_sec > 0.15);
  const closureSec = typeof telemetry.closure_duration_sec === 'number' ? telemetry.closure_duration_sec : 0.0;

  useEffect(() => {
    let timer;
    if (isRunning) {
      timer = setInterval(() => {
        setElapsed((prev) => prev + 1);
      }, 1000);
    } else {
      setElapsed(0);
    }
    return () => clearInterval(timer);
  }, [isRunning]);

  const formatTime = (seconds) => {
    const hrs = String(Math.floor(seconds / 3600)).padStart(2, '0');
    const mins = String(Math.floor((seconds % 3600) / 60)).padStart(2, '0');
    const secs = String(seconds % 60).padStart(2, '0');
    return `${hrs}:${mins}:${secs}`;
  };

  return (
    <div className="card live-monitoring-card">
      <div className="card-header">
        <div className="card-title-group">
          <div className="camera-icon-pill">
            <Video className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="card-title">Live Driver Monitoring</h2>
            <p className="card-subtitle">Real-time analysis of driver's face, eyes and head movements</p>
          </div>
        </div>

        <div className="monitoring-actions">
          <span className={`badge-pill ${isRunning ? 'success' : 'warning'}`}>
            <span className="pulse-dot"></span>
            {isRunning ? 'Camera Active' : 'Camera Ready'}
          </span>

          {isRunning ? (
            <button className="btn-stop-camera" onClick={onStopCamera} id="stop-camera-btn">
              <Square size={13} fill="currentColor" />
              <span>Stop Camera</span>
            </button>
          ) : (
            <button className="btn-start-camera" onClick={onStartCamera} id="start-camera-btn">
              <Play size={13} fill="currentColor" />
              <span>Start Camera</span>
            </button>
          )}
        </div>
      </div>

      {/* Main Video Viewport */}
      <div className={`video-viewport ${isEyesClosed ? 'eyes-closed-warning' : ''}`} id="video-viewport">
        {/* Top-left REC timer */}
        <div className="rec-indicator">
          <span className="rec-dot"></span>
          <span>REC {formatTime(elapsed)}</span>
        </div>

        {/* Real-time Eye Closing Status Indicator Badge */}
        {isEyesClosed ? (
          <div className="eye-closed-indicator-badge" id="eye-closed-indicator">
            <EyeOff size={16} className="eye-closed-pulse-icon" />
            <span className="eye-closed-title">EYES CLOSED</span>
            <span className="eye-closed-timer">{closureSec.toFixed(2)}s</span>
          </div>
        ) : (
          <div className="eye-open-indicator-badge" id="eye-open-indicator">
            <Eye size={14} className="eye-open-icon" />
            <span>EYES OPEN</span>
          </div>
        )}

        {/* Real-time Visual Alert Banner */}
        {telemetry.stage > 0 && (
          <div className={`viewport-alert-banner stage-${telemetry.stage}`}>
            <ShieldAlert className="alert-banner-icon" size={24} />
            <div className="alert-banner-text">
              <span className="banner-title">
                {telemetry.stage === 3
                  ? 'CRITICAL ALERT — PULL OVER SAFELY'
                  : telemetry.stage === 2
                  ? 'DROWSINESS WARNING — REST RECOMMENDED'
                  : 'EARLY WARNING — STAY ATTENTIVE'}
              </span>
              <span className="banner-desc">
                {telemetry.voice_message || (telemetry.stage === 3 ? 'Immediate Rest Required' : telemetry.stage === 2 ? 'Eye Closure / Yawn Detected' : 'Eyes Closed')}
              </span>
            </div>
          </div>
        )}

        {/* Top-right & corners framing brackets */}
        <div className="viewport-brackets">
          <div className="corner corner-tl"></div>
          <div className="corner corner-tr"></div>
          <div className="corner corner-bl"></div>
          <div className="corner corner-br"></div>
        </div>

        {/* Video feed or synthesized stream */}
        {streamData?.frame ? (
          <img src={streamData.frame} alt="Live Driver Feed" className="video-feed-image" />
        ) : (
          <div className="synthetic-driver-placeholder">
            <div className="simulated-driver-silhouette">
              {/* SVG Driver In Vehicle */}
              <svg viewBox="0 0 640 360" className="driver-svg">
                <rect width="640" height="360" fill="#1b2434" />
                {/* Windshield view & road horizon */}
                <path d="M0 160 L640 160 L640 360 L0 360 Z" fill="#243044" />
                <path d="M300 160 L340 160 L480 360 L160 360 Z" fill="#2d3b52" opacity="0.6" />
                {/* Car roof and pillars */}
                <path d="M0 0 L140 150 L0 240 Z" fill="#141a24" />
                <path d="M640 0 L500 150 L640 240 Z" fill="#141a24" />
                {/* Driver head & shoulders */}
                <ellipse cx="320" cy="180" rx="55" ry="68" fill="#d4a373" />
                <ellipse cx="320" cy="300" rx="110" ry="70" fill="#1f2937" />
                {/* Eyes */}
                <circle cx="298" cy="170" r="7" fill="#ffffff" />
                <circle cx="298" cy="170" r="3.5" fill="#1e293b" />
                <circle cx="342" cy="170" r="7" fill="#ffffff" />
                <circle cx="342" cy="170" r="3.5" fill="#1e293b" />
                {/* Steering wheel */}
                <path d="M220 340 A 100 100 0 0 1 420 340" stroke="#0f172a" strokeWidth="16" fill="none" />
              </svg>
            </div>
          </div>
        )}

        {/* Facial Landmarks Overlay for simulation mode */}
        {!streamData?.frame && showLandmarks && (
          <svg className="landmarks-overlay" viewBox="0 0 640 360" preserveAspectRatio="none">
            {/* Face Bounding Box with Corner Brackets */}
            <g stroke="#10b981" strokeWidth="3" fill="none">
              <path d="M 230 110 L 255 110" />
              <path d="M 230 110 L 230 135" />
              <path d="M 410 110 L 385 110" />
              <path d="M 410 110 L 410 135" />
              <path d="M 230 250 L 255 250" />
              <path d="M 230 250 L 230 225" />
              <path d="M 410 250 L 385 250" />
              <path d="M 410 250 L 410 225" />
            </g>

            {/* Mesh Lines */}
            <g stroke="#10b981" strokeWidth="1" strokeOpacity="0.5" fill="none">
              <polygon points="320,130 280,165 320,175 360,165" />
              <polygon points="280,165 298,170 320,185 342,170 360,165" />
              <polygon points="298,170 320,205 342,170" />
              <polygon points="290,215 320,205 350,215 320,230" />
              <line x1="280" y1="165" x2="270" y2="210" />
              <line x1="360" y1="165" x2="370" y2="210" />
              <line x1="270" y1="210" x2="320" y2="245" />
              <line x1="370" y1="210" x2="320" y2="245" />
            </g>

            {/* Mesh Points */}
            <g fill="#10b981">
              <circle cx="320" cy="130" r="2.5" />
              <circle cx="280" cy="165" r="2.5" />
              <circle cx="360" cy="165" r="2.5" />
              <circle cx="298" cy="170" r="2.5" />
              <circle cx="342" cy="170" r="2.5" />
              <circle cx="320" cy="205" r="2.5" />
              <circle cx="290" cy="215" r="2.5" />
              <circle cx="350" cy="215" r="2.5" />
              <circle cx="320" cy="230" r="2.5" />
            </g>
          </svg>
        )}

        {/* Bottom-left HUD Telemetry Strip */}
        <div className="telemetry-hud-strip">
          <span className="hud-item">
            <span className="hud-label">Eye:</span>
            <span className={`hud-val ${telemetry.eye_state === 'CLOSED' ? 'red' : telemetry.eye_state === 'UNCERTAIN' ? 'amber' : 'green'}`}>{telemetry.eye_state || 'OPEN'}</span>
          </span>
          <span className="hud-item">
            <span className="hud-label">Closure:</span>
            <span className="hud-val cyan">{typeof telemetry.closure_duration_sec === 'number' ? telemetry.closure_duration_sec.toFixed(2) : '0.00'}s</span>
          </span>
          <span className="hud-item">
            <span className="hud-label">Latency:</span>
            <span className="hud-val green">{typeof telemetry.latency_ms === 'number' ? telemetry.latency_ms.toFixed(1) : '1.9'}ms</span>
          </span>
          <span className="hud-item">
            <span className="hud-label">EAR:</span>
            <span className="hud-val green">{typeof telemetry.ear === 'number' ? telemetry.ear.toFixed(2) : '0.35'}</span>
          </span>
          <span className="hud-item">
            <span className="hud-label">MAR:</span>
            <span className="hud-val green">{typeof telemetry.mar === 'number' ? telemetry.mar.toFixed(2) : '0.14'}</span>
          </span>
          <span className="hud-item">
            <span className="hud-label">Attention:</span>
            <span className="hud-val green">{telemetry.attention_score ?? 95}%</span>
          </span>
        </div>

        {/* Bottom-right Resolution / FPS pill */}
        <div className="resolution-hud-pill">
          <div className="fps-val">FPS: {Math.round(telemetry.fps || 30)}</div>
          <div className="res-val">1280 × 720</div>
        </div>
      </div>
    </div>
  );
}
