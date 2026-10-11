import React from 'react';
import { BarChart2, Eye, EyeOff, Clock, Gauge, Cpu, Activity, Smile } from 'lucide-react';
import './DriverMetrics.css';

export default function DriverMetrics({ telemetry }) {
  const t = telemetry || {
    eye_state: 'OPEN',
    confidence: 0.95,
    closure_duration_sec: 0.0,
    latency_ms: 1.93,
    fps: 30.0,
    attention_score: 95,
    ear: 0.32,
    mar: 0.15,
    bpm: 18,
  };

  const eyeState = t.eye_state || 'OPEN';
  const confidencePct = Math.round((t.confidence || 0.95) * 100);
  const closureSec = typeof t.closure_duration_sec === 'number' ? t.closure_duration_sec.toFixed(2) : '0.00';
  const latencyMs = typeof t.latency_ms === 'number' ? t.latency_ms.toFixed(2) : '1.93';
  const fpsVal = typeof t.fps === 'number' ? t.fps.toFixed(1) : '30.0';
  const earVal = typeof t.ear === 'number' ? t.ear.toFixed(2) : '0.32';
  const marVal = typeof t.mar === 'number' ? t.mar.toFixed(2) : '0.15';
  const attentionScore = t.attention_score || 95;
  const bpm = t.bpm || 18;

  // Dynamic status classes
  const eyeStateClass = eyeState === 'OPEN' ? 'state-open' : eyeState === 'CLOSED' ? 'state-closed' : 'state-uncertain';
  const closureColorClass = t.closure_duration_sec >= 1.8 ? 'red' : t.closure_duration_sec >= 0.8 ? 'amber' : 'green';

  const earPercent = Math.min(100, Math.max(0, (Number(earVal) / 0.40) * 100));
  const marPercent = Math.min(100, Math.max(0, (Number(marVal) / 0.60) * 100));
  const blinkPercent = Math.min(100, Math.max(0, (bpm / 35) * 100));

  return (
    <div className="card driver-metrics-card">
      <div className="card-header">
        <div className="card-title-group">
          <div className="metrics-icon-pill">
            <BarChart2 size={18} />
          </div>
          <div>
            <h3 className="card-title">Driver Metrics</h3>
            <p className="card-subtitle">Real-time eye state, temporal closure & latency</p>
          </div>
        </div>
      </div>

      <div className="metrics-list">
        {/* Metric 1: Attention Score */}
        <div className="metric-row">
          <div className="metric-header-row">
            <div className="metric-name-group">
              <Eye size={17} className="metric-icon text-emerald-500" />
              <span className="metric-label">Attention Score</span>
            </div>
            <span className="metric-value green">{attentionScore}%</span>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill green"
              style={{ width: `${attentionScore}%` }}
            ></div>
          </div>
        </div>

        {/* Metric 2: Eye State & Confidence */}
        <div className="metric-row">
          <div className="metric-header-row">
            <div className="metric-name-group">
              <Eye size={17} className="metric-icon text-sky-500" />
              <span className="metric-label">Eye State</span>
            </div>
            <div className="metric-val-group">
              <span className={`eye-state-badge ${eyeStateClass}`}>{eyeState}</span>
              <span className="metric-value blue">{confidencePct}%</span>
            </div>
          </div>
        </div>

        {/* Metric 3: Eye Openness (EAR) */}
        <div className="metric-row">
          <div className="metric-header-row">
            <div className="metric-name-group">
              <EyeOff size={17} className="metric-icon text-sky-500" />
              <span className="metric-label">Eye Openness (EAR)</span>
            </div>
            <span className="metric-value blue">{earVal}</span>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill blue"
              style={{ width: `${earPercent}%` }}
            ></div>
          </div>
        </div>

        {/* Metric 4: Sustained Closure Duration */}
        <div className="metric-row">
          <div className="metric-header-row">
            <div className="metric-name-group">
              <Clock size={17} className="metric-icon text-amber-500" />
              <span className="metric-label">Closure Duration</span>
            </div>
            <span className={`metric-value ${closureColorClass}`}>{closureSec}s</span>
          </div>
          <div className="progress-track">
            <div
              className={`progress-fill ${closureColorClass}`}
              style={{ width: `${Math.min(100, (Number(closureSec) / 3.0) * 100)}%` }}
            ></div>
          </div>
        </div>

        {/* Metric 5: Mouth Openness (MAR) */}
        <div className="metric-row">
          <div className="metric-header-row">
            <div className="metric-name-group">
              <Smile size={17} className="metric-icon text-amber-500" />
              <span className="metric-label">Mouth Openness (MAR)</span>
            </div>
            <span className="metric-value amber">{marVal}</span>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill amber"
              style={{ width: `${marPercent}%` }}
            ></div>
          </div>
        </div>

        {/* Metric 6: Blink Rate */}
        <div className="metric-row">
          <div className="metric-header-row">
            <div className="metric-name-group">
              <Activity size={17} className="metric-icon text-purple-500" />
              <span className="metric-label">Blink Rate</span>
            </div>
            <span className="metric-value purple">{bpm} bpm</span>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill purple"
              style={{ width: `${blinkPercent}%` }}
            ></div>
          </div>
        </div>

        {/* Metric 7: Inference Latency & FPS */}
        <div className="metric-row">
          <div className="metric-header-row">
            <div className="metric-name-group">
              <Cpu size={17} className="metric-icon text-emerald-500" />
              <span className="metric-label">Inference Latency</span>
            </div>
            <span className="metric-value green">{latencyMs} ms ({fpsVal} FPS)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
