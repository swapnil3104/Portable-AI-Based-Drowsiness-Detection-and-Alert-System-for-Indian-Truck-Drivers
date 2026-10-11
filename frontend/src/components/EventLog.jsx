import React from 'react';
import { FileText, Trash2, CheckCircle2, AlertCircle, AlertTriangle } from 'lucide-react';
import './EventLog.css';

export default function EventLog({ events = [], onClearLog }) {
  const displayEvents = events.length > 0 ? events : [
    { timestamp: '17:23:15', event_type: 'SYSTEM_ONLINE', stage: 0, description: 'DrishtiDrive AI Core Initialized', measurements: 'Model: Depthwise-separable CNN', severity: 'info' },
    { timestamp: '17:23:16', event_type: 'CAMERA_CONNECTED', stage: 0, description: 'Camera 0 Connected (1280x720)', measurements: 'FPS Target: 30', severity: 'info' },
    { timestamp: '17:23:20', event_type: 'DRIVER_ACQUIRED', stage: 0, description: 'Driver Face & Eyes Tracked', measurements: 'Confidence: 98.5%', severity: 'info' },
    { timestamp: '17:23:45', event_type: 'OBSERVATION_NORMAL', stage: 0, description: 'Attentive Monitoring Active', measurements: 'Closure: 0.00s', severity: 'info' },
  ];

  const formatTime = (ts) => {
    if (!ts) return '12:00:00';
    if (ts.length <= 8) return ts;
    try {
      const d = new Date(ts);
      return d.toLocaleTimeString([], { hour12: false });
    } catch {
      return ts.slice(11, 19);
    }
  };

  return (
    <div className="card event-log-card">
      <div className="card-header">
        <div className="card-title-group">
          <div className="log-icon-pill">
            <FileText size={18} />
          </div>
          <div>
            <h3 className="card-title">Event Log</h3>
            <p className="card-subtitle">Timestamps, alert stages & duration metrics</p>
          </div>
        </div>

        <button className="btn-clear-log" onClick={onClearLog} id="clear-log-btn" title="Clear Event Log History">
          <Trash2 size={13} />
          <span>Clear Log</span>
        </button>
      </div>

      <div className="log-entries-scroll" id="log-entries-scroll">
        {displayEvents.map((e, index) => {
          const isCritical = e.severity === 'critical' || e.stage === 3;
          const isWarning = e.severity === 'warning' || e.stage === 2 || e.stage === 1;
          const stageBadge = e.stage > 0 ? `Stage ${e.stage}` : null;

          return (
            <div key={index} className="log-row">
              <span className="log-time">{formatTime(e.timestamp)}</span>
              <span className={`log-indicator ${isCritical ? 'critical' : isWarning ? 'alert' : 'success'}`}>
                {isCritical ? (
                  <AlertTriangle size={14} className="text-red-500" />
                ) : isWarning ? (
                  <AlertCircle size={14} className="text-amber-500" />
                ) : (
                  <CheckCircle2 size={14} className="text-emerald-500" />
                )}
              </span>
              <div className="log-content-col">
                <div className="log-desc-line">
                  {stageBadge && <span className={`log-stage-tag stage-${e.stage}`}>{stageBadge}</span>}
                  <span className="log-desc">{e.description}</span>
                </div>
                {e.measurements && (
                  <span className="log-measurements">{e.measurements}</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
