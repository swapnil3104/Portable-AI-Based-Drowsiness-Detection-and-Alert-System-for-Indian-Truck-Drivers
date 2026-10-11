import React from 'react';
import { Bell, Volume2, Siren, Play, CheckCircle2, Radio } from 'lucide-react';
import './AlertSystem.css';

export default function AlertSystem({
  activeStage = 0,
  onPreviewAlert,
  voiceMessage = '',
  audioStatus = 'Online & Ready (Prerecorded MP3 & Offline TTS)',
}) {
  const stageMessages = {
    0: 'Normal monitoring active. Stay attentive.',
    1: 'Please stay alert. Keep your eyes on the road.',
    2: 'Warning! You may be getting drowsy. Please find a safe place to stop.',
    3: 'Danger! Pull over safely as soon as possible. Do not continue driving while drowsy.',
  };

  const stageTitles = {
    0: 'Normal Monitoring — Attentive',
    1: 'Stage 1 — Attention Warning',
    2: 'Stage 2 — Drowsiness Warning',
    3: 'Stage 3 — Critical Warning',
  };

  const currentMessage = voiceMessage || stageMessages[activeStage] || stageMessages[0];
  const currentTitle = stageTitles[activeStage] || stageTitles[0];

  return (
    <div className="card alert-system-card">
      <div className="card-header">
        <div className="card-title-group">
          <div className="alert-header-icon">
            <Bell className="w-5 h-5 text-amber-500" fill="currentColor" />
          </div>
          <div>
            <h3 className="card-title">Alert System</h3>
            <p className="card-subtitle">Three-stage escalating audio-visual directives</p>
          </div>
        </div>

        <button
          className="btn-preview-alert"
          onClick={() => onPreviewAlert && onPreviewAlert(activeStage > 0 ? activeStage : 1)}
          title="Test Alert System Audio and Visual Warning"
          id="preview-alert-btn"
        >
          <Play size={12} fill="currentColor" />
          <span>Preview Alert</span>
        </button>
      </div>

      {/* Active Stage & Spoken Voice Message Box */}
      <div className={`active-alert-banner stage-${activeStage}`}>
        <div className="alert-banner-header">
          <span className="alert-banner-tag">{currentTitle}</span>
          <span className="audio-hardware-status">
            <Radio size={12} className="inline mr-1" />
            {audioStatus}
          </span>
        </div>
        <p className="active-voice-message">
          <span className="voice-prefix">Voice Directive:</span> "{currentMessage}"
        </p>
      </div>

      <div className="alert-stages-grid">
        {/* Stage 1 */}
        <div className={`stage-card stage-1 ${activeStage === 1 ? 'is-active' : ''}`}>
          <div className="stage-icon-wrap green">
            <Volume2 size={24} />
          </div>
          <div className="stage-info">
            <div className="stage-name">Stage 1</div>
            <div className="stage-type">Attention Warning</div>
            <div className="stage-desc">Gentle Beep + Voice</div>
          </div>
          <div className="stage-status-pill">
            <span className={`status-indicator-dot ${activeStage === 1 ? 'green' : 'gray'}`}></span>
            <span className="stage-status-label">{activeStage === 1 ? 'Active' : 'Ready'}</span>
          </div>
        </div>

        {/* Stage 2 */}
        <div className={`stage-card stage-2 ${activeStage === 2 ? 'is-active' : ''}`}>
          <div className="stage-icon-wrap amber">
            <Volume2 size={24} />
          </div>
          <div className="stage-info">
            <div className="stage-name">Stage 2</div>
            <div className="stage-type">Drowsiness Warning</div>
            <div className="stage-desc">Firm Repeating Beep</div>
          </div>
          <div className="stage-status-pill">
            <span className={`status-indicator-dot ${activeStage === 2 ? 'amber' : 'gray'}`}></span>
            <span className="stage-status-label">{activeStage === 2 ? 'Active' : 'Ready'}</span>
          </div>
        </div>

        {/* Stage 3 */}
        <div className={`stage-card stage-3 ${activeStage === 3 ? 'is-active' : ''}`}>
          <div className="stage-icon-wrap red">
            <Siren size={24} />
          </div>
          <div className="stage-info">
            <div className="stage-name">Stage 3</div>
            <div className="stage-type">Critical Warning</div>
            <div className="stage-desc">Urgent Alarm + Safe Stop</div>
          </div>
          <div className="stage-status-pill">
            <span className={`status-indicator-dot ${activeStage === 3 ? 'red' : 'gray'}`}></span>
            <span className="stage-status-label">{activeStage === 3 ? 'Active' : 'Ready'}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
