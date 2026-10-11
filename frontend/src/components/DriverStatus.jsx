import React from 'react';
import { User, Activity } from 'lucide-react';
import './DriverStatus.css';

export default function DriverStatus({ status = 'Normal', attentionScore = 95 }) {
  // SVG circular gauge math: radius 52 -> circumference 326.7
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (attentionScore / 100) * circumference;

  const isAlert = attentionScore >= 75;

  return (
    <div className="card driver-status-card">
      <div className="card-header">
        <div className="card-title-group">
          <div className="user-icon-pill">
            <User size={18} />
          </div>
          <h3 className="card-title">Driver Status</h3>
        </div>

        <span className={`badge-pill ${status === 'Normal' ? 'success' : status.startsWith('Stage') ? 'danger' : 'warning'}`}>
          {status}
        </span>
      </div>

      <div className="status-content-row">
        {/* Circular Progress Gauge */}
        <div className="gauge-container">
          <svg className="circular-gauge" viewBox="0 0 130 130">
            {/* Background track */}
            <circle
              cx="65"
              cy="65"
              r={radius}
              className="gauge-track"
              strokeWidth="11"
            />
            {/* Value stroke */}
            <circle
              cx="65"
              cy="65"
              r={radius}
              className="gauge-value"
              strokeWidth="11"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
            />
          </svg>
          <div className="gauge-text-overlay">
            <span className="gauge-score">{attentionScore}%</span>
            <span className="gauge-label">Attention Score</span>
          </div>
        </div>

        {/* Status description */}
        <div className="status-description-box">
          <div className="status-sub">Driver is</div>
          <div className="status-main-alert">
            <span className="alert-word">{isAlert ? 'Alert' : 'Drowsy'}</span>
            <Activity className="pulse-wave-icon" size={24} />
          </div>
          <div className="status-advice">{isAlert ? 'Keep Driving Safe!' : 'Pull Over Safely & Rest'}</div>
        </div>
      </div>

      {/* Experimental risk indicator notice required by safety standard */}
      <div className="experimental-disclaimer-pill" role="note">
        <span className="disclaimer-icon">ℹ</span>
        <span>Experimental estimate only — not a medical diagnosis or validated fatigue score.</span>
      </div>
    </div>
  );
}
