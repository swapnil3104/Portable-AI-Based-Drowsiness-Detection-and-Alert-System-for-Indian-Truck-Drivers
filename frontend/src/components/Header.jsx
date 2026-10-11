import React from 'react';
import { Moon, Minus, Square, X } from 'lucide-react';
import './Header.css';

export default function Header({ systemOnline = true }) {
  return (
    <header className="app-header">
      <div className="brand-group">
        {/* Eye and road logo mark */}
        <div className="logo-mark" aria-hidden="true">
          <svg viewBox="0 0 48 32" fill="none" className="logo-svg">
            {/* Outer stylized eye outline */}
            <path
              d="M4 16C10 6 38 6 44 16C38 26 10 26 4 16Z"
              stroke="#1877f2"
              strokeWidth="2.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
            {/* Iris / Inner road curves */}
            <circle cx="24" cy="16" r="8" fill="#1877f2" fillOpacity="0.1" stroke="#1877f2" strokeWidth="2.2" />
            <path d="M21 24L23 16L24 10" stroke="#1877f2" strokeWidth="1.8" strokeLinecap="round" />
            <path d="M27 24L25 16L24 10" stroke="#1877f2" strokeWidth="1.8" strokeLinecap="round" />
            <path d="M24 15V13" stroke="#ffffff" strokeWidth="1.8" strokeLinecap="round" />
            <path d="M24 21V19" stroke="#ffffff" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        </div>
        <div className="brand-text">
          <h1 className="brand-title">DrishtiDrive AI</h1>
          <p className="brand-tagline">A Watchful Eye on Every Journey.</p>
        </div>
      </div>

      <div className="header-actions">
        <button className="icon-btn" title="Toggle Dark/Light Mode" aria-label="Toggle Theme">
          <Moon className="w-5 h-5 text-slate-500" />
        </button>
        <div className="window-controls" aria-hidden="true">
          <button className="win-btn" title="Minimize"><Minus size={14} /></button>
          <button className="win-btn" title="Maximize"><Square size={12} /></button>
          <button className="win-btn win-close" title="Close"><X size={14} /></button>
        </div>
        <div className="system-status-indicator">
          <span className={`status-dot ${systemOnline ? 'online' : 'fault'}`}></span>
          <span className="status-label">{systemOnline ? 'System Online' : 'System Offline'}</span>
        </div>
      </div>
    </header>
  );
}
