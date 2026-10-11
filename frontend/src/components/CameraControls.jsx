import React from 'react';
import { Camera, Sliders } from 'lucide-react';
import './CameraControls.css';

export default function CameraControls({
  devices = [],
  selectedDevice = 0,
  onSelectDevice,
  sensitivity = 1.0,
  onChangeSensitivity,
  showLandmarks = true,
  onToggleLandmarks,
  resolution = '1280x720',
  onChangeResolution,
}) {
  return (
    <div className="card camera-controls-card">
      <div className="card-header-clean">
        <div className="card-title-group">
          <div className="control-icon-pill">
            <Camera size={18} />
          </div>
          <h3 className="card-title">Camera Controls</h3>
        </div>
      </div>

      <div className="controls-grid">
        {/* Camera Source Selector */}
        <div className="control-field">
          <label className="field-label" htmlFor="camera-select">Camera Source</label>
          <div className="select-wrapper">
            <select
              id="camera-select"
              className="custom-select"
              value={selectedDevice}
              onChange={(e) => {
                const val = e.target.value;
                const num = Number(val);
                onSelectDevice(!isNaN(num) && val.trim() !== '' ? num : val);
              }}
            >
              {devices.length > 0 ? (
                devices.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))
              ) : (
                <option value={0}>Camera 1 (Web Camera / HD)</option>
              )}
            </select>
          </div>
        </div>

        {/* Detection Sensitivity Slider */}
        <div className="control-field">
          <label className="field-label" htmlFor="sensitivity-slider">Detection Sensitivity</label>
          <div className="slider-wrapper">
            <input
              id="sensitivity-slider"
              type="range"
              min="0.5"
              max="1.5"
              step="0.05"
              value={sensitivity}
              onChange={(e) => onChangeSensitivity(Number(e.target.value))}
              className="custom-slider"
            />
            <div className="slider-labels">
              <span>More Sensitive</span>
              <span>Less Sensitive</span>
            </div>
          </div>
        </div>

        {/* Show Facial Landmarks Toggle */}
        <div className="control-field toggle-field">
          <label className="toggle-label" htmlFor="landmarks-toggle">
            <div className="switch">
              <input
                id="landmarks-toggle"
                type="checkbox"
                checked={showLandmarks}
                onChange={(e) => onToggleLandmarks(e.target.checked)}
              />
              <span className="slider round"></span>
            </div>
            <span className="toggle-text">Show Facial Landmarks</span>
          </label>
        </div>

        {/* Resolution Selector */}
        <div className="control-field">
          <label className="field-label" htmlFor="resolution-select">Resolution</label>
          <div className="select-wrapper">
            <select
              id="resolution-select"
              className="custom-select"
              value={resolution}
              onChange={(e) => onChangeResolution && onChangeResolution(e.target.value)}
            >
              <option value="1280x720">1280 × 720 (HD)</option>
              <option value="1920x1080">1920 × 1080 (Full HD)</option>
              <option value="640x480">640 × 480 (SD)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Privacy default notice */}
      <div className="privacy-guarantee-strip" role="note">
        <span className="privacy-badge">🔒 Local Privacy</span>
        <span className="privacy-text">Zero facial images or video recorded. All inference is strictly ephemeral.</span>
      </div>
    </div>
  );
}
