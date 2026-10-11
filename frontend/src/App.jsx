import React, { useState } from 'react';
import Header from './components/Header';
import LiveMonitoring from './components/LiveMonitoring';
import CameraControls from './components/CameraControls';
import AlertSystem from './components/AlertSystem';
import DriverStatus from './components/DriverStatus';
import DriverMetrics from './components/DriverMetrics';
import EventLog from './components/EventLog';
import { useCameraStream } from './hooks/useCameraStream';
import './App.css';

export default function App() {
  const {
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
  } = useCameraStream();

  const [showLandmarks, setShowLandmarks] = useState(true);
  const [resolution, setResolution] = useState('1280x720');

  const activeStage = streamData?.telemetry?.stage ?? 0;
  const driverStatus = streamData?.telemetry?.status_text || 'Normal';
  const attentionScore = streamData?.telemetry?.attention_score ?? 95;
  const voiceMessage = streamData?.telemetry?.voice_message || '';
  const audioStatus = streamData?.telemetry?.audio_status || 'Ready';

  return (
    <div className="drishti-app">
      {/* Top Header */}
      <Header systemOnline={true} />

      {/* Main Dashboard Grid */}
      <main className="dashboard-layout">
        {/* Left Column: Live Monitoring + Controls + Alert System */}
        <div className="dashboard-left-col">
          {/* Live Driver Monitoring Card */}
          <LiveMonitoring
            streamData={streamData}
            onStartCamera={() => startCamera(selectedDevice)}
            onStopCamera={stopCamera}
            showLandmarks={showLandmarks}
          />

          {/* Lower Two Cards Grid */}
          <div className="bottom-cards-grid">
            <CameraControls
              devices={devices}
              selectedDevice={selectedDevice}
              onSelectDevice={(id) => {
                setSelectedDevice(id);
                if (streamData.active) startCamera(id);
              }}
              sensitivity={sensitivity}
              onChangeSensitivity={updateSensitivity}
              showLandmarks={showLandmarks}
              onToggleLandmarks={setShowLandmarks}
              resolution={resolution}
              onChangeResolution={setResolution}
            />

            <AlertSystem
              activeStage={activeStage}
              onPreviewAlert={previewAlert}
              voiceMessage={voiceMessage}
              audioStatus={audioStatus}
            />
          </div>
        </div>

        {/* Right Column: Driver Status + Metrics + Event Log */}
        <div className="dashboard-right-col">
          <DriverStatus
            status={driverStatus}
            attentionScore={attentionScore}
          />

          <DriverMetrics
            telemetry={streamData?.telemetry}
          />

          <EventLog
            events={events}
            onClearLog={clearLog}
          />
        </div>
      </main>
    </div>
  );
}
