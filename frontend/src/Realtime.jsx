import { useEffect, useRef } from "react";
import { useLiveData } from "./useLiveData";

export default function Realtime() {
  const {
    connected,
    latestReading,
    readings,
    readingsCount,
    bufferedDays,
    lastPrediction,
    hasAnyData,
  } = useLiveData();
  const feedRef = useRef(null);

  const feedItems = [...readings].reverse();

  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = 0;
    }
  }, [readings]);

  const riskClass = lastPrediction?.risk_class || null;

  return (
    <div className="page-container realtime-page">
      {/* Page Header */}
      <div className="page-header realtime-header">
        <div>
          <h1 className="page-main-title">Real-time Sensor Feed</h1>
          <p className="page-subtitle">Live readings from the COM8 sensor network.</p>
        </div>
        <div className="header-status">
          <div className={`status-pill ${connected ? "optimal" : ""}`}>
            <span className={`status-dot ${connected ? "pulse" : "disconnected"}`} />
            {connected ? "System Optimal" : "No Data Fetched"}
          </div>
          <div className="status-pill update">
            <span className="material-symbols-outlined text-[16px]">sync</span>
            Buffered: {bufferedDays ?? 0}/30 days
          </div>
        </div>
      </div>

      {!hasAnyData ? (
        <div className="empty-state glass-panel">
          <span className="material-symbols-outlined empty-icon">sensors_off</span>
          <h3>No Data Fetched</h3>
          <p>Waiting for readings from the sensor on COM8. Connect the device and data will appear here in real time.</p>
        </div>
      ) : (
        <div className="realtime-grid">
          {/* Left Column: Telemetry Stream */}
          <div className="column-left glass-panel">
            <div className="panel-header">
              <h3 className="panel-title">
                <span className="material-symbols-outlined">sensors</span>
                Telemetry Stream
              </h3>
              <button className="icon-btn" aria-label="Filter">
                <span className="material-symbols-outlined">filter_list</span>
              </button>
            </div>
            <div className="feed-container" ref={feedRef} role="log" aria-live="polite">
              {feedItems.map((item, i) => {
                const warn =
                  Math.abs(item.tilt_deg) > 0.01 || /critical|watch/.test(riskClass || "");
                return (
                  <div
                    key={`${item.date}-${i}`}
                    className={`data-row glass-panel ${warn ? "warning-row" : ""}`}
                  >
                    <div className="row-header">
                      <span className={`sensor-id text-primary bg-primary/10 border-primary/20`}>
                        {item.date}
                      </span>
                      <span className="timestamp">reading</span>
                    </div>
                    <div className="metrics-grid">
                      <div className="metric">
                        <span className="metric-label">TEMP (C)</span>
                        <span className="metric-value">{item.temp_c?.toFixed(1)}</span>
                      </div>
                      <div className="metric">
                        <span className="metric-label">HUMIDITY (%)</span>
                        <span className="metric-value">{item.humidity_pct?.toFixed(1)}</span>
                      </div>
                      <div className="metric">
                        <span className="metric-label">TILT (deg)</span>
                        <span className={`metric-value ${warn ? "text-tertiary" : ""}`}>
                          {item.tilt_deg?.toFixed(4)}
                        </span>
                      </div>
                      <div className="metric">
                        <span className="metric-label">DIST (mm)</span>
                        <span className={`metric-value ${warn ? "text-tertiary" : ""}`}>
                          {item.distance_mm?.toFixed(1)}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Right Column: Visualization & Details */}
          <div className="column-right">
            {/* Live Prediction Status */}
            <div className="glass-panel viz-panel">
              <div className="viz-header">
                <div className="viz-badges">
                  <span className="badge primary">Live Prediction Status</span>
                  <span className={`badge ${riskClass === "critical" ? "warning" : ""}`}>
                    {riskClass ? riskClass.toUpperCase() : "No AI result yet"}
                  </span>
                </div>
                <div className="viz-controls">
                  <span className={`status-dot ${connected ? "pulse" : "disconnected"}`} />
                </div>
              </div>
              <div className="viz-canvas">
                <div className="viz-bg" />
                <div className="viz-grid" />
                {riskClass && (
                  <div className={`anomaly-marker ${riskClass === "critical" ? "critical" : "normal"}`} style={{ top: "45%", left: "45%" }}>
                    <span className="pulse-ring" />
                    <span className="pulse-dot" />
                  </div>
                )}
              </div>
            </div>

            {/* Metrics Bento Grid */}
            <div className="metrics-bento">
              <div className="glass-panel metric-card">
                <div className="metric-header">
                  <span className="material-symbols-outlined text-primary">straighten</span>
                  <span className="metric-badge">Latest Tilt</span>
                </div>
                <div className="metric-value-row">
                  <span className="metric-big">{latestReading?.tilt_deg?.toFixed(3) ?? "—"}</span>
                  <span className="metric-unit">deg</span>
                </div>
              </div>

              <div className="glass-panel metric-card warning">
                <div className="metric-header">
                  <span className="material-symbols-outlined text-tertiary">height</span>
                  <span className="metric-badge warning">Latest Distance</span>
                </div>
                <div className="metric-value-row">
                  <span className="metric-big text-tertiary">{latestReading?.distance_mm?.toFixed(1) ?? "—"}</span>
                  <span className="metric-unit text-tertiary/70">mm</span>
                </div>
              </div>

              <div className="glass-panel metric-card wide">
                <div className="metric-header">
                  <span className="metric-label">Buffered Days Progress</span>
                  <span className="metric-value text-primary">{bufferedDays ?? 0}/30</span>
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${Math.min(100, ((bufferedDays ?? 0) / 30) * 100)}%` }}
                  />
                </div>
                <div className="metric-footer">
                  <span>{readingsCount ?? 0} Readings</span>
                  <span>GBDT · LSTM</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
