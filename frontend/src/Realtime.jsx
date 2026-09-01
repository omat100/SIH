import { useState, useEffect, useRef } from "react";

const MOCK_SENSORS = [
  { id: "SN-892-A", status: "normal" },
  { id: "SN-441-B", status: "warning" },
  { id: "SN-102-C", status: "normal" },
  { id: "SN-776-D", status: "normal" },
  { id: "SN-334-E", status: "warning" },
];

function generateTelemetry(sensor) {
  const base = sensor.status === "warning" ? 0.04 : 0.002;
  return {
    tilt: base + (Math.random() - 0.5) * 0.005,
    vib: sensor.status === "warning" ? 12 + Math.random() * 5 : 3 + Math.random() * 2,
    dist: sensor.status === "warning" ? -1.2 - Math.random() * 0.5 : -0.1 + Math.random() * 0.1,
    timestamp: new Date().toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit", fractionalSecondDigits: 3 }),
  };
}

export default function Realtime() {
  const [feedItems, setFeedItems] = useState([]);
  const [liveStatus, setLiveStatus] = useState(null);
  const [metrics, setMetrics] = useState({
    maxVelocity: 2.4,
    peakStress: 89,
    networkLoad: 94,
    activeNodes: 1024,
  });
  const feedRef = useRef(null);

  // Poll live ESP32 status/latest (real routes) to back header/metrics
  useEffect(() => {
    const fetchLive = async () => {
      try {
        const statusRes = await fetch("/api/live/status");
        setLiveStatus(await statusRes.json());
      } catch (e) {
        console.error("Realtime live fetch error:", e);
      }
    };
    fetchLive();
    const interval = setInterval(fetchLive, 5000);
    return () => clearInterval(interval);
  }, []);

  // Simulate incoming telemetry
  useEffect(() => {
    // Initial items
    const initial = MOCK_SENSORS.map((s, i) => ({
      id: `init-${i}`,
      sensor: s.id,
      status: s.status,
      ...generateTelemetry(s),
    }));
    setFeedItems(initial);

    const interval = setInterval(() => {
      const sensor = MOCK_SENSORS[Math.floor(Math.random() * MOCK_SENSORS.length)];
      const data = generateTelemetry(sensor);
      const newItem = {
        id: `item-${Date.now()}`,
        sensor: sensor.id,
        status: sensor.status,
        ...data,
      };
      setFeedItems((prev) => [newItem, ...prev.slice(0, 49)]);

      // Update metrics occasionally
      if (Math.random() > 0.7) {
        setMetrics((m) => ({
          ...m,
          maxVelocity: Math.max(m.maxVelocity, data.vib * 0.2),
          peakStress: sensor.status === "warning" ? Math.max(m.peakStress, 85 + Math.random() * 10) : m.peakStress,
        }));
      }
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  // Auto-scroll feed
  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = 0;
    }
  }, [feedItems]);

  const getStatusColor = (status) => {
    if (status === "warning") return "text-tertiary bg-tertiary/10 border-tertiary/20";
    return "text-primary bg-primary/10 border-primary/20";
  };

  const getStatusIcon = (status) => {
    if (status === "warning") return <span className="material-symbols-outlined text-[14px]">warning</span>;
    return null;
  };

  return (
    <div className="page-container realtime-page">
      {/* Page Header */}
      <div className="page-header realtime-header">
        <div>
          <h1 className="page-main-title">Real-time Sensor Feed</h1>
          <p className="page-subtitle">Live telemetry from Sector 7G underground network.</p>
        </div>
        <div className="header-status">
          <div className={`status-pill ${liveStatus?.connected ? "optimal" : ""}`}>
            <span className={`status-dot ${liveStatus?.connected ? "pulse" : "disconnected"}`} />
            {liveStatus?.connected ? "System Optimal" : "ESP32 Offline"}
          </div>
          <div className="status-pill update">
            <span className="material-symbols-outlined text-[16px]">sync</span>
            Buffered: {liveStatus?.buffered_days ?? 0}/30 days
          </div>
        </div>
      </div>

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
            {feedItems.map((item) => (
              <div
                key={item.id}
                className={`data-row glass-panel ${item.status === "warning" ? "warning-row" : ""}`}
              >
                <div className="row-header">
                  <span className={`sensor-id ${getStatusColor(item.status)}`}>
                    {getStatusIcon(item.status)}
                    {item.sensor}
                  </span>
                  <span className="timestamp">{item.timestamp}</span>
                </div>
                <div className="metrics-grid">
                  <div className="metric">
                    <span className="metric-label">TILT (Δ)</span>
                    <span className={`metric-value ${item.tilt > 0.01 ? "text-tertiary" : ""}`}>
                      {item.tilt > 0 ? "+" : ""}{item.tilt.toFixed(3)}°
                    </span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">VIB (Hz)</span>
                    <span className="metric-value">{item.vib.toFixed(1)}</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">DIST (mm)</span>
                    <span className={`metric-value ${item.dist < -0.5 ? "text-tertiary" : ""}`}>
                      {item.dist.toFixed(1)}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Column: Visualization & Details */}
        <div className="column-right">
          {/* 3D Spatial View */}
          <div className="glass-panel viz-panel">
            <div className="viz-header">
              <div className="viz-badges">
                <span className="badge primary">3D Spatial View</span>
                <span className="badge warning">2 Anomalies Detected</span>
              </div>
              <div className="viz-controls">
                <button className="icon-btn" aria-label="Zoom in">
                  <span className="material-symbols-outlined">add</span>
                </button>
                <button className="icon-btn" aria-label="Zoom out">
                  <span className="material-symbols-outlined">remove</span>
                </button>
              </div>
            </div>
            <div className="viz-canvas">
              <div className="viz-bg" />
              <div className="viz-grid" />
              {/* Anomaly markers */}
              <div className="anomaly-marker critical" style={{ top: "25%", left: "35%" }}>
                <span className="pulse-ring" />
                <span className="pulse-dot" />
              </div>
              <div className="anomaly-marker normal" style={{ top: "65%", left: "70%" }} />
              <div className="anomaly-marker normal" style={{ top: "30%", left: "20%" }} />
            </div>
          </div>

          {/* Metrics Bento Grid */}
          <div className="metrics-bento">
            <div className="glass-panel metric-card">
              <div className="metric-header">
                <span className="material-symbols-outlined text-primary">speed</span>
                <span className="metric-badge">Max Velocity</span>
              </div>
              <div className="metric-value-row">
                <span className="metric-big">{metrics.maxVelocity.toFixed(1)}</span>
                <span className="metric-unit">mm/s</span>
              </div>
            </div>

            <div className="glass-panel metric-card warning">
              <div className="metric-header">
                <span className="material-symbols-outlined text-tertiary">priority_high</span>
                <span className="metric-badge warning">Peak Stress</span>
              </div>
              <div className="metric-value-row">
                <span className="metric-big text-tertiary">{metrics.peakStress}</span>
                <span className="metric-unit text-tertiary/70">MPa</span>
              </div>
            </div>

            <div className="glass-panel metric-card wide">
              <div className="metric-header">
                <span className="metric-label">Buffered Days Progress</span>
                <span className="metric-value text-primary">{liveStatus?.buffered_days ?? 0}/30</span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${Math.min(100, ((liveStatus?.buffered_days ?? 0) / 30) * 100)}%` }}
                />
              </div>
              <div className="metric-footer">
                <span>{liveStatus?.readings_count ?? 0} Readings</span>
                <span>GBDT · LSTM</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}