import { useState, useRef, useEffect } from "react";
import { useLiveData } from "./useLiveData";

export default function Analytics() {
  const [timeRange, setTimeRange] = useState("1W");
  const [health, setHealth] = useState(null);
  const chartRef = useRef(null);
  const { connected, status, readings, readingsCount, bufferedDays, lastPrediction, hasAnyData } =
    useLiveData();

  useEffect(() => {
    let active = true;
    fetch("/api/manual/health")
      .then((r) => r.json())
      .then((data) => {
        if (active) setHealth({ health: data, status: status });
      })
      .catch((e) => console.error("Analytics health fetch error:", e));
    return () => {
      active = false;
    };
    // status intentionally read-only for diagnostics; re-fetch on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Model the readings series for the selected range
  const series = (() => {
    if (!hasAnyData) return [];
    const limit = timeRange === "1D" ? 24 : timeRange === "1W" ? 7 * 4 : timeRange === "1M" ? 30 : 90;
    return [...readings].slice(-limit);
  })();

  // Draw chart from real readings
  useEffect(() => {
    const canvas = chartRef.current;
    if (!canvas || series.length === 0) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.offsetWidth * dpr;
    canvas.height = canvas.offsetHeight * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;
    const padding = { top: 20, right: 20, bottom: 40, left: 60 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    const distVals = series.map((p) => p.distance_mm);
    const tiltVals = series.map((p) => p.tilt_deg);
    const minY = Math.min(...distVals, ...tiltVals) - 2;
    const maxY = Math.max(...distVals, ...tiltVals) + 2;
    const yScale = (y) => padding.top + plotHeight * (1 - (y - minY) / (maxY - minY));
    const xScale = (i) => padding.left + (plotWidth / Math.max(series.length - 1, 1)) * i;

    // Grid lines
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.lineWidth = 1;
    [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
      const y = padding.top + plotHeight * t;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
    });

    // Y-axis labels
    ctx.fillStyle = "rgba(194,198,214,0.6)";
    ctx.font = "10px Geist, Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    [minY, (minY + maxY) / 2, maxY].forEach((val, i) => {
      const y = padding.top + plotHeight * (i / 2);
      ctx.fillText(val.toFixed(0), padding.left - 10, y);
    });

    const drawSeries = (values, color) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      values.forEach((v, i) => {
        const x = xScale(i);
        const y = yScale(v);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    };

    drawSeries(distVals, "#4edea3");
    drawSeries(tiltVals, "#adc6ff");

    // X-axis labels
    ctx.fillStyle = "rgba(194,198,214,0.6)";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const count = series.length;
    const labelEvery = Math.max(1, Math.floor(count / 6));
    series.forEach((p, i) => {
      if (i % labelEvery !== 0) return;
      const label = (p.date || "").slice(5);
      ctx.fillText(label, xScale(i), height - padding.bottom + 8);
    });
  }, [series, hasAnyData]);

  const riskClass = lastPrediction?.risk_class || null;

  return (
    <div className="page-container analytics-page">
      {/* Page Header */}
      <div className="page-header analytics-header">
        <div>
          <h2 className="page-main-title">Analytics</h2>
          <p className="page-subtitle">Historical trends from live sensor readings.</p>
        </div>
        <div className="header-actions">
          <div className="segmented-control">
            {["1D", "1W", "1M"].map((r) => (
              <button
                key={r}
                className={`segment ${timeRange === r ? "active" : ""}`}
                onClick={() => setTimeRange(r)}
              >
                {r}
              </button>
            ))}
          </div>
          <button className="btn-secondary icon-btn-lg outline">
            <span className="material-symbols-outlined">download</span>
            Export Data
          </button>
        </div>
      </div>

      {!hasAnyData ? (
        <div className="empty-state glass-panel">
          <span className="material-symbols-outlined empty-icon">monitoring</span>
          <h3>No Data Fetched</h3>
          <p>Waiting for readings from the sensor on COM8. Historical analytics will appear once data is available.</p>
        </div>
      ) : (
        <div className="bento-grid">
          {/* Main Trend Chart */}
          <div className="glass-panel chart-panel">
            <div className="panel-header">
              <div>
                <h3 className="panel-title">Sensor Trends</h3>
                <p className="panel-subtitle">Distance (mm) and tilt (deg) over the selected period</p>
              </div>
              <div className="series-legend">
                <span className="legend-item secondary">
                  <span className="legend-dot" />
                  Distance (mm)
                </span>
                <span className="legend-item primary">
                  <span className="legend-dot" />
                  Tilt (deg)
                </span>
              </div>
            </div>
            <div className="chart-wrapper">
              {series.length === 0 ? (
                <div className="empty-state">
                  <h3>No readings in range</h3>
                </div>
              ) : (
                <canvas ref={chartRef} className="chart-canvas" />
              )}
            </div>
          </div>

          {/* Risk Status */}
          <div className="glass-panel heatmap-panel">
            <div className="panel-header">
              <h3 className="panel-title">Current Risk</h3>
              <p className="panel-subtitle">Latest AI prediction</p>
            </div>
            <div className="heatmap-canvas">
              <div className="heatmap-bg" />
              <div className="heatmap-grid" />
              <div
                className={`risk-marker ${riskClass === "critical" ? "critical" : riskClass === "watch" ? "warning" : "primary"}`}
                style={{ top: "45%", left: "45%" }}
              >
                <span className="ping-ring" />
                <span className="ping-dot" />
              </div>
            </div>
            <div className="risk-scale">
              <span className="text-on-surface-variant">Stable</span>
              <div className="scale-bar">
                <div className="scale-gradient" />
              </div>
              <span className="text-tertiary">Critical</span>
            </div>
          </div>

          {/* Key Metrics Row */}
          <div className="glass-panel metric-panel">
            <div className="metric-header">
              <span className="material-symbols-outlined text-primary">sensors</span>
              <span className={`metric-tag ${connected ? "secondary" : "critical"}`}>
                {connected ? "CONNECTED" : "OFFLINE"}
              </span>
            </div>
            <p className="metric-label">Live Sensor Status</p>
            <p className="metric-big glow">{bufferedDays ?? 0} <span className="metric-unit">/ 30 days</span></p>
          </div>

          <div className="glass-panel metric-panel warning">
            <div className="metric-header">
              <span className="material-symbols-outlined text-tertiary">database</span>
              <span className={`metric-tag ${bufferedDays >= 30 ? "secondary" : "critical"}`}>
                {bufferedDays >= 30 ? "READY" : "BUFFERING"}
              </span>
            </div>
            <p className="metric-label">Buffered Readings</p>
            <p className="metric-big">{readingsCount ?? 0}</p>
          </div>

          {/* Model Performance */}
          <div className="glass-panel model-panel wide">
            <div className="panel-header">
              <h3 className="panel-title">Predictive Model Status</h3>
              <span className="badge primary">Models: {health?.health?.models ? Object.values(health.health.models).filter((s) => s === "loaded").length : 0}/{health?.health?.models ? Object.keys(health.health.models).length : 0} loaded</span>
            </div>
            <div className="diagnostics-grid">
              <div className="diagnostic">
                <div className="diag-header">
                  <span>GBDT Model</span>
                  <span className={`diag-value ${health?.health?.models?.gbdt === "loaded" ? "secondary" : "tertiary"}`}>
                    {health?.health?.models?.gbdt ? health.health.models.gbdt.toUpperCase() : "..."}
                  </span>
                </div>
                <div className="diag-bar">
                  <div className="diag-fill secondary" style={{ width: health?.health?.models?.gbdt === "loaded" ? "100%" : "0%" }} />
                </div>
              </div>
              <div className="diagnostic">
                <div className="diag-header">
                  <span>LSTM Model</span>
                  <span className={`diag-value ${health?.health?.models?.torch === "loaded" ? "secondary" : "tertiary"}`}>
                    {health?.health?.models?.torch ? health.health.models.torch.toUpperCase() : "..."}
                  </span>
                </div>
                <div className="diag-bar">
                  <div className="diag-fill primary" style={{ width: health?.health?.models?.torch === "loaded" ? "100%" : "0%" }} />
                </div>
              </div>
              <div className="diagnostic">
                <div className="diag-header">
                  <span>Buffered Days</span>
                  <span className="diag-value">{bufferedDays ?? 0}</span>
                </div>
                <div className="diag-bar">
                  <div className="diag-fill neutral" style={{ width: `${Math.min(100, ((bufferedDays ?? 0) / 30) * 100)}%` }} />
                </div>
              </div>
              <div className="diagnostic">
                <div className="diag-header">
                  <span>Predictions Available</span>
                  <span className="diag-value tertiary">
                    {lastPrediction ? 1 : 0}
                  </span>
                </div>
                <div className="diag-bar">
                  <div className="diag-fill tertiary" style={{ width: `${lastPrediction ? 100 : 0}%` }} />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
