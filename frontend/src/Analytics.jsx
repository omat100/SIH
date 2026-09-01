import { useState, useRef, useEffect } from "react";

export default function Analytics() {
  const [timeRange, setTimeRange] = useState("1W");
  const [health, setHealth] = useState(null);
  const chartRef = useRef(null);

  // Fetch real model health + live status to back diagnostic panels
  useEffect(() => {
    let active = true;
    Promise.all([
      fetch("/api/manual/health").then((r) => r.json()),
      fetch("/api/live/status").then((r) => r.json()),
    ])
      .then(([healthData, statusData]) => {
        if (active) setHealth({ health: healthData, status: statusData });
      })
      .catch((e) => console.error("Analytics fetch error:", e));
    return () => {
      active = false;
    };
  }, []);

  // Draw chart on mount and when timeRange changes
  useEffect(() => {
    const canvas = chartRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.offsetWidth * dpr;
    canvas.height = canvas.offsetHeight * dpr;
    ctx.scale(dpr, dpr);

    // Clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;
    const padding = { top: 20, right: 20, bottom: 40, left: 60 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    // Generate mock data
    const days = timeRange === "1D" ? 24 : timeRange === "1W" ? 7 : timeRange === "1M" ? 30 : 90;
    const points = Array.from({ length: days }, (_, i) => {
      const base = Math.sin(i * 0.5) * 3 + Math.cos(i * 0.3) * 2;
      const noise = (Math.random() - 0.5) * 1.5;
      return {
        x: i,
        y1: -(base + noise + i * 0.15), // Sensor A1
        y2: -(base * 1.5 + noise * 1.2 + i * 0.25), // Sensor B2
      };
    });

    // Scales
    const minY = Math.min(...points.flatMap(p => [p.y1, p.y2])) - 2;
    const maxY = Math.max(...points.flatMap(p => [p.y1, p.y2])) + 2;
    const yScale = (y) => padding.top + plotHeight * (1 - (y - minY) / (maxY - minY));
    const xScale = (i) => padding.left + (plotWidth / (days - 1)) * i;

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
      ctx.fillText(val.toFixed(0) + "mm", padding.left - 10, y);
    });

    // Warning threshold line
    const warnY = yScale(-5);
    ctx.strokeStyle = "rgba(255,178,183,0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padding.left, warnY);
    ctx.lineTo(width - padding.right, warnY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw areas and lines
    const drawSeries = (points, color, areaColor) => {
      // Area
      const gradient = ctx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
      gradient.addColorStop(0, areaColor + "33");
      gradient.addColorStop(1, areaColor + "00");
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.moveTo(xScale(0), yScale(points[0].y));
      points.forEach((p, i) => ctx.lineTo(xScale(i), yScale(p.y)));
      ctx.lineTo(xScale(days - 1), height - padding.bottom);
      ctx.lineTo(padding.left, height - padding.bottom);
      ctx.closePath();
      ctx.fill();

      // Line
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(xScale(0), yScale(points[0].y));
      points.forEach((p, i) => ctx.lineTo(xScale(i), yScale(p.y)));
      ctx.stroke();
    };

    drawSeries(points.map(p => ({ x: p.x, y: p.y1 })), "#4edea3", "#4edea3");
    drawSeries(points.map(p => ({ x: p.x, y: p.y2 })), "#adc6ff", "#adc6ff");

    // Active point indicator (last point of B2)
    const last = points[points.length - 1];
    const lx = xScale(last.x);
    const ly = yScale(last.y2);
    ctx.fillStyle = "#020617";
    ctx.strokeStyle = "#adc6ff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(lx, ly, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Vertical line
    ctx.strokeStyle = "rgba(173,198,255,0.5)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(lx, padding.top);
    ctx.lineTo(lx, height - padding.bottom);
    ctx.stroke();
    ctx.setLineDash([]);

    // X-axis labels
    ctx.fillStyle = "rgba(194,198,214,0.6)";
    ctx.font = "10px Geist, Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const labels = timeRange === "1D"
      ? ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"]
      : timeRange === "1W"
        ? ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        : timeRange === "1M"
          ? ["Week 1", "Week 2", "Week 3", "Week 4"]
          : ["Month 1", "Month 2", "Month 3"];
    labels.forEach((label, i) => {
      const x = padding.left + (plotWidth / (labels.length - 1)) * i;
      ctx.fillText(label, x, height - padding.bottom + 8);
    });
  }, [timeRange]);

  return (
    <div className="page-container analytics-page">
      {/* Page Header */}
      <div className="page-header analytics-header">
        <div>
          <h2 className="page-main-title">Analytics</h2>
          <p className="page-subtitle">Comprehensive subsidence modeling and historical trends.</p>
        </div>
        <div className="header-actions">
          <div className="segmented-control">
            {["1D", "1W", "1M", "YTD"].map((r) => (
              <button
                key={r}
                className={`segment ${timeRange === r ? "active" : ""}`}
                onClick={() => setTimeRange(r)}
              >
                {r}
              </button>
            ))}
          </div>
          <button className="btn-secondary icon-btn-lg">
            <span className="material-symbols-outlined">calendar_month</span>
            Custom Range
          </button>
          <button className="btn-secondary icon-btn-lg outline">
            <span className="material-symbols-outlined">download</span>
            Export Data
          </button>
        </div>
      </div>

      <div className="bento-grid">
        {/* Main Trend Chart */}
        <div className="glass-panel chart-panel">
          <div className="panel-header">
            <div>
              <h3 className="panel-title">Deformation Trends</h3>
              <p className="panel-subtitle">Vertical displacement over selected period</p>
            </div>
            <div className="series-legend">
              <span className="legend-item secondary">
                <span className="legend-dot" />
                Sensor A1
              </span>
              <span className="legend-item primary">
                <span className="legend-dot" />
                Sensor B2
              </span>
            </div>
          </div>
          <div className="chart-wrapper">
            <canvas ref={chartRef} className="chart-canvas" />
            {/* Tooltip overlay */}
            <div className="chart-tooltip">
              <p className="tooltip-date">2023-10-27 14:00</p>
              <p className="tooltip-value"><span className="text-primary">B2:</span> -11.4mm</p>
              <p className="tooltip-warning">Threshold Exceeded</p>
            </div>
            <div className="x-axis-labels">
              {(timeRange === "1D" ? ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"] :
               timeRange === "1W" ? ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] :
               timeRange === "1M" ? ["Week 1", "Week 2", "Week 3", "Week 4"] :
               ["Month 1", "Month 2", "Month 3"]).map((l, i) => (
                <span key={i} style={{ left: `${(100 / 6) * i}%` }}>{l}</span>
              ))}
            </div>
          </div>
        </div>

        {/* Risk Zones Heatmap */}
        <div className="glass-panel heatmap-panel">
          <div className="panel-header">
            <h3 className="panel-title">Subsidence Risk Map</h3>
            <p className="panel-subtitle">Sector 7G Topology overlay</p>
          </div>
          <div className="heatmap-canvas">
            <div className="heatmap-bg" />
            <div className="heatmap-grid" />
            <div className="heatmap-blob warning" style={{ top: "25%", left: "30%" }} />
            <div className="heatmap-blob critical" style={{ top: "28%", left: "38%" }} />
            <div className="heatmap-blob primary" style={{ bottom: "20%", right: "20%" }} />
            <div className="risk-marker critical" style={{ top: "28%", left: "38%" }}>
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
            <span className={`metric-tag ${health?.status?.connected ? "secondary" : "critical"}`}>
              {health?.status?.connected ? "CONNECTED" : "OFFLINE"}
            </span>
          </div>
          <p className="metric-label">Live ESP32 Status</p>
          <p className="metric-big glow">{health?.status?.buffered_days ?? 0} <span className="metric-unit">/ 30 days</span></p>
        </div>

        <div className="glass-panel metric-panel warning">
          <div className="metric-header">
            <span className="material-symbols-outlined text-tertiary">database</span>
            <span className={`metric-tag ${health?.status?.buffered_days >= 30 ? "secondary" : "critical"}`}>
              {health?.status?.buffered_days >= 30 ? "READY" : "BUFFERING"}
            </span>
          </div>
          <p className="metric-label">Buffered Readings</p>
          <p className="metric-big">{health?.status?.readings_count ?? 0}</p>
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
                <span className="diag-value">{health?.status?.buffered_days ?? 0}</span>
              </div>
              <div className="diag-bar">
                <div className="diag-fill neutral" style={{ width: `${Math.min(100, ((health?.status?.buffered_days ?? 0) / 30) * 100)}%` }} />
              </div>
            </div>
            <div className="diagnostic">
              <div className="diag-header">
                <span>Predictions Available</span>
                <span className="diag-value tertiary">
                  {health?.status?.has_predictions
                    ? Object.entries(health.status.has_predictions).filter(([, v]) => v).length
                    : 0}
                </span>
              </div>
              <div className="diag-bar">
                <div className="diag-fill tertiary" style={{ width: `${((health?.status?.has_predictions ? Object.entries(health.status.has_predictions).filter(([, v]) => v).length : 0) / 2) * 100}%` }} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}