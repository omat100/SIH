import { useEffect, useState } from "react";

const POLL_INTERVAL = 3000;

export default function EspLivePanel() {
  const [status, setStatus] = useState(null);
  const [latest, setLatest] = useState({ gbdt: null, torch: null });
  const [selectedModel, setSelectedModel] = useState("gbdt");

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [statusRes, latestRes] = await Promise.all([
          fetch("/api/live/status"),
          fetch("/api/live/latest"),
        ]);
        const statusData = await statusRes.json();
        const latestData = await latestRes.json();
        setStatus(statusData);
        setLatest(latestData);
      } catch (e) {
        console.error("Live panel fetch error:", e);
      }
    };

    fetchAll();
    const interval = setInterval(fetchAll, POLL_INTERVAL);
    return () => clearInterval(interval);
  }, []);

  const prediction = latest[selectedModel];

  const riskColor = (cls) => ({
    safe: "#22c55e",
    watch: "#f59e0b",
    critical: "#ef4444",
  }[cls] || "#6b7280");

  // regression payload: colour by the fixed geotech band when present,
  // otherwise by the recall-tuned alert flag
  const alertColor = (alert, band) =>
    band ? riskColor(band) : { elevated: "#f59e0b", normal: "#22c55e" }[alert] || "#6b7280";

  const isRegression = (p) =>
    p?.task === "regression" || p?.predicted_excess_rate_mm_day != null;

  const formatTime = (iso) => {
    if (!iso) return "—";
    try {
      return new Date(iso).toLocaleTimeString();
    } catch {
      return iso;
    }
  };

  return (
    <section className="panel live-panel">
      <div className="live-header">
        <h2>Live ESP32 Monitoring</h2>
        <div className="live-status">
          <span className={`status-dot ${status?.connected ? "connected" : "disconnected"}`} />
          <span>{status?.connected ? "Connected" : "Disconnected"}</span>
          {status?.last_error && (
            <span className="status-error" title={status.last_error}>
              ⚠ {status.last_error.substring(0, 50)}
            </span>
          )}
        </div>
      </div>

      <div className="live-meta">
        <div className="meta-item">
          <span className="meta-label">Buffered Days</span>
          <span className="meta-value">{status?.buffered_days || 0} / 30</span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Total Readings</span>
          <span className="meta-value">{status?.readings_count || 0}</span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Last Prediction</span>
          <span className="meta-value">{formatTime(status?.last_prediction_ts)}</span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Model</span>
          <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} className="model-select">
            <option value="gbdt">GBDT</option>
            <option value="torch">LSTM</option>
          </select>
        </div>
      </div>

      {prediction ? (
        <div className="live-prediction">
          {isRegression(prediction) ? (
            <>
              <div
                className="risk-card"
                style={{ borderColor: alertColor(prediction.alert, prediction.geotech_band) }}
              >
                <div
                  className="risk-class"
                  style={{ background: alertColor(prediction.alert, prediction.geotech_band) }}
                >
                  {(prediction.alert || prediction.geotech_band || "prediction").toUpperCase()}
                </div>
                <div className="meta">
                  <span>Horizon: {prediction.horizon_days} days</span>
                  <span>Model: {prediction.model}</span>
                  <span>As of: {prediction.as_of}</span>
                </div>
              </div>

              <div className="probabilities">
                <h3>Forecast</h3>
                <div className="prob-bar">
                  <span className="prob-label">Excess settlement rate</span>
                  <span className="prob-value">
                    {prediction.predicted_excess_rate_mm_day?.toFixed(4)} mm/day
                  </span>
                </div>
                {prediction.operating_threshold_mm_day != null && (
                  <div className="prob-bar">
                    <span className="prob-label">Alert threshold</span>
                    <span className="prob-value">
                      {prediction.operating_threshold_mm_day.toFixed(4)} mm/day
                    </span>
                  </div>
                )}
                {prediction.geotech_band && (
                  <div className="prob-bar">
                    <span className="prob-label">Geotech band</span>
                    <span className="prob-value">{prediction.geotech_band}</span>
                  </div>
                )}
              </div>
            </>
          ) : (
            <>
              <div className="risk-card" style={{ borderColor: riskColor(prediction.risk_class) }}>
                <div className="risk-class" style={{ background: riskColor(prediction.risk_class) }}>
                  {prediction.risk_class.toUpperCase()}
                </div>
                <div className="meta">
                  <span>Horizon: {prediction.horizon_days} days</span>
                  <span>Model: {prediction.model}</span>
                  <span>As of: {prediction.as_of}</span>
                </div>
              </div>

              <div className="probabilities">
                <h3>Class Probabilities</h3>
                {Object.entries(prediction.probabilities).map(([cls, prob]) => (
                  <div key={cls} className="prob-bar">
                    <span className="prob-label">{cls}</span>
                    <div className="prob-track">
                      <div
                        className="prob-fill"
                        style={{
                          width: `${prob * 100}%`,
                          background: riskColor(cls),
                        }}
                      />
                    </div>
                    <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </>
          )}

          {prediction.signals && Object.keys(prediction.signals).length > 0 && (
            <div className="signals">
              <h3>Key Signals</h3>
              <table>
                <thead>
                  <tr>
                    <th>Signal</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(prediction.signals).map(([k, v]) => (
                    <tr key={k}>
                      <td>{k.replace(/_/g, " ")}</td>
                      <td>{typeof v === "number" ? v.toFixed(4) : v}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ) : (
        <div className="live-empty">
          {status?.buffered_days >= 30
            ? "Waiting for inference..."
            : `Collecting data... ${status?.buffered_days || 0}/30 days buffered`}
        </div>
      )}

      <div className="live-actions">
        <button onClick={() => fetch("/api/live/trigger", { method: "POST" })} className="btn-secondary">
          Trigger Inference Now
        </button>
        <a href="/api/live/buffer" target="_blank" className="btn-secondary link-btn">
          View Buffer Debug
        </a>
      </div>
    </section>
  );
}