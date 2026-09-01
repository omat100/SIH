import { useState } from "react";
import { useLiveData } from "./useLiveData";

export default function EspLivePanel() {
  const { connected, status, predictions, readingsCount, bufferedDays, hasAnyData } =
    useLiveData();
  const [selectedModel, setSelectedModel] = useState("gbdt");

  const prediction = predictions[selectedModel] || null;

  const riskColor = (cls) => ({
    safe: "#22c55e",
    watch: "#f59e0b",
    critical: "#ef4444",
  }[cls] || "#6b7280");

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
          <span className={`status-dot ${connected ? "connected" : "disconnected"}`} />
          <span>{connected ? "Connected" : "Disconnected"}</span>
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
          <span className="meta-value">{bufferedDays || 0} / 30</span>
        </div>
        <div className="meta-item">
          <span className="meta-label">Total Readings</span>
          <span className="meta-value">{readingsCount || 0}</span>
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

      {!hasAnyData ? (
        <div className="live-empty">
          No data fetched — waiting for the AI result from the backend.
        </div>
      ) : prediction ? (
        <div className="live-prediction">
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
          {bufferedDays >= 30
            ? "Waiting for inference..."
            : `Collecting data... ${bufferedDays || 0}/30 days buffered`}
        </div>
      )}

      <div className="live-actions">
        <button onClick={() => fetch("/api/live/trigger", { method: "POST" })} className="btn-secondary">
          Trigger Inference Now
        </button>
        <a href="/api/live/readings" target="_blank" className="btn-secondary link-btn">
          View Raw Readings
        </a>
      </div>
    </section>
  );
}
