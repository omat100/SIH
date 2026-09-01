import { useState, useEffect } from "react";

export default function ModelTuning() {
  const [hyperparams, setHyperparams] = useState({
    learning_rate: 0.05,
    max_iter: 600,
    max_depth: 10,
    n_estimators: 200,
    min_samples_split: 5,
  });

  const [health, setHealth] = useState(null);

  useEffect(() => {
    let active = true;
    fetch("/api/manual/health")
      .then((r) => r.json())
      .then((data) => {
        if (active) setHealth(data);
      })
      .catch((e) => console.error("Health fetch error:", e));
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="page-container">
      <div className="page-header">
        <div>
          <h2 className="page-main-title">Model Tuning</h2>
          <p className="page-subtitle">Adjust hyperparameters and retrain models.</p>
        </div>
        <button className="btn-secondary" onClick={() => window.location.reload()}>
          <span className="material-symbols-outlined">refresh</span>
          Refresh Model Status
        </button>
      </div>

      <div className="tuning-grid">
        {/* Hyperparameters Panel */}
        <div className="glass-panel">
          <h3 className="panel-title">Hyperparameters</h3>
          <p className="panel-subtitle">Tune and retrain on the backend CLI (python -m minesub train).</p>
          <div className="params-form">
            {Object.entries(hyperparams).map(([key, value]) => (
              <div key={key} className="param-row">
                <label className="param-label">{key.replace(/_/g, " ")}</label>
                <input
                  type="number"
                  step={key.includes("rate") ? 0.01 : 1}
                  value={value}
                  onChange={(e) => setHyperparams({ ...hyperparams, [key]: parseFloat(e.target.value) })}
                  className="form-input"
                />
              </div>
            ))}
          </div>
        </div>

        {/* Model Status */}
        <div className="glass-panel">
          <h3 className="panel-title">Registered Models</h3>
          <div className="history-list">
            {health?.models ? (
              Object.entries(health.models).map(([name, status]) => (
                <div key={name} className="history-item">
                  <span className="history-meta">
                    {name === "gbdt" ? "GBDT" : name === "torch" ? "LSTM" : name}
                  </span>
                  <span className={`badge ${status === "loaded" ? "primary" : "warning"}`}>
                    {String(status).toUpperCase()}
                  </span>
                </div>
              ))
            ) : (
              <p className="hint">Loading model status from /api/manual/health...</p>
            )}
          </div>
        </div>

        {/* Current Status */}
        {health && (
          <div className="glass-panel wide">
            <h3 className="panel-title">Model Backend Status</h3>
            <div className="metrics-display">
              <div className="metric-big">
                <span className="metric-label">Config</span>
                <span className="metric-value" style={{ fontSize: "14px" }}>{health.config}</span>
              </div>
              <div className="metric-big">
                <span className="metric-label">Models Loaded</span>
                <span className="metric-value">
                  {health.models ? Object.values(health.models).filter((s) => s === "loaded").length : 0} / {health.models ? Object.keys(health.models).length : 0}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}