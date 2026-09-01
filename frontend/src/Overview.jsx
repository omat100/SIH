import { useState } from "react";
import ReadingForm from "./ReadingForm";
import EspLivePanel from "./EspLivePanel";
import { useLiveData } from "./useLiveData";

export default function Overview() {
  const { hasAnyData, readingsCount, bufferedDays, lastPrediction } = useLiveData();
  const [model, setModel] = useState("gbdt");
  const [readingsCountForm, setReadingsCountForm] = useState(0);
  const [showResult, setShowResult] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = async (readings) => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetch(`/api/manual/${model}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ readings }),
      });
      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.error || "Prediction failed");
      }
      const result = await resp.json();
      setPrediction(result);
      setShowResult(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const riskColor = (cls) => ({
    safe: "#22c55e",
    watch: "#f59e0b",
    critical: "#ef4444",
  }[cls] || "#6b7280");

  return (
    <div className="page-container">
      <div className="page-header">
        <div className="page-header-content">
          <h3 className="page-title">
            <span className="material-symbols-outlined page-icon">memory</span>
            Run Inference
          </h3>
        </div>
      </div>

      <div className="inference-canvas glass-panel">
        {/* Model Selection */}
        <div className="form-section">
          <label className="form-label">Active Model</label>
          <div className="select-wrapper">
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="form-select"
            >
              <option value="gbdt">GBDT</option>
              <option value="torch">LSTM</option>
            </select>
            <span className="select-icon material-symbols-outlined">expand_more</span>
          </div>
        </div>

        {/* Sensor Readings Form */}
        <div className="form-section">
          <div className="section-header">
            <h4 className="section-title">Sensor Readings</h4>
            <div className="readings-badge">
              <span className="material-symbols-outlined">database</span>
              Readings: {readingsCountForm} / 30
            </div>
          </div>

          <ReadingForm
            model={model}
            loading={loading}
            onPredict={handlePredict}
            onCountChange={setReadingsCountForm}
          />

          {hasAnyData && (
            <p className="hint">
              Live backend buffer: {readingsCount} readings across {bufferedDays} days. Latest risk:{" "}
              {lastPrediction ? lastPrediction.risk_class.toUpperCase() : "no AI result yet"}.
            </p>
          )}
        </div>
      </div>

      {error && (
        <div className="error-banner" role="alert">
          <span className="material-symbols-outlined">error</span>
          {error}
        </div>
      )}

      {/* Results Section */}
      {showResult && prediction && (
        <div className="results-panel glass-panel">
          <div className="results-header">
            <h3 className="results-title">Prediction Result</h3>
          </div>
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
            <h4 className="section-title">Class Probabilities</h4>
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
              <h4 className="section-title">Key Signals</h4>
              <table className="signals-table">
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
      )}

      {/* Live Monitoring Panel */}
      <EspLivePanel />
    </div>
  );
}