import { useState, useMemo } from "react";

const REQUIRED_FIELDS = [
  { key: "date", label: "Date", type: "date" },
  { key: "temp_c", label: "Temp (°C)", type: "number", step: 0.1, min: -40, max: 60 },
  { key: "humidity_pct", label: "Humidity (%)", type: "number", step: 0.1, min: 0, max: 100 },
  { key: "tilt_deg", label: "Tilt (deg)", type: "number", step: 0.0001, min: 0, max: 90 },
  { key: "distance_mm", label: "Distance (mm)", type: "number", step: 0.1, min: 0, max: 10000 },
];

export default function ReadingForm({ onPredict, model, loading }) {
  const [readings, setReadings] = useState([]);
  const [inputValues, setInputValues] = useState(() => getDefaults());

  function getDefaults() {
    const today = new Date();
    today.setDate(today.getDate() - 29);
    return {
      date: today.toISOString().split("T")[0],
      temp_c: "",
      humidity_pct: "",
      tilt_deg: "",
      distance_mm: "",
    };
  }

  function handleChange(e) {
    const { name, value } = e.target;
    setInputValues((prev) => ({ ...prev, [name]: value }));
  }

  function addReading() {
    const missing = REQUIRED_FIELDS.filter((f) => !inputValues[f.key] && inputValues[f.key] !== 0);
    if (missing.length) {
      alert(`Please fill all fields: ${missing.map((m) => m.label).join(", ")}`);
      return;
    }

    const newReading = {
      date: inputValues.date,
      temp_c: parseFloat(inputValues.temp_c),
      humidity_pct: parseFloat(inputValues.humidity_pct),
      tilt_deg: parseFloat(inputValues.tilt_deg),
      distance_mm: parseFloat(inputValues.distance_mm),
    };

    const nextDate = new Date(inputValues.date);
    nextDate.setDate(nextDate.getDate() + 1);

    setReadings((prev) => [...prev, newReading]);
    setInputValues({
      date: nextDate.toISOString().split("T")[0],
      temp_c: "",
      humidity_pct: "",
      tilt_deg: "",
      distance_mm: "",
    });
  }

  function removeReading(index) {
    setReadings((prev) => prev.filter((_, i) => i !== index));
  }

  function clearAll() {
    setReadings([]);
    setInputValues(getDefaults());
  }

  function loadSample() {
    const sample = [];
    const baseDate = new Date("2024-01-01");
    for (let i = 0; i < 35; i++) {
      const d = new Date(baseDate);
      d.setDate(d.getDate() + i);
      sample.push({
        date: d.toISOString().split("T")[0],
        temp_c: 18 + Math.sin(i / 5) * 6 + (Math.random() - 0.5) * 2,
        humidity_pct: 55 + Math.cos(i / 5) * 5 + (Math.random() - 0.5) * 5,
        tilt_deg: 0.002 * i + 0.00002 * i * i + (Math.random() - 0.5) * 0.001,
        distance_mm: 0.4 * i + 0.01 * i * i + (Math.random() - 0.5) * 0.5,
      });
    }
    setReadings(sample);
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (readings.length < 30) return;
    onPredict(readings);
  }

  const canSubmit = readings.length >= 30 && !loading;

  return (
    <section className="panel reading-form">
      <header className="form-header">
        <h2>Add Readings</h2>
        <span className="count-badge">Readings: {readings.length} / 30</span>
      </header>

      <form className="input-row" onSubmit={(e) => { e.preventDefault(); addReading(); }}>
        {REQUIRED_FIELDS.map((field) => (
          <div key={field.key} className="input-group">
            <label htmlFor={field.key}>{field.label}</label>
            <input
              id={field.key}
              name={field.key}
              type={field.type}
              value={inputValues[field.key] ?? ""}
              onChange={handleChange}
              step={field.step}
              min={field.min}
              max={field.max}
              required
              placeholder={field.label}
            />
          </div>
        ))}
        <button type="button" onClick={addReading} className="btn-add" disabled={loading}>
          Add Reading
        </button>
      </form>

      {readings.length > 0 && (
        <div className="readings-table-wrapper">
          <table className="readings-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Date</th>
                <th>Temp (°C)</th>
                <th>Humidity (%)</th>
                <th>Tilt (deg)</th>
                <th>Distance (mm)</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {readings.map((r, i) => (
                <tr key={i}>
                  <td>{i + 1}</td>
                  <td>{r.date}</td>
                  <td>{r.temp_c.toFixed(1)}</td>
                  <td>{r.humidity_pct.toFixed(1)}</td>
                  <td>{r.tilt_deg.toFixed(4)}</td>
                  <td>{r.distance_mm.toFixed(1)}</td>
                  <td>
                    <button
                      type="button"
                      className="btn-delete"
                      onClick={() => removeReading(i)}
                      aria-label="Remove reading"
                    >
                      ✕
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="form-actions">
        <form onSubmit={handleSubmit}>
          <button
            type="submit"
            className="btn-primary"
            disabled={!canSubmit}
          >
            {loading ? "Predicting..." : "Run Prediction"}
          </button>
        </form>
        <div className="secondary-actions">
          <button type="button" className="btn-secondary" onClick={loadSample} disabled={loading}>
            Load Sample (35 readings)
          </button>
          {readings.length > 0 && (
            <button type="button" className="btn-secondary btn-danger" onClick={clearAll} disabled={loading}>
              Clear All
            </button>
          )}
        </div>
      </div>

      {readings.length > 0 && readings.length < 30 && (
        <p className="hint">Add {30 - readings.length} more reading{30 - readings.length !== 1 ? "s" : ""} to enable prediction.</p>
      )}
    </section>
  );
}