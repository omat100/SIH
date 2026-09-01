import { useState, useEffect, useRef } from "react";
import { useLiveData } from "./useLiveData";

const LOG_LEVELS = ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"];

export default function Logs() {
  const [logs, setLogs] = useState([]);
  const [filterLevel, setFilterLevel] = useState("ALL");
  const [search, setSearch] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);
  const {
    connected,
    latestReading,
    readingsCount,
    predictions,
  } = useLiveData();
  const prevReadingDate = useRef(null);
  const prevPrediction = useRef(null);

  // Append a real log entry whenever a new reading or prediction arrives, or
  // when the connection state changes. Uses an interval so state changes are
  // observed without triggering cascading renders from the effect body.
  useEffect(() => {
    const tick = () => {
      const time = new Date().toLocaleTimeString("en-US", { hour12: false });
      const entries = [];

      if (latestReading && latestReading.date !== prevReadingDate.current) {
        prevReadingDate.current = latestReading.date;
        entries.push({
          level: "DEBUG",
          timestamp: time,
          source: "sensor",
          message: `Reading: temp=${latestReading.temp_c?.toFixed?.(1)}C hum=${latestReading.humidity_pct?.toFixed?.(1)}% tilt=${latestReading.tilt_deg?.toFixed?.(4)}deg dist=${latestReading.distance_mm?.toFixed?.(1)}mm (${readingsCount ?? 0} total)`,
        });
      }

      Object.entries(predictions || {}).forEach(([name, pred]) => {
        if (!pred || pred.as_of === prevPrediction.current?.[name]) return;
        prevPrediction.current = { ...(prevPrediction.current || {}), [name]: pred.as_of };
        entries.push({
          level:
            pred.risk_class === "critical"
              ? "CRITICAL"
              : pred.risk_class === "watch"
                ? "WARN"
                : "INFO",
          timestamp: time,
          source: name,
          message: `Prediction: ${pred.risk_class.toUpperCase()} (p=${(pred.probabilities?.[pred.risk_class] ?? 0).toFixed(2)}) | horizon: ${pred.horizon_days}d`,
        });
      });

      if (entries.length > 0) {
        setLogs((prev) => [...entries, ...prev].slice(0, 300));
      }
    };

    const interval = setInterval(tick, 1500);
    return () => clearInterval(interval);
  }, [connected, latestReading, predictions, readingsCount]);

  const filteredLogs = logs.filter((log) => {
    if (filterLevel !== "ALL" && log.level !== filterLevel) return false;
    if (search && !log.message.toLowerCase().includes(search.toLowerCase()) && !log.source.includes(search)) return false;
    return true;
  });

  const getLevelStyle = (level) => {
    const styles = {
      DEBUG: "text-outline",
      INFO: "text-primary",
      WARN: "text-warning",
      ERROR: "text-tertiary",
      CRITICAL: "text-error font-bold",
    };
    return styles[level] || "";
  };

  const getLevelBg = (level) => {
    const styles = {
      DEBUG: "bg-white/5",
      INFO: "bg-primary/10",
      WARN: "bg-warning/10",
      ERROR: "bg-tertiary/10",
      CRITICAL: "bg-error/10",
    };
    return styles[level] || "";
  };

  return (
    <div className="page-container logs-page">
      <div className="page-header">
        <div>
          <h2 className="page-main-title">System Logs</h2>
          <p className="page-subtitle">Real-time system events and diagnostic information.</p>
        </div>
        <div className="header-actions">
          <div className="segmented-control">
            {["ALL", ...LOG_LEVELS].map((l) => (
              <button
                key={l}
                className={`segment ${filterLevel === l ? "active" : ""}`}
                onClick={() => setFilterLevel(l)}
              >
                {l}
              </button>
            ))}
          </div>
          <div className="search-input">
            <span className="material-symbols-outlined">search</span>
            <input
              type="text"
              placeholder="Filter logs..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="form-input"
            />
          </div>
          <label className="toggle-switch">
            <input
              type="checkbox"
              checked={autoScroll}
              onChange={(e) => setAutoScroll(e.target.checked)}
            />
            <span className="toggle-slider" />
            <span className="toggle-label">Auto-scroll</span>
          </label>
          <button className="btn-secondary" onClick={() => setLogs([])}>
            <span className="material-symbols-outlined">clear_all</span>
            Clear
          </button>
        </div>
      </div>

      <div className="glass-panel log-panel">
        <div className="log-table">
          <div className="log-header">
            <div className="col time">Time</div>
            <div className="col level">Level</div>
            <div className="col source">Source</div>
            <div className="col message">Message</div>
          </div>
          <div className="log-body">
            {filteredLogs.length === 0 ? (
              <div className="empty-state">
                <span className="material-symbols-outlined empty-icon">terminal</span>
                <h3>No Data Fetched</h3>
                <p>No system events yet — waiting for data from the backend sensor on COM8.</p>
              </div>
            ) : (
              filteredLogs.map((log, i) => (
                <div key={i} className={`log-row ${getLevelBg(log.level)}`}>
                  <div className="col time">{log.timestamp}</div>
                  <div className="col level">
                    <span className={`level-badge ${getLevelStyle(log.level)}`}>{log.level}</span>
                  </div>
                  <div className="col source">{log.source}</div>
                  <div className="col message">{log.message}</div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}