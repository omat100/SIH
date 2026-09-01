import { useState, useEffect } from "react";

const LOG_LEVELS = ["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"];
const MOCK_LOGS = [
  { level: "INFO", timestamp: "10:42:15.234", source: "serial", message: "ESP32 connected on COM7 @ 115200" },
  { level: "INFO", timestamp: "10:42:10.112", source: "inference", message: "Auto-inference triggered (30 days buffered)" },
  { level: "INFO", timestamp: "10:42:05.001", source: "gbdt", message: "Prediction: WATCH (p=0.62) | horizon: 21d" },
  { level: "INFO", timestamp: "10:42:05.002", source: "torch", message: "Prediction: SAFE (p=0.46) | horizon: 21d" },
  { level: "WARN", timestamp: "10:41:58.901", source: "sensor/SN-441-B", message: "Tilt anomaly detected: +0.045°" },
  { level: "INFO", timestamp: "10:41:55.334", source: "serial", message: "Received reading from SN-441-B" },
  { level: "DEBUG", timestamp: "10:41:50.123", source: "pipeline", message: "Daily buffer: 28 days, 847 readings" },
  { level: "ERROR", timestamp: "10:40:12.445", source: "serial", message: "Connection lost, retrying in 5s..." },
  { level: "INFO", timestamp: "10:40:17.001", source: "serial", message: "Reconnected successfully" },
  { level: "CRITICAL", timestamp: "10:39:00.000", source: "alert", message: "Subsidence threshold exceeded at Sensor B2" },
];

export default function Logs() {
  const [logs, setLogs] = useState(MOCK_LOGS);
  const [filterLevel, setFilterLevel] = useState("ALL");
  const [search, setSearch] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);

  // Pull real live status + model health and prepend as log events
  useEffect(() => {
    let active = true;
    const prependReal = async () => {
      try {
        const [statusRes, healthRes] = await Promise.all([
          fetch("/api/live/status"),
          fetch("/api/manual/health"),
        ]);
        const status = await statusRes.json();
        const health = await healthRes.json();
        if (!active) return;
        const realLogs = [];
        realLogs.push({
          level: status?.connected ? "INFO" : "ERROR",
          timestamp: new Date().toLocaleTimeString("en-US", { hour12: false }),
          source: "serial",
          message: status?.connected
            ? `ESP32 connected (${status.buffered_days ?? 0}/30 days buffered)`
            : `ESP32 disconnected: ${status?.last_error || "no connection"}`,
        });
        if (health?.models) {
          Object.entries(health.models).forEach(([name, st]) => {
            realLogs.push({
              level: st === "loaded" ? "INFO" : "WARN",
              timestamp: new Date().toLocaleTimeString("en-US", { hour12: false }),
              source: name,
              message: `Model ${name === "torch" ? "LSTM" : name} ${st}`,
            });
          });
        }
        setLogs((prev) => [...realLogs, ...prev]);
      } catch (e) {
        console.error("Logs live fetch error:", e);
      }
    };
    prependReal();
    return () => {
      active = false;
    };
  }, []);

  // Simulate live logs
  useEffect(() => {
    const interval = setInterval(() => {
      if (!autoScroll) return;
      const newLog = {
        level: LOG_LEVELS[Math.floor(Math.random() * LOG_LEVELS.length)],
        timestamp: new Date().toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit", fractionalSecondDigits: 3 }),
        source: ["serial", "inference", "gbdt", "torch", "sensor/SN-892-A", "sensor/SN-441-B"][Math.floor(Math.random() * 6)],
        message: "System heartbeat",
      };
      setLogs((prev) => [newLog, ...prev.slice(0, 999)]);
    }, 3000);
    return () => clearInterval(interval);
  }, [autoScroll]);

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
            {filteredLogs.map((log, i) => (
              <div key={i} className={`log-row ${getLevelBg(log.level)}`}>
                <div className="col time">{log.timestamp}</div>
                <div className="col level">
                  <span className={`level-badge ${getLevelStyle(log.level)}`}>{log.level}</span>
                </div>
                <div className="col source">{log.source}</div>
                <div className="col message">{log.message}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}