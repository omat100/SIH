import { useEffect, useRef, useState } from "react";
import { LiveDataContext } from "./liveDataContext";

const EMPTY_PREDICTIONS = { gbdt: null, torch: null };

export function LiveDataProvider({ children }) {
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState(null);
  const [latestReading, setLatestReading] = useState(null);
  const [readings, setReadings] = useState([]);
  const [readingsCount, setReadingsCount] = useState(0);
  const [bufferedDays, setBufferedDays] = useState(0);
  const [predictions, setPredictions] = useState(EMPTY_PREDICTIONS);
  const [loaded, setLoaded] = useState(false);
  const sourceRef = useRef(null);

  useEffect(() => {
    let active = true;

    async function hydrate() {
      try {
        const resp = await fetch("/api/live/readings");
        const data = await resp.json();
        if (!active) return;
        setStatus(data.status || null);
        setLatestReading(data.latest_reading || null);
        setReadingsCount(data.readings_count || 0);
        setBufferedDays(data.buffered_days || 0);
        setPredictions(data.predictions || EMPTY_PREDICTIONS);
        if (data.buffer) {
          const all = [];
          Object.entries(data.buffer).forEach(([date, dayReadings]) => {
            dayReadings.forEach((r) => all.push({ ...r, date }));
          });
          setReadings(all);
        }
      } catch (e) {
        console.error("liveData hydrate error:", e);
      } finally {
        if (active) setLoaded(true);
      }
    }

    function applyReading(reading) {
      const item = {
        date: reading.date,
        temp_c: reading.temp_c,
        humidity_pct: reading.humidity_pct,
        tilt_deg: reading.tilt_deg,
        distance_mm: reading.distance_mm,
      };
      setLatestReading(item);
      setReadings((prev) => [...prev, item]);
      if (typeof reading.readings_count === "number") {
        setReadingsCount(reading.readings_count);
      }
    }

    hydrate();

    const source = new EventSource("/api/live/stream");
    sourceRef.current = source;

    source.onopen = () => active && setConnected(true);
    source.onerror = () => active && setConnected(false);

    source.addEventListener("reading", (e) => {
      if (active) applyReading(JSON.parse(e.data));
    });
    source.addEventListener("prediction", (e) => {
      if (active) setPredictions(JSON.parse(e.data) || EMPTY_PREDICTIONS);
    });
    source.addEventListener("status", (e) => {
      if (!active) return;
      const s = JSON.parse(e.data);
      setStatus(s);
      if (typeof s.buffered_days === "number") setBufferedDays(s.buffered_days);
      if (typeof s.readings_count === "number") setReadingsCount(s.readings_count);
    });

    return () => {
      active = false;
      source.close();
      sourceRef.current = null;
    };
  }, []);

  const lastPrediction = predictions.gbdt || predictions.torch || null;
  const hasAnyData = connected && readingsCount > 0;

  const value = {
    connected,
    status,
    latestReading,
    readings,
    readingsCount,
    bufferedDays,
    predictions,
    lastPrediction,
    hasAnyData,
    loaded,
  };

  return (
    <LiveDataContext.Provider value={value}>{children}</LiveDataContext.Provider>
  );
}
