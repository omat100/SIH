import { useContext } from "react";
import { LiveDataContext } from "./liveDataContext";

export function useLiveData() {
  const ctx = useContext(LiveDataContext);
  if (ctx == null) {
    throw new Error("useLiveData must be used within a LiveDataProvider");
  }
  return ctx;
}
