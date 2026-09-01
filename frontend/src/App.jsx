import { Routes, Route } from "react-router-dom";
import Layout from "./Layout";
import Overview from "./Overview";
import Realtime from "./Realtime";
import Analytics from "./Analytics";
import ModelTuning from "./ModelTuning";
import Logs from "./Logs";

function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Overview />} />
        <Route path="/realtime" element={<Realtime />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/model-tuning" element={<ModelTuning />} />
        <Route path="/logs" element={<Logs />} />
      </Route>
    </Routes>
  );
}

export default App;