import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import App from "./App";
import FleetOverview from "./pages/FleetOverview";
import MachineDetailPage from "./pages/MachineDetail";
import TopologyView from "./pages/TopologyView";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />}>
          <Route index element={<Navigate to="/topology" replace />} />
          <Route path="topology" element={<TopologyView />} />
          <Route path="fleet" element={<FleetOverview />} />
          <Route path="machines/:id" element={<MachineDetailPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
