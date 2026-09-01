import { Outlet, NavLink, useLocation } from "react-router-dom";

const navItems = [
  { path: "/", label: "Overview", icon: "dashboard" },
  { path: "/realtime", label: "Real-time Feed", icon: "dynamic_feed" },
  { path: "/analytics", label: "Analytics", icon: "analytics" },
  { path: "/model-tuning", label: "Model Tuning", icon: "precision_manufacturing" },
  { path: "/logs", label: "System Logs", icon: "terminal" },
];

export default function Layout() {
  const location = useLocation();

  return (
    <div className="layout">
      {/* Side Nav */}
      <nav className="sidenav" role="navigation" aria-label="Main navigation">
        <div className="sidenav-header">
          <div className="sidenav-logo">
            <span className="material-symbols-outlined">sensors</span>
          </div>
          <div>
            <h1 className="sidenav-title">Subsidence Control</h1>
            <p className="sidenav-subtitle">Vigilant Monitoring</p>
          </div>
        </div>

        <ul className="sidenav-menu">
          {navItems.map((item) => (
            <li key={item.path}>
              <NavLink
                to={item.path}
                className={({ isActive }) =>
                  `sidenav-link ${isActive ? "active" : ""}`
                }
              >
                <span className="material-symbols-outlined">{item.icon}</span>
                <span>{item.label}</span>
              </NavLink>
            </li>
          ))}
        </ul>

        <div className="sidenav-footer">
          <NavLink to="/support" className="sidenav-link">
            <span className="material-symbols-outlined">help</span>
            <span>Support</span>
          </NavLink>
          <NavLink to="/alerts" className="sidenav-link alert">
            <span className="material-symbols-outlined">warning</span>
            <span>Alert Status</span>
          </NavLink>
          <button className="sidenav-cta">
            <span className="material-symbols-outlined">play_arrow</span>
            Run Prediction
          </button>
        </div>
      </nav>

      {/* Main Content */}
      <main className="main-content">
        {/* Top Bar */}
        <header className="topbar">
          <div className="topbar-title">
            <h2>Mine Subsidence Early Warning</h2>
          </div>
          <div className="topbar-status">
            <span className="model-badge">
              <span className="status-dot" />
              Model: GBDT
            </span>
            <span className="divider" />
            <span className="version-badge">Torch: 2.1.0</span>
          </div>
        </header>

        {/* Page Content */}
        <Outlet />
      </main>
    </div>
  );
}