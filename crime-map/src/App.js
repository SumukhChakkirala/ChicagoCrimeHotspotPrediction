import { useState, useEffect, useCallback } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup, useMapEvents } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

const API = "http://localhost:8000";

function riskColor(score) {
  if (score > 0.7) return "#d73027";
  if (score > 0.5) return "#fc8d59";
  if (score > 0.35) return "#fee090";
  if (score > 0.2) return "#91bfdb";
  return "#4575b4";
}

function riskLabel(score) {
  if (score > 0.7) return "High";
  if (score > 0.4) return "Medium";
  return "Low";
}

function MapClickHandler({ onPredict }) {
  useMapEvents({
    click(e) {
      onPredict(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

function Legend() {
  const items = [
    { color: "#d73027", label: "High risk (>0.7)" },
    { color: "#fc8d59", label: "Med-high (0.5–0.7)" },
    { color: "#fee090", label: "Medium (0.35–0.5)" },
    { color: "#91bfdb", label: "Low-med (0.2–0.35)" },
    { color: "#4575b4", label: "Low (<0.2)" },
  ];
  return (
    <div className="legend">
      <h4>Risk Level</h4>
      {items.map((i) => (
        <div key={i.label} className="legend-row">
          <span className="legend-dot" style={{ background: i.color }} />
          {i.label}
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [nodes, setNodes]               = useState([]);
  const [crimeTypes, setCrimeTypes]     = useState([]);
  const [stats, setStats]               = useState(null);
  const [selectedType, setSelectedType] = useState("ALL");
  const [prediction, setPrediction]     = useState(null);
  const [loading, setLoading]           = useState(true);
  const [error, setError]               = useState(null);

  // Date + time state
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const defaultDate = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
  const defaultTime = `${pad(now.getHours())}:00`;

  const [selectedDate, setSelectedDate] = useState(defaultDate);
  const [selectedTime, setSelectedTime] = useState(defaultTime);

  // Parse date/time into hour, dow, month for the API
  const getTemporalParams = useCallback(() => {
    const dt = new Date(`${selectedDate}T${selectedTime}`);
    return {
      hour:        dt.getHours(),
      day_of_week: dt.getDay() === 0 ? 6 : dt.getDay() - 1, // 0=Mon…6=Sun
      month:       dt.getMonth() + 1,
    };
  }, [selectedDate, selectedTime]);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/crime-types`).then((r) => r.json()),
      fetch(`${API}/api/stats`).then((r) => r.json()),
    ])
      .then(([ct, st]) => {
        setCrimeTypes(["ALL", ...ct.crime_types]);
        setStats(st);
      })
      .catch(() => setError("Cannot reach API. Is the backend running on port 8000?"));
  }, []);

  useEffect(() => {
    setLoading(true);
    const url =
      selectedType === "ALL"
        ? `${API}/api/nodes`
        : `${API}/api/nodes?crime_type=${selectedType}`;

    fetch(url)
      .then((r) => r.json())
      .then((d) => { setNodes(d.nodes); setLoading(false); })
      .catch(() => { setError("Failed to load node data"); setLoading(false); });
  }, [selectedType]);

  const handlePredict = useCallback((lat, lon) => {
    const { hour, day_of_week, month } = getTemporalParams();
    const body = {
      lat,
      lon,
      hour,
      day_of_week,
      month,
      crime_type: selectedType === "ALL" ? null : selectedType,
    };

    fetch(`${API}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then((r) => {
        if (!r.ok) throw new Error("Location is outside Chicago bounds");
        return r.json();
      })
      .then((d) => setPrediction({ ...d, clickLat: lat, clickLon: lon }))
      .catch((err) => {
        setPrediction(null);
        setError(err.message + " — please click within the Chicago city area");
        setTimeout(() => setError(null), 4000);
      });
  }, [selectedType, getTemporalParams]);

  // Re-predict when date/time changes if we already have a prediction
  useEffect(() => {
    if (prediction) {
      handlePredict(prediction.clickLat, prediction.clickLon);
    }
  }, [selectedDate, selectedTime]);

  const dayNames = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
  const monthNames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

  const temporalInfo = () => {
    const { hour, day_of_week, month } = getTemporalParams();
    return `${dayNames[day_of_week]}, ${monthNames[month - 1]}, ${hour}:00`;
  };

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-left">
          <h1>🗺 Chicago Crime Risk Map</h1>
          <span className="subtitle">GCN-based spatial crime hotspot prediction</span>
        </div>
        <div className="header-right">
          {/* Date picker */}
          <div className="control-group">
            <label>Date</label>
            <input
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              className="date-input"
            />
          </div>
          {/* Time picker */}
          <div className="control-group">
            <label>Time</label>
            <input
              type="time"
              value={selectedTime}
              onChange={(e) => setSelectedTime(e.target.value)}
              className="date-input"
            />
          </div>
          {/* Crime type */}
          <div className="control-group">
            <label>Crime Type</label>
            <select
              value={selectedType}
              onChange={(e) => setSelectedType(e.target.value)}
            >
              {crimeTypes.map((ct) => (
                <option key={ct} value={ct}>{ct}</option>
              ))}
            </select>
          </div>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="main">
        {/* ── Map ── */}
        <div className="map-wrap">
          {loading && <div className="map-loading">Loading nodes…</div>}
          <MapContainer
            center={[41.85, -87.65]}
            zoom={12}
            style={{ height: "100%", width: "100%" }}
          >
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution="© OpenStreetMap contributors"
            />
            <MapClickHandler onPredict={handlePredict} />

            {nodes.map((node, i) => (
              <CircleMarker
                key={i}
                center={[node.lat, node.lon]}
                radius={8}
                pathOptions={{
                  fillColor: riskColor(node.risk_score),
                  fillOpacity: 0.8,
                  color: "#fff",
                  weight: 0.5,
                }}
              >
                <Popup>
                  <strong>{node.dominant_crime}</strong><br />
                  Risk: <b>{(node.risk_score * 100).toFixed(1)}%</b>
                  &nbsp;({riskLabel(node.risk_score)})<br />
                  Historical crimes: {node.crime_count}<br />
                  <hr style={{ margin: "6px 0" }} />
                  <small>
                    {Object.entries(node.risk_scores)
                      .sort((a, b) => b[1] - a[1])
                      .slice(0, 3)
                      .map(([k, v]) => `${k}: ${(v * 100).toFixed(1)}%`)
                      .join(" | ")}
                  </small>
                </Popup>
              </CircleMarker>
            ))}
          </MapContainer>
          <Legend />
        </div>

        {/* ── Sidebar ── */}
        <aside className="sidebar">

          {/* Temporal info card */}
          <div className="card time-card">
            <h3>🕐 Selected Time</h3>
            <p className="time-display">{temporalInfo()}</p>
            <p className="time-hint">Click anywhere on the map to predict crime risk for this date & time</p>
          </div>

          {/* Stats */}
          {stats && (
            <div className="card">
              <h3>📊 Overview</h3>
              <div className="stat-row">
                <span>Grid cells</span>
                <b>{stats.total_nodes}</b>
              </div>
              <div className="stat-row">
                <span>Avg crimes/cell</span>
                <b>{stats.avg_crime_count.toLocaleString()}</b>
              </div>
              <h4 style={{ marginTop: 12 }}>Top Hotspots</h4>
              {stats.top_hotspots.map((h, i) => (
                <div key={i} className="hotspot-row">
                  <span className="risk-pill" style={{ background: riskColor(h.risk) }}>
                    {(h.risk * 100).toFixed(0)}%
                  </span>
                  <span>{h.crime}</span>
                  <span className="coords">{h.lat.toFixed(3)}, {h.lon.toFixed(3)}</span>
                </div>
              ))}
            </div>
          )}

          {/* Prediction panel */}
          {prediction && (
            <div className="card prediction-card">
              <h3>📍 Prediction</h3>
              <div className="stat-row">
                <span>Location</span>
                <b>{prediction.clickLat.toFixed(4)}, {prediction.clickLon.toFixed(4)}</b>
              </div>
              <div className="stat-row">
                <span>Time</span>
                <b>{temporalInfo()}</b>
              </div>
              <div className="stat-row">
                <span>Predicted crime</span>
                <b>{prediction.predicted_crime}</b>
              </div>
              <div className="stat-row">
                <span>Risk score</span>
                <b style={{ color: riskColor(prediction.risk_score) }}>
                  {(prediction.risk_score * 100).toFixed(1)}% ({riskLabel(prediction.risk_score)})
                </b>
              </div>
              {prediction.temporal_adjusted && (
                <div className="temporal-badge">⏱ Temporally adjusted</div>
              )}
              <h4 style={{ marginTop: 12 }}>Crime Risk Breakdown</h4>
              {Object.entries(prediction.risk_scores)
                .sort((a, b) => b[1] - a[1])
                .map(([crime, risk]) => (
                  <div key={crime} className="risk-bar-row">
                    <span className="risk-bar-label">{crime}</span>
                    <div className="risk-bar-bg">
                      <div
                        className="risk-bar-fill"
                        style={{
                          width: `${(risk * 100).toFixed(1)}%`,
                          background: riskColor(risk),
                        }}
                      />
                    </div>
                    <span className="risk-bar-pct">{(risk * 100).toFixed(1)}%</span>
                  </div>
                ))}
            </div>
          )}

          {!prediction && (
            <div className="card hint-card">
              <p>🖱 Click anywhere on the map to get a crime risk prediction for the selected date & time.</p>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
