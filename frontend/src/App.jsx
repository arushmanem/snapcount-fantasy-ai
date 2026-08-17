import { useState, useEffect } from 'react';
import axios from 'axios';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis
} from 'recharts';
import { Trophy, TrendingUp, Settings, Activity } from 'lucide-react';
import './App.css';
import PlayerSearch from './PlayerSearch';

function App() {
  const [position, setPosition] = useState('QB');
  const [players, setPlayers] = useState([]);
  const [p1, setP1] = useState('');
  const [p2, setP2] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [graphData, setGraphData] = useState([]);
  const [radarData, setRadarData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [scoring, setScoring] = useState('ppr');

  // NEW: week selector
  const [weeks, setWeeks] = useState([]);
  const [week, setWeek] = useState(null);

  useEffect(() => {
    axios.get(`http://localhost:8000/players/${position}`)
      .then(res => {
        setPlayers(res.data.players);
        if (res.data.players.length >= 2) {
          setP1(res.data.players[0]);
          setP2(res.data.players[1]);
        }
      })
      .catch(err => console.error("API Error:", err));
  }, [position]);

  // Fetch available weeks once
  useEffect(() => {
    axios.get('http://localhost:8000/weeks')
      .then(res => {
        setWeeks(res.data.weeks || []);
        setWeek(res.data.default_week ?? null);
      })
      .catch(err => console.error("Weeks API Error:", err));
  }, []);

  const handleAnalyze = async () => {
    if (!p1 || !p2) return;
    if (!week) return;

    setLoading(true);
    setPrediction(null);

    try {
      const predRes = await axios.get(
        `http://localhost:8000/predict?player1=${encodeURIComponent(p1)}&player2=${encodeURIComponent(p2)}&format=${scoring}&week=${week}`
      );
      setPrediction(predRes.data);

      const graphRes = await axios.get(
        `http://localhost:8000/compare?player1=${encodeURIComponent(p1)}&player2=${encodeURIComponent(p2)}&format=${scoring}&week=${week}`
      );      
      setGraphData(processGraphData(graphRes.data.history));
      setRadarData(graphRes.data.radar);
    } catch (error) {
      console.error(error);
      alert("Error fetching data. Check backend console.");
    }

    setLoading(false);
  };

  return (
    <div className="container">
      <header>
        <h1>🏈 SnapCount</h1>
        <div className="subtitle">AI-Powered Fantasy Assistant</div>
      </header>

      <div className="controls">
        {/* Row 1: Settings */}
        <div className="controls-row settings-row">
          <div className="settings-group">
            <Settings size={20} color="#94a3b8" />
            <select value={scoring} onChange={(e) => setScoring(e.target.value)}>
              <option value="ppr">PPR (1.0)</option>
              <option value="half_ppr">Half PPR (0.5)</option>
              <option value="standard">Standard (0.0)</option>
            </select>
          </div>

          <select value={week ?? ''} onChange={(e) => setWeek(Number(e.target.value))}>
            {weeks.map(w => <option key={w} value={w}>Week {w}</option>)}
          </select>

          <select value={position} onChange={(e) => setPosition(e.target.value)}>
            <option value="QB">Quarterbacks</option>
            <option value="RB">Running Backs</option>
            <option value="WR">Wide Receivers</option>
            <option value="TE">Tight Ends</option>
          </select>
        </div>

        {/* Row 2: Player comparison */}
        <div className="controls-row compare-row">
          <PlayerSearch
            players={players}
            value={p1}
            onChange={setP1}
            placeholder="Search Player 1..."
          />

          <span className="vs-label">VS</span>

          <PlayerSearch
            players={players}
            value={p2}
            onChange={setP2}
            placeholder="Search Player 2..."
          />

          <button
            className="analyze-btn"
            onClick={handleAnalyze}
            disabled={!p1 || !p2 || loading}
          >
            {loading ? 'Loading...' : 'ANALYZE'}
          </button>

        </div>
      </div>


      {loading && <div className="loading">Crunching the numbers...</div>}

      {prediction && (
        <div className="dashboard">
          <div className="card winner-section">
            <Trophy size={48} color="#22c55e" style={{ marginBottom: '10px' }} />
            <h2>Start <span style={{ color: '#22c55e' }}>{prediction.winner}</span></h2>
            <p className="subtitle">Projected Advantage: +{prediction.margin} Points</p>
            <p className="subtitle" style={{ marginTop: '-8px' }}>
              Week {prediction.week ?? week} • {String(prediction.format ?? scoring).toUpperCase()}
            </p>

            <div style={{
              margin: '20px auto',
              padding: '15px',
              background: 'rgba(16, 185, 129, 0.1)',
              borderLeft: '4px solid #10b981',
              borderRadius: '8px',
              maxWidth: '600px',
              textAlign: 'left',
              lineHeight: '1.6',
              fontSize: '1rem',
              color: '#d1fae5'
            }}>
              <strong>AI Analysis:</strong> {prediction.summary}
            </div>

            <div style={{ display: 'flex', justifyContent: 'center', gap: '40px', marginTop: '30px', flexWrap: 'wrap' }}>
              <PlayerStatBox data={prediction.details[0]} />
              <div style={{ width: '1px', background: '#334155', display: 'block' }}></div>
              <PlayerStatBox data={prediction.details[1]} />
            </div>
          </div>

          <div className="card" style={{ minHeight: '400px' }}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '10px' }}><TrendingUp /> Season Trend</h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={graphData}>
                <defs>
                  <linearGradient id="colorP1" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorP2" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22c55e" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="week" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px' }} itemStyle={{ color: '#fff' }} />
                <Legend />
                <Area type="monotone" dataKey={p1} stroke="#3b82f6" fillOpacity={1} fill="url(#colorP1)" />
                <Area type="monotone" dataKey={p2} stroke="#22c55e" fillOpacity={1} fill="url(#colorP2)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="card" style={{ minHeight: '400px' }}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: '10px' }}><Activity /> Skill Profile (Percentile)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart outerRadius={90} data={radarData}>
                <PolarGrid stroke="#334155" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                <Radar name={p1} dataKey="A" stroke="#3b82f6" strokeWidth={3} fill="#3b82f6" fillOpacity={0.4} />
                <Radar name={p2} dataKey="B" stroke="#22c55e" strokeWidth={3} fill="#22c55e" fillOpacity={0.4} />
                <Legend />
                <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px' }} itemStyle={{ color: '#fff' }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

function PlayerStatBox({ data }) {
  return (
    <div style={{ textAlign: 'left', minWidth: '200px' }}>
      <h3 style={{ margin: '0 0 10px 0', fontSize: '1.2rem' }}>{data.name}</h3>
      <div className="stat-row"><span style={{ color: '#94a3b8' }}>Projected</span> <span style={{ fontSize: '1.1em', color: '#fff' }}>{data.projected_points}</span></div>
      <div className="stat-row"><span style={{ color: '#94a3b8' }}>Avg</span> <span>{data.avg_points}</span></div>
      <div className="stat-row"><span style={{ color: '#94a3b8' }}>Opp (W{data.week ?? ''})</span> <span>{data.opponent}</span></div>

      <div style={{ marginTop: '15px', padding: '10px', background: 'rgba(255,255,255,0.05)', borderRadius: '8px' }}>
        <span style={{ fontSize: '0.8rem', textTransform: 'uppercase', color: '#64748b', fontWeight: 'bold' }}>Key Factors</span>
        <ul style={{ paddingLeft: '0', margin: '5px 0 0 0', listStyle: 'none' }}>
          {data.reasons && data.reasons.length > 0 ? (
            data.reasons.map((reason, i) => (
              <li key={i} style={{ fontSize: '0.85rem', marginBottom: '4px', color: '#cbd5e1' }}>
                {reason}
              </li>
            ))
          ) : (
            <li style={{ fontSize: '0.85rem', color: '#64748b' }}>No significant factors.</li>
          )}
        </ul>
      </div>
    </div>
  );
}

function processGraphData(rawData) {
  const weeks = {};
  (rawData || []).forEach(record => {
    if (!weeks[record.week]) weeks[record.week] = { week: record.week };
    weeks[record.week][record.player_name] = record.fantasy_points_ppr;
  });
  return Object.values(weeks).sort((a, b) => a.week - b.week);
}

export default App;
