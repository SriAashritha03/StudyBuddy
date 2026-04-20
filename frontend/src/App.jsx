import { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import './index.css';

const API_BASE_URL = 'http://localhost:5000/api';

function App() {
  const [stateData, setStateData] = useState({
    current_state_str: 'Waiting...',
    emotion_str: '...',
    time_unfocused: 0,
    yawn_count: 0,
    should_popup: false,
    popup_message: '',
    popup_title: '',
    suggestion: '',
    popup_state_trigger: ''
  });

  const [analyticsData, setAnalyticsData] = useState([]);
  const [activePomodoro, setActivePomodoro] = useState(false);
  const [breakMins, setBreakMins] = useState(5);

  const fetchAnalytics = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/analytics`);
      const data = await res.json();
      setAnalyticsData(data.analytics);
      setActivePomodoro(data.active_pomodoro);
    } catch (err) { }
  };

  useEffect(() => {
    fetchAnalytics();
    
    const fetchState = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/state`);
        const data = await res.json();
        setStateData(data);
      } catch (err) { }
    };
    
    const interval = setInterval(fetchState, 1000);
    return () => clearInterval(interval);
  }, []);

  const togglePomodoro = async () => {
    try {
      if (activePomodoro) {
        await fetch(`${API_BASE_URL}/end_session`, { method: 'POST' });
        setActivePomodoro(false);
        fetchAnalytics(); // Refresh charts
      } else {
        await fetch(`${API_BASE_URL}/start_session`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}' });
        setActivePomodoro(true);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleFeedback = async (reward) => {
    try {
      await fetch(`${API_BASE_URL}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reward: reward,
          state: stateData.popup_state_trigger,
          action: stateData.suggestion
        })
      });
    } catch (err) { }
  };

  const startBreak = async () => {
    try {
      // If we take a break, end the current session so it gets logged!
      if (activePomodoro) {
          await fetch(`${API_BASE_URL}/end_session`, { method: 'POST' });
          setActivePomodoro(false);
          fetchAnalytics();
      }
      
      await fetch(`${API_BASE_URL}/break`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ minutes: breakMins })
      });
    } catch (err) { }
  };

  const dismissPopup = async () => {
    try {
      await fetch(`${API_BASE_URL}/dismiss`, { method: 'POST' });
    } catch (e) {}
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>AI Study Assistant</h1>
        <p>Premium Real-Time Analysis Dashboard</p>
      </header>

      <main className="dashboard">
        
        {/* Video Column */}
        <section className="video-section glass-panel">
          <div className="video-wrapper">
             <img src="http://localhost:5000/video_feed" alt="Video Feed Off: Start Backend!" />
          </div>
          <div className="status-badges">
            <div className={`badge ${stateData.current_state_str === "Normal" ? 'badge-ok' : 'badge-warn'}`}>
               State: {stateData.current_state_str}
            </div>
            <div className="badge badge-neutral">
               Face Scan: {stateData.emotion_str}
            </div>
            <button 
              className={`badge badge-pointer ${activePomodoro ? 'badge-danger' : 'badge-success'}`}
              onClick={togglePomodoro}
            >
              {activePomodoro ? "⏹ End Pomodoro" : "▶ Start Pomodoro"}
            </button>
          </div>
        </section>

        {/* Stats Column */}
        <section className="stats-section glass-panel">
           <h2>Activity Metrics</h2>
           
           <div className="stat-card">
              <span className="stat-label">Time Unfocused</span>
              <span className={`stat-value ${stateData.time_unfocused > 45 ? 'alert' : ''}`}>
                 {stateData.time_unfocused}s
              </span>
           </div>

           <div className="stat-card">
              <span className="stat-label">Yawns (Last 60s)</span>
              <span className={`stat-value ${stateData.yawn_count >= 3 ? 'alert' : ''}`}>
                 {stateData.yawn_count}
              </span>
           </div>
           
           <div className="instructions">
              <h3>Pomodoro Mode</h3>
              <p>Click "Start Pomodoro" to begin logging your session to the database. The AI acts as a failsafe and will pop up an alert if you reach fatigue or focus thresholds.</p>
           </div>
        </section>
      </main>

      {/* Analytics Column */}
      <section className="analytics-section glass-panel">
        <h2>Weekly Study Analytics</h2>
        <div style={{ width: '100%', height: 300, marginTop: '20px' }}>
          <ResponsiveContainer>
            <BarChart data={analyticsData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{ backgroundColor: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: '#fff' }} />
              <Legend />
              <Bar dataKey="Study Minutes" fill="#6366f1" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Unfocused Mins" fill="#ef4444" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Yawns" fill="#10b981" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Break Modal Overlay */}
      {stateData.should_popup && (
        <div className="modal-overlay">
          <div className="modal glass-panel">
            <button className="close-btn" onClick={dismissPopup}>&times;</button>
            <h2 className="modal-title">{stateData.popup_title}</h2>
            <p className="modal-message">{stateData.popup_message}</p>
            
            <div className="ai-suggestion">
               <strong>AI Suggests:</strong> {stateData.suggestion}
            </div>

            <div className="feedback-container">
               <p>Was this suggestion helpful?</p>
               <div className="fb-buttons">
                 <button className="btn-success" onClick={() => handleFeedback(1.0)}>Yes (+1)</button>
                 <button className="btn-danger" onClick={() => handleFeedback(-1.0)}>No (-1)</button>
               </div>
            </div>

            <div className="stopwatch-container">
               <label>Set Timer (mins): </label>
               <input 
                 type="number" 
                 min="1" max="120" 
                 value={breakMins}
                 onChange={(e) => setBreakMins(e.target.value)} 
               />
               <button className="btn-primary" onClick={startBreak}>Start Break Timer</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
