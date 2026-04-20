import { useState, useEffect, useRef } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import './index.css';

const API_BASE_URL = 'http://localhost:5000/api';
const VIDEO_FEED_URL = 'http://localhost:5000/video_feed';

function App() {
  const canvasRef = useRef(null);
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
  const [isBreakActive, setIsBreakActive] = useState(false);
  const [breakEndTime, setBreakEndTime] = useState(null);
  const [breakTimeRemaining, setBreakTimeRemaining] = useState(0);

  const fetchAnalytics = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/analytics`);
      const data = await res.json();
      setAnalyticsData(data.analytics);
      setActivePomodoro(data.active_pomodoro);
    } catch (err) { }
  };

  // Real-time video streaming with Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let isRunning = true;
    let failCount = 0;
    let frameCount = 0;
    let lastLogTime = Date.now();

    const streamVideo = async () => {
      while (isRunning) {
        try {
          const response = await fetch(`${API_BASE_URL.replace('/api', '')}/video_feed_json`);
          if (response.ok) {
            const data = await response.json();
            const img = new Image();

            // Create a promise that resolves when image loads OR times out
            const loadPromise = new Promise((resolve) => {
              let loaded = false;

              img.onload = () => {
                if (!loaded) {
                  loaded = true;
                  if (isRunning) {
                    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                    frameCount++;

                    // Log every 30 frames
                    const now = Date.now();
                    if (now - lastLogTime > 1000) {
                      console.log(`📹 Frames rendered: ${frameCount} | Backend timestamp: ${data.timestamp?.toFixed(2)}`);
                      lastLogTime = now;
                    }
                  }
                  resolve();
                }
              };

              img.onerror = () => {
                if (!loaded) {
                  loaded = true;
                  console.error('❌ Image failed to load');
                  resolve();
                }
              };

              // Timeout after 1 second
              setTimeout(() => {
                if (!loaded) {
                  loaded = true;
                  console.warn('⏱ Image load timeout');
                  resolve();
                }
              }, 1000);

              img.src = `data:image/jpeg;base64,${data.frame}`;
            });

            await loadPromise;
            failCount = 0;
          } else if (response.status === 503) {
            // Camera initializing
            failCount++;
            if (failCount === 1) {
              ctx.fillStyle = '#333';
              ctx.fillRect(0, 0, canvas.width, canvas.height);
              ctx.fillStyle = '#FFF';
              ctx.font = '20px Arial';
              ctx.textAlign = 'center';
              ctx.fillText('Initializing camera...', canvas.width / 2, canvas.height / 2);
              console.log('⏳ Camera initializing...');
            }
            await new Promise(resolve => setTimeout(resolve, 100));
          }
        } catch (err) {
          console.error('❌ Video fetch error:', err);
          await new Promise(resolve => setTimeout(resolve, 100));
        }
        // Minimal delay - as fast as possible
        await new Promise(resolve => setTimeout(resolve, 16)); // ~60fps
      }
    };

    streamVideo();

    return () => {
      isRunning = false;
    };
  }, []);

  useEffect(() => {
    fetchAnalytics();

    const fetchState = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/state`);
        const data = await res.json();
        setStateData(data);
      } catch (err) { }
    };

    const interval = setInterval(fetchState, 500); // Check state more frequently
    return () => clearInterval(interval);
  }, []);

  // Break timer countdown
  useEffect(() => {
    if (!isBreakActive || !breakEndTime) return;

    const timer = setInterval(() => {
      const now = Date.now();
      const remaining = Math.max(0, breakEndTime - now);
      setBreakTimeRemaining(remaining);

      if (remaining === 0) {
        setIsBreakActive(false);
        setBreakEndTime(null);
      }
    }, 1000);

    return () => clearInterval(timer);
  }, [isBreakActive, breakEndTime]);

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
      // Auto-dismiss modal after feedback
      dismissPopup();
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

      // Set break end time
      const endTime = Date.now() + (parseInt(breakMins) * 60 * 1000);
      setBreakEndTime(endTime);
      setIsBreakActive(true);

      // Auto-dismiss modal
      dismissPopup();
    } catch (err) { }
  };

  const stopBreak = async () => {
    try {
      await fetch(`${API_BASE_URL}/stop_break`, { method: 'POST' });
      setIsBreakActive(false);
      setBreakEndTime(null);
      setBreakTimeRemaining(0);
    } catch (err) { }
  };

  const dismissPopup = async () => {
    try {
      await fetch(`${API_BASE_URL}/dismiss`, { method: 'POST' });
    } catch (e) {}
  };

  // Format break time remaining
  const formatBreakTime = (ms) => {
    if (ms <= 0) return "0:00";
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
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
             <canvas ref={canvasRef} width={640} height={480} style={{ width: '100%', height: 'auto', borderRadius: '12px', backgroundColor: '#000' }} />
          </div>
          <div className="status-badges">
            {isBreakActive ? (
              <>
                <div className="badge badge-break">
                  ☕ ON BREAK
                </div>
                <div className="badge badge-break-timer">
                  ⏱ {formatBreakTime(breakTimeRemaining)}
                </div>
                <button
                  className="badge badge-danger"
                  onClick={stopBreak}
                >
                  ⏹ Stop Break
                </button>
              </>
            ) : (
              <>
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
              </>
            )}
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
               <p>Do you accept this suggestion?</p>
               <div className="fb-buttons">
                 <button className="btn-success" onClick={() => { handleFeedback(1.0); }}>✓ Accept</button>
                 <button className="btn-danger" onClick={() => { handleFeedback(-1.0); }}>✕ Decline</button>
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
