import sqlite3
import os
import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'study_assistant.db')

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    # Create the sessions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,
            duration_seconds INTEGER DEFAULT 0,
            unfocused_seconds INTEGER DEFAULT 0,
            yawn_count INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def create_session(user_id="local_tester"):
    """
    Creates a new study session in the database and returns the session_id.
    """
    conn = get_connection()
    c = conn.cursor()
    today_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute('''
        INSERT INTO sessions (user_id, date, duration_seconds, unfocused_seconds, yawn_count)
        VALUES (?, ?, 0, 0, 0)
    ''', (user_id, today_str))
    
    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return session_id

def update_session(session_id, additional_duration, additional_unfocused, additional_yawns):
    """
    Updates an active session incrementally so we don't lose data if the server crashes.
    """
    conn = get_connection()
    c = conn.cursor()
    
    c.execute('''
        UPDATE sessions 
        SET duration_seconds = duration_seconds + ?,
            unfocused_seconds = unfocused_seconds + ?,
            yawn_count = yawn_count + ?
        WHERE id = ?
    ''', (additional_duration, additional_unfocused, additional_yawns, session_id))
    
    conn.commit()
    conn.close()

def get_weekly_analytics(user_id="local_tester"):
    """
    Returns analytics grouped by date (ignoring time) for the last 7 days.
    """
    conn = get_connection()
    c = conn.cursor()
    
    # We use SUBSTR(date, 1, 10) to get just the YYYY-MM-DD part and sum the stats per day
    c.execute('''
        SELECT 
            SUBSTR(date, 1, 10) as day,
            SUM(duration_seconds) as total_duration,
            SUM(unfocused_seconds) as total_unfocused,
            SUM(yawn_count) as total_yawns
        FROM sessions
        WHERE user_id = ?
        GROUP BY day
        ORDER BY day DESC
        LIMIT 7
    ''', (user_id,))
    
    rows = c.fetchall()
    conn.close()
    
    # Format the data for recharts (reverse so oldest is first)
    analytics_data = []
    for row in reversed(rows):
        day, dur, unfoc, yawns = row
        analytics_data.append({
            "name": day,
            "Study Minutes": round(dur / 60, 1),
            "Unfocused Mins": round(unfoc / 60, 1),
            "Yawns": yawns
        })
        
    return analytics_data

# Initialize exact schema on import
init_db()
