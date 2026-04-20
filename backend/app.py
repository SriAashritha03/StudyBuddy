import cv2
import numpy as np
import tensorflow as tf
import time
import os
import sys
import threading
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from deepface import DeepFace

# Import RL agent and DB
sys.path.append(os.path.join(os.path.dirname(__file__)))
from q_learning import StudyAssistantRL
import db

app = Flask(__name__)
CORS(app)

MODEL_PATH = r"d:\AI_Study_Assistant\backend\study_assistant_efficientnet_best.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded.")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

rl_agent = StudyAssistantRL(user_id='local_tester', model_dir='backend/user_models')

# Detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# Global State for tracking AI metrics
app_state = {
    "current_state_str": "Waiting...",
    "emotion_str": "Waiting...",
    "bad_state_start_time": None,
    "last_break_time": 0,
    "yawn_timestamps": [],
    "previous_state_str": "Normal",
    "global_break_timer_end": 0,
    "should_popup": False,
    "popup_state_trigger": "",
    "popup_message": "",
    "popup_title": "",
    "suggestion": "",
    "time_unfocused": 0,
    # Phase 3 Pomodoro State
    "active_session_id": None,
    "session_duration_secs": 0,
    "session_unfocused_secs": 0,
    "session_yawn_count": 0,
    "last_loop_time": 0,
    "last_inference_time": 0,
    "cached_emotion": "neutral",
    "cached_predicted_idx": 0
}

SUSTAINED_TIME = 60
COOLDOWN_TIME = 300

def preprocess_frame(frame, target_size=(224, 224)):
    img = cv2.resize(frame, target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

camera = None
latest_frame = None
camera_thread = None

def update_camera_loop():
    global camera, latest_frame
    # Force DirectShow backend on Windows which natively supports CAP_PROP_BUFFERSIZE=1
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Force a standard non-HD resolution so it runs extremely fast
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        # Blocking call ensures we read exactly as fast as hardware pushes
        success, frame = camera.read()
        if success:
            latest_frame = frame

def generate_frames():
    global camera, app_state, latest_frame, camera_thread
    
    if camera_thread is None:
        camera_thread = threading.Thread(target=update_camera_loop, daemon=True)
        camera_thread.start()
        
    while True:
        if latest_frame is None:
            time.sleep(0.1)
            continue
            
        # Copy to avoid modifying the thread's frame simultaneously
        frame = latest_frame.copy()
        frame = cv2.flip(frame, 1)
        current_time = time.time()
        
        # 0. Check Break Mode
        if app_state["global_break_timer_end"] > 0:
            rem_secs = int(app_state["global_break_timer_end"] - current_time)
            if rem_secs > 0:
                mins, secs = divmod(rem_secs, 60)
                app_state["current_state_str"] = "ON BREAK"
                app_state["time_unfocused"] = 0
                app_state["yawn_timestamps"] = []
                
                # Render Break overlay server side
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)
                cv2.putText(frame, "ON BREAK", (150, 200), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 4)
                cv2.putText(frame, f"{mins}:{secs:02d} remaining", (150, 300), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                continue
            else:
                app_state["global_break_timer_end"] = 0

        # AI Throttling: Run heavy inference only 3 times a second to keep video feed at 30 FPS!
        should_run_ai = (current_time - app_state["last_inference_time"]) > 0.3
        
        if should_run_ai:
            processed_img = preprocess_frame(frame)
            predictions = model.predict(processed_img, verbose=0) if model else [[0]*3]
            app_state["cached_predicted_idx"] = np.argmax(predictions, axis=1)[0]
            app_state["last_inference_time"] = current_time
            
        predicted_class_idx = app_state["cached_predicted_idx"]
        
        # Haar Cascades are fast enough for every frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        face_found = len(faces) > 0
        eyes_found = False
        
        if face_found:
            (x, y, w, h) = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=7)
            eyes_found = len(eyes) > 0
            
            if should_run_ai:
                try:
                    results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                    app_state["cached_emotion"] = results[0]['dominant_emotion'] if isinstance(results, list) else results['dominant_emotion']
                except:
                    pass
                    
        emotion_str = app_state["cached_emotion"]
        is_bad_mood = emotion_str in ['sad', 'angry', 'fear', 'disgust']
        
        if not face_found:
            current_state_str = "Away / Not at Desk"
        elif not eyes_found:
            current_state_str = "Sleeping / Eyes Closed"
        elif predicted_class_idx == 2:
            current_state_str = "Yawning"
        elif is_bad_mood:
            current_state_str = "Not in Good Mood"
        else:
            current_state_str = "Normal"
            
        app_state["emotion_str"] = emotion_str
        
        # Yawning filter
        if current_state_str == "Yawning" and app_state["previous_state_str"] != "Yawning":
            app_state["yawn_timestamps"].append(current_time)
            if app_state["active_session_id"]:
                app_state["session_yawn_count"] += 1
            
        app_state["previous_state_str"] = current_state_str
        app_state["yawn_timestamps"] = [t for t in app_state["yawn_timestamps"] if current_time - t <= 60]
        
        # Pomodoro Database metrics update
        if app_state["active_session_id"]:
            time_delta = current_time - (app_state["last_loop_time"] or current_time)
            app_state["session_duration_secs"] += time_delta
            if current_state_str in ['Not in Good Mood', 'Yawning', 'Away / Not at Desk', 'Sleeping / Eyes Closed']:
                app_state["session_unfocused_secs"] += time_delta
                
        app_state["last_loop_time"] = current_time
        
        # Timer Logic
        bad_states = ['Not in Good Mood', 'Yawning', 'Away / Not at Desk', 'Sleeping / Eyes Closed']
        
        should_trigger = False
        if len(app_state["yawn_timestamps"]) >= 5:
            should_trigger = True
            app_state["popup_title"] = "Sleepy Yawning Alert"
            app_state["popup_message"] = "You've yawned 5 times in under a minute! You are exhausted. Take a break!"
        elif current_state_str in bad_states:
            if app_state["bad_state_start_time"] is None:
                app_state["bad_state_start_time"] = current_time
            else:
                elapsed = current_time - app_state["bad_state_start_time"]
                app_state["time_unfocused"] = int(elapsed)
                if elapsed >= SUSTAINED_TIME:
                    should_trigger = True
                    app_state["popup_title"] = "Break Time"
                    app_state["popup_message"] = "You've been unfocused! Take a break!"
        else:
            app_state["bad_state_start_time"] = None
            app_state["time_unfocused"] = 0
            
        # If we need a new popup and we aren't currently showing one
        if should_trigger and not app_state["should_popup"]:
            triggering_state = "Yawning" if len(app_state["yawn_timestamps"]) >= 5 else current_state_str
            app_state["popup_state_trigger"] = triggering_state
            
            if app_state["last_break_time"] != 0 and (current_time - app_state["last_break_time"] < COOLDOWN_TIME):
                app_state["popup_title"] = "Need more break?"
                app_state["popup_message"] = "Still showing fatigue... Need more rest?"
                
            app_state["suggestion"] = rl_agent.get_action(triggering_state)
            app_state["should_popup"] = True
            
            # Reset triggers
            app_state["last_break_time"] = current_time
            app_state["bad_state_start_time"] = None
            app_state["yawn_timestamps"] = []
        
        app_state["current_state_str"] = current_state_str
        
        # Draw basic debug on the MJPEG stream just in case (optional, we mostly rely on pure UI now)
        color = (0, 255, 0) if current_state_str == "Normal" else (0, 0, 255)
        cv2.putText(frame, f"State: {current_state_str}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"DeepFace: {emotion_str}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Encode MJPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
        # Crucial Network Throttling!
        # Do not yield thousands of duplicate frames per second and crash the browser
        time.sleep(0.03) 

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state')
def get_state():
    return jsonify({
        "current_state_str": app_state["current_state_str"],
        "emotion_str": app_state["emotion_str"],
        "time_unfocused": app_state["time_unfocused"],
        "yawn_count": len(app_state["yawn_timestamps"]),
        "should_popup": app_state["should_popup"],
        "popup_message": app_state["popup_message"],
        "popup_title": app_state["popup_title"],
        "suggestion": app_state["suggestion"],
        "popup_state_trigger": app_state["popup_state_trigger"]
    })
    
@app.route('/api/dismiss', methods=['POST'])
def dismiss_popup():
    app_state["should_popup"] = False
    return jsonify({"status": "ok"})

@app.route('/api/feedback', methods=['POST'])
def feedback():
    data = request.json
    reward = data.get('reward')
    action = data.get('action') 
    state = data.get('state')
    
    rl_agent.update(state=state, action=action, reward=reward)
    app_state["should_popup"] = False # Auto dismiss on feedback
    return jsonify({"status": "ok"})

@app.route('/api/break', methods=['POST'])
def start_break():
    data = request.json
    mins = int(data.get('minutes', 5))
    app_state["global_break_timer_end"] = time.time() + (mins * 60)
    app_state["should_popup"] = False
    return jsonify({"status": "break_started", "duration": mins})

# --- Phase 3 Routes ---

@app.route('/api/start_session', methods=['POST'])
def start_session():
    data = request.json
    session_id = db.create_session("local_tester")
    app_state["active_session_id"] = session_id
    app_state["session_duration_secs"] = 0
    app_state["session_unfocused_secs"] = 0
    app_state["session_yawn_count"] = 0
    app_state["last_loop_time"] = time.time()
    return jsonify({"status": "started", "session_id": session_id})

@app.route('/api/end_session', methods=['POST'])
def end_session():
    if not app_state["active_session_id"]:
        return jsonify({"status": "no_active_session"}), 400
        
    db.update_session(
        app_state["active_session_id"],
        int(app_state["session_duration_secs"]),
        int(app_state["session_unfocused_secs"]),
        app_state["session_yawn_count"]
    )
    
    app_state["active_session_id"] = None
    return jsonify({"status": "ended"})

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    data = db.get_weekly_analytics("local_tester")
    # if empty db, return a placeholder so the UI has something to show!
    if not data:
        data = [
            {"name": "No Data Yet", "Study Minutes": 0, "Unfocused Mins": 0, "Yawns": 0}
        ]
    return jsonify({"analytics": data, "active_pomodoro": app_state["active_session_id"] is not None})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
