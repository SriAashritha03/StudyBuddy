import cv2
import numpy as np
import tensorflow as tf
import time
import os
import sys
import threading
import base64
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
latest_frame_time = 0
camera_thread = None
emotion_cache = {"emotion": "neutral", "time": 0}

def find_working_camera():
    """Try to find a working camera with different backends and indices"""
    print("Searching for available cameras...")

    # Try different backends and camera indices
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_VFW, "VFW"),
        (cv2.CAP_ANY, "AUTO")
    ]

    for backend, backend_name in backends:
        for camera_index in range(5):  # Try indices 0-4
            print(f"  Trying backend={backend_name}, index={camera_index}...")
            try:
                cap = cv2.VideoCapture(camera_index, backend)
                if cap.isOpened():
                    # Test if we can actually read a frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"✓ SUCCESS: Found camera at index {camera_index} using {backend_name}")
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        return cap
                    else:
                        cap.release()
            except Exception as e:
                print(f"    Error: {e}")

    print("✗ No working camera found!")
    return None

def update_camera_loop():
    global camera, latest_frame, latest_frame_time
    try:
        print("Initializing camera...")

        # Try to find a working camera
        camera = find_working_camera()

        if camera is None:
            print("ERROR: Could not initialize any camera!")
            print("\nTroubleshooting steps:")
            print("1. Check if camera is connected")
            print("2. Close other applications using the camera (Zoom, Discord, etc.)")
            print("3. Try unplugging and replugging the camera")
            print("4. Check Device Manager for camera drivers")
            return

        print("Camera initialized. Waiting for first frame...")

        frame_count = 0
        error_count = 0
        last_print_time = time.time()

        while True:
            # Blocking call ensures we read exactly as fast as hardware pushes
            success, frame = camera.read()
            if success and frame is not None:
                latest_frame = frame.copy()  # Make sure to copy the frame
                latest_frame_time = time.time()
                frame_count += 1
                error_count = 0  # Reset error count

                # Print status every 10 frames (more frequent for debugging)
                if frame_count % 10 == 0:
                    now = time.time()
                    fps = 10 / (now - last_print_time) if (now - last_print_time) > 0 else 0
                    print(f"✓ [DEBUG] Frames: {frame_count} | FPS: {fps:.1f} | Latest: {latest_frame_time:.2f}")
                    last_print_time = now

                if frame_count == 1:
                    print("✓ First frame captured!")
            else:
                error_count += 1
                if error_count % 30 == 0:  # Print every 30 failed attempts
                    print(f"⚠ Warning: Failed to read frame from camera (attempt {error_count})")
                time.sleep(0.01)

    except Exception as e:
        print(f"❌ Error in camera loop: {e}")
        import traceback
        traceback.print_exc()
        time.sleep(1)
    finally:
        if camera is not None:
            camera.release()
            print("📹 Camera released")

def ai_processing_loop():
    """Background thread for heavy AI processing - doesn't block video feed"""
    global app_state, latest_frame

    while True:
        try:
            if latest_frame is None:
                time.sleep(0.1)
                continue

            current_time = time.time()
            frame = latest_frame.copy()

            # Check Break Mode
            if app_state["global_break_timer_end"] > 0:
                rem_secs = int(app_state["global_break_timer_end"] - current_time)
                if rem_secs > 0:
                    mins, secs = divmod(rem_secs, 60)
                    app_state["current_state_str"] = "ON BREAK"
                    app_state["time_unfocused"] = 0
                    app_state["yawn_timestamps"] = []
                    continue
                else:
                    app_state["global_break_timer_end"] = 0

            # Run model inference every 0.5 seconds
            should_run_model = (current_time - app_state["last_inference_time"]) > 0.5
            if should_run_model:
                processed_img = preprocess_frame(frame)
                predictions = model.predict(processed_img, verbose=0) if model else [[0]*3]
                app_state["cached_predicted_idx"] = np.argmax(predictions, axis=1)[0]
                app_state["last_inference_time"] = current_time

            predicted_class_idx = app_state["cached_predicted_idx"]

            # Haar Cascades for face/eye detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            face_found = len(faces) > 0
            eyes_found = False

            if face_found:
                (x, y, w, h) = faces[0]
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=7)
                eyes_found = len(eyes) > 0

                # Run DeepFace every 2 seconds
                if (current_time - emotion_cache["time"]) > 2.0:
                    try:
                        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                        emotion = results[0]['dominant_emotion'] if isinstance(results, list) else results['dominant_emotion']
                        emotion_cache["emotion"] = emotion
                        emotion_cache["time"] = current_time
                        app_state["cached_emotion"] = emotion
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

            # If we need a new popup
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

        except Exception as e:
            print(f"Error in AI processing: {e}")
            time.sleep(0.1)

def generate_frames():
    """Legacy function - no longer used"""
    pass


@app.route('/video_feed')
def video_feed():
    """Raw video feed - minimal processing for low latency"""
    global latest_frame, camera_thread

    if camera_thread is None:
        camera_thread = threading.Thread(target=update_camera_loop, daemon=True)
        camera_thread.start()

    def generate_raw_frames():
        while True:
            if latest_frame is None:
                time.sleep(0.001)
                continue

            frame = latest_frame.copy()
            frame = cv2.flip(frame, 1)

            # Minimal processing - just add state text overlay
            color = (0, 255, 0) if app_state["current_state_str"] == "Normal" else (0, 0, 255)
            cv2.putText(frame, f"State: {app_state['current_state_str']}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Emotion: {app_state['cached_emotion']}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Ultra-fast JPEG encoding (quality 75)
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate_raw_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_json')
def video_feed_json():
    """JSON-based real-time frame delivery - better for canvas rendering"""
    global latest_frame, camera_thread

    if camera_thread is None:
        camera_thread = threading.Thread(target=update_camera_loop, daemon=True)
        camera_thread.start()

    # Wait up to 2 seconds for first frame
    wait_time = 0
    while latest_frame is None and wait_time < 2:
        time.sleep(0.05)
        wait_time += 0.05

    if latest_frame is None:
        # Return a placeholder image if camera still not ready
        return jsonify({"error": "Camera initializing..."}), 503

    try:
        frame = latest_frame.copy()
        frame = cv2.flip(frame, 1)

        # Minimal overlay
        color = (0, 255, 0) if app_state["current_state_str"] == "Normal" else (0, 0, 255)
        cv2.putText(frame, f"State: {app_state['current_state_str']}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Emotion: {app_state['cached_emotion']}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Add timestamp to verify frames are updating
        timestamp = time.time()
        cv2.putText(frame, f"T: {timestamp:.2f}", (15, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ret:
            return jsonify({"error": "Failed to encode frame"}), 500

        frame_bytes = buffer.tobytes()
        frame_b64 = base64.b64encode(frame_bytes).decode('utf-8')

        return jsonify({
            "frame": frame_b64,
            "timestamp": timestamp,
            "state": app_state["current_state_str"],
            "emotion": app_state["cached_emotion"]
        })
    except Exception as e:
        print(f"Error in video_feed_json: {e}")
        return jsonify({"error": str(e)}), 500

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

@app.route('/api/stop_break', methods=['POST'])
def stop_break():
    """Stop the current break early"""
    app_state["global_break_timer_end"] = 0
    app_state["current_state_str"] = "Normal"
    return jsonify({"status": "break_stopped"})

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
    # Start AI processing in a background thread
    ai_thread = threading.Thread(target=ai_processing_loop, daemon=True)
    ai_thread.start()

    # Start camera thread
    camera_thread_init = threading.Thread(target=update_camera_loop, daemon=True)
    camera_thread_init.start()

    app.run(host='0.0.0.0', port=5000, threaded=True)
