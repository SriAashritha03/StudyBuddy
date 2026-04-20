import cv2
import numpy as np
import tensorflow as tf
import time
import os
import sys
import threading
import tkinter as tk
from tkinter import messagebox
from deepface import DeepFace

# Ensure backend modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
from q_learning import StudyAssistantRL

# Paths Configure
MODEL_PATH = r"d:\AI_Study_Assistant\backend\study_assistant_efficientnet_best.h5"

CLASS_MAPPING = {
    0: 'Normal',
    1: 'Not in Good Mood',
    2: 'Yawning'
}

# Global break timer (seconds epoch)
global_break_timer_end = 0

def preprocess_frame(frame, target_size=(224, 224)):
    img = cv2.resize(frame, target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0 # Rescale
    return img_array

def show_interactive_popup(title, message, rl_agent, state_str, suggestion_str):
    def run_gui():
        global global_break_timer_end
        root = tk.Tk()
        root.title(title)
        root.geometry("450x380")
        root.attributes("-topmost", True)
        
        # Message Label
        lbl = tk.Label(root, text=message, font=("Helvetica", 14, "bold"), justify="center", wraplength=400)
        lbl.pack(pady=15)
        
        # Feedback Frame
        fb_frame = tk.Frame(root, bd=2, relief="groove")
        fb_frame.pack(pady=10, padx=20, fill="x")
        tk.Label(fb_frame, text="Was this suggestion helpful?", font=("Helvetica", 11)).pack(pady=5)
        
        def give_feedback(reward):
            rl_agent.update(state=state_str, action=suggestion_str, reward=reward)
            btn_yes.config(state="disabled")
            btn_no.config(state="disabled")
            lbl_thx.config(text="Feedback saved to AI logic!")
            
        btn_yes = tk.Button(fb_frame, text="Yes (+1)", command=lambda: give_feedback(1.0), bg="lightgreen", width=10)
        btn_yes.pack(side="left", padx=40, pady=10)
        btn_no = tk.Button(fb_frame, text="No (-1)", command=lambda: give_feedback(-1.0), bg="lightcoral", width=10)
        btn_no.pack(side="right", padx=40, pady=10)
        lbl_thx = tk.Label(fb_frame, text="", fg="blue")
        lbl_thx.pack(side="bottom", pady=5)
        
        # Stopwatch Frame
        sw_frame = tk.Frame(root, bd=2, relief="groove")
        sw_frame.pack(pady=10, padx=20, fill="x")
        tk.Label(sw_frame, text="Set Break Stopwatch (Minutes):", font=("Helvetica", 11)).pack(pady=5)
        
        spin = tk.Spinbox(sw_frame, from_=1, to=120, width=5, font=("Helvetica", 12))
        spin.pack(pady=5)
        
        def start_timer():
            global global_break_timer_end
            mins = int(spin.get())
            global_break_timer_end = time.time() + (mins * 60)
            root.destroy()
            
        tk.Button(sw_frame, text="Start Break Countdown", command=start_timer, bg="lightblue", font=("Helvetica", 11, "bold")).pack(pady=10)
        
        root.mainloop()

    threading.Thread(target=run_gui).start()

def run_opencv_test():
    global global_break_timer_end
    print("Initializing RL Agent...")
    rl_agent = StudyAssistantRL(user_id='local_tester', model_dir='backend/user_models')
    
    print(f"Loading EfficientNet Yawning Model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Opening Webcam...")
    cap = cv2.VideoCapture(0)
    
    # Initialize native OpenCV face and strict eye detectors
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # eyeglasses cascade is much stricter and usually drops detection instantly when eyes are closed
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
    
    SUSTAINED_TIME = 60  
    COOLDOWN_TIME = 300   
    
    bad_state_start_time = None
    last_break_time = 0 
    
    yawn_timestamps = []
    previous_state_str = "Normal"
    
    print("\n--- TEST SCRIPT RUNNING ---")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        
        # 0. Check if we are currently ON A BREAK
        if global_break_timer_end > 0:
            rem_secs = int(global_break_timer_end - time.time())
            if rem_secs > 0:
                mins, secs = divmod(rem_secs, 60)
                cv2.rectangle(frame, (0, 0), (800, 600), (0, 0, 0), -1)
                cv2.putText(frame, "ON BREAK", (150, 200), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 4)
                cv2.putText(frame, f"{mins}:{secs:02d} remaining", (150, 300), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)
                
                # Prevent popup logic from running while on break
                bad_state_start_time = None
                
                cv2.imshow("AI Study Assistant - Enhanced Face/Mood Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue
            else:
                # Timer finished!
                global_break_timer_end = 0

        # Normal tracking when NOT on break
        processed_img = preprocess_frame(frame)
        
        # 1. Custom EfficientNet Model
        predictions = model.predict(processed_img, verbose=0)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        
        # 2. Manual OpenCV Face & Eye Detection + DeepFace Emotion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        face_found = len(faces) > 0
        eyes_found = False
        
        emotion_str = "neutral"
        if face_found:
            # Check for eyes within the first face found
            (x, y, w, h) = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            # using minNeighbors=7 to be incredibly strict; if eyes close, it will definitely drop to 0
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=7)
            eyes_found = len(eyes) > 0
            
            try:
                # We set enforce_detection to False since we manually verified the face exists
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                if isinstance(results, list):
                    emotion_str = results[0]['dominant_emotion']
                else:
                    emotion_str = results['dominant_emotion']
            except Exception:
                emotion_str = "neutral"
        else:
            emotion_str = "None"
            
        # 3. Decision Logic Combine
        current_state_str = "Unknown"
        is_bad_mood = emotion_str in ['sad', 'angry', 'fear', 'disgust']
        
        if not face_found:
            current_state_str = "Away / Not at Desk"
            confidence = 1.0
        elif not eyes_found:
            current_state_str = "Sleeping / Eyes Closed"
            confidence = 1.0
        elif predicted_class_idx == 2:
            current_state_str = "Yawning"
            confidence = float(predictions[0][2])
        elif is_bad_mood:
            current_state_str = "Not in Good Mood"
            confidence = 0.99
        else:
            current_state_str = "Normal"
            confidence = 0.99
            
        current_time = time.time()
        
        # --- Frequent Yawning Logic (5 times / min) ---
        if current_state_str == "Yawning" and previous_state_str != "Yawning":
            yawn_timestamps.append(current_time)
        previous_state_str = current_state_str
        
        # Keep only yawns from the last 60 seconds
        yawn_timestamps = [t for t in yawn_timestamps if current_time - t <= 60]
        
        # 4. Timer Logic
        bad_states = ['Not in Good Mood', 'Yawning', 'Away / Not at Desk', 'Sleeping / Eyes Closed']
        
        if len(yawn_timestamps) >= 5:
            if last_break_time != 0 and (current_time - last_break_time < COOLDOWN_TIME):
                msg = "You are still yawning heavily...\nYou definitely need more time to rest!"
                title = "Need more break?"
            else:
                msg = "You've yawned 5 times in under a minute!\nYou are exhausted. Take a break!"
                title = "Sleepy Yawning Alert"
            
            suggestion = rl_agent.get_action("Yawning")
            show_interactive_popup(title, msg, rl_agent, "Yawning", suggestion)
            
            last_break_time = current_time
            bad_state_start_time = None
            yawn_timestamps.clear()
            
        elif current_state_str in bad_states:
            if bad_state_start_time is None:
                bad_state_start_time = current_time 
            else:
                elapsed_bad_time = current_time - bad_state_start_time
                if elapsed_bad_time >= SUSTAINED_TIME:
                    # Timer triggered!
                    if last_break_time != 0 and (current_time - last_break_time < COOLDOWN_TIME):
                        msg = "You are not back to normal.\nDo you want some more time for a break?"
                        title = "Need more break?"
                        
                        # Show interactive UI
                        suggestion = rl_agent.get_action(current_state_str)
                        show_interactive_popup(title, msg, rl_agent, current_state_str, suggestion)
                    else:
                        suggestion = rl_agent.get_action(current_state_str)
                        msg = f"Take a break!\n\nAI Suggests: {suggestion}"
                        title = "Break Time"
                        
                        show_interactive_popup(title, msg, rl_agent, current_state_str, suggestion)
                        
                    last_break_time = current_time
                    bad_state_start_time = None 
        else:
            bad_state_start_time = None
                
        # 5. Visual Overlays
        cv2.rectangle(frame, (5, 5), (600, 100), (0, 0, 0), -1) 
        
        color = (0, 255, 0) if current_state_str == "Normal" else (0, 0, 255)
        cv2.putText(frame, f"State: {current_state_str} | DeepFace: {emotion_str}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
        if bad_state_start_time is not None:
             tracker_sec = int(current_time - bad_state_start_time)
             cv2.putText(frame, f"Unfocused for: {tracker_sec}s / 60s -> WILL SUGGEST BREAK SOON", (15, 75), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
             cv2.putText(frame, "User is playing attention. Studying nicely.", (15, 75), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
        cv2.putText(frame, "Controls: Press [q] to Quit", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("AI Study Assistant - Enhanced Face/Mood Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_opencv_test()
