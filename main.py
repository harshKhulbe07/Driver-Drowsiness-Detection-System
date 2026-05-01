import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
import pygame

# ==========================================
# 1. SETUP: CONFIGURATION & WEIGHTS
# ==========================================
EYE_WEIGHTS_FILE = 'eye_weights.weights.h5' 
YAWN_WEIGHTS_FILE = 'yawn_weights.weights.h5'
ALARM_FILE = 'alarm.wav' 

# TUNED THRESHOLDS
EYE_CLOSED_THRESHOLD = 0.50  
YAWN_THRESHOLD = 0.60
HEAD_DROP_THRESHOLD = -15.0  

HISTORY_LENGTH = 5 

# AUDIO SETUP
pygame.mixer.init()
try:
    alarm_sound = pygame.mixer.Sound(ALARM_FILE)
except:
    print(f"WARNING: Could not find '{ALARM_FILE}'. Make sure it is in the same folder!")
    alarm_sound = None

def build_mobilenet_skeleton():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights=None)
    inputs = tf.keras.Input(shape=(96, 96, 3))
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)

print("Loading AI Brains...")
if not os.path.exists(EYE_WEIGHTS_FILE) or not os.path.exists(YAWN_WEIGHTS_FILE):
    print("ERROR: Missing weight files! Check your filenames in the script.")
    exit()

eye_model = build_mobilenet_skeleton()
eye_model.load_weights(EYE_WEIGHTS_FILE)

yawn_model = build_mobilenet_skeleton()
yawn_model.load_weights(YAWN_WEIGHTS_FILE)
print("Both Models Loaded Successfully!")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

HEAD_ANCHOR_INDICES = [1, 152, 33, 263, 127, 356]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_square_crop(frame, landmarks, indices, w, h, pad_ratio=0.2):
    xs = [landmarks[i].x * w for i in indices]
    ys = [landmarks[i].y * h for i in indices]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
    max_dim = max(x_max - x_min, y_max - y_min)
    pad = int(max_dim * pad_ratio)
    half = (max_dim // 2) + pad
    x1, x2 = max(0, cx - half), min(w, cx + half)
    y1, y2 = max(0, cy - half), min(h, cy + half)

    if x2 > x1 and y2 > y1:
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
    return None, None

def process_eye(frame, landmarks, indices, w, h, model):
    eye_crop, eye_box = get_square_crop(frame, landmarks, indices, w, h, pad_ratio=0.2)
    if eye_crop is not None:
        gray_eye = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
        sim_ir_eye = cv2.cvtColor(gray_eye, cv2.COLOR_GRAY2RGB)
        eye_resized = cv2.resize(sim_ir_eye, (96, 96))
        pred = model.predict(np.expand_dims(eye_resized, axis=0), verbose=0)[0][0]
        return pred, eye_box
    return None, None

def get_head_tilt(landmarks_list, w, h):
    face_2d, face_3d = [], []
    for idx in HEAD_ANCHOR_INDICES:
        lm = landmarks_list[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z])
        
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    
    focal_length = 1 * w
    cam_matrix = np.array([[focal_length, 0, w / 2],
                           [0, focal_length, h / 2],
                           [0, 0, 1]])
    
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, np.zeros((4, 1), dtype=np.float64))
    if not success:
        return None
        
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles[0] * 360

# ==========================================
# 3. MAIN WEBCAM LOOP
# ==========================================
cap = cv2.VideoCapture(0)

# Memory & Logic Variables
eye_history, yawn_history, head_history = [], [], []
fatigue_score = 0.0
consecutive_closed_frames = 0 
frame_count = 0

last_ui = {
    'eye_state': "Tracking...", 'yawn_state': "Tracking...", 'head_state': "UPRIGHT",
    'eye_color': (255,255,255), 'yawn_color': (255,255,255),
    'r_box': None, 'l_box': None, 'm_box': None,
    'eye_val': 0.0, 'yawn_val': 0.0, 'pitch_val': 0.0
}

print("Starting Webcam. Press 'ESC' to exit.")
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame_count += 1
    h, w, _ = frame.shape
    
    # CRITICAL: Reset alert flag every frame
    critical_alert = False

    # --- PERFORMANCE OPTIMIZATION: Heavy AI runs every 2nd frame ---
    if frame_count % 2 == 0:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                # 1. PROCESS HEAD TILT 
                pitch = get_head_tilt(face_landmarks.landmark, w, h)
                if pitch is not None:
                    head_history.append(pitch)
                    if len(head_history) > HISTORY_LENGTH: head_history.pop(0)
                    smooth_pitch = np.mean(head_history)
                    
                    last_ui['pitch_val'] = smooth_pitch
                    if smooth_pitch < HEAD_DROP_THRESHOLD:
                        last_ui['head_state'] = "NODDING OFF"
                    else:
                        last_ui['head_state'] = "UPRIGHT"

                # 2. PROCESS EYES
                right_pred, r_box = process_eye(frame, face_landmarks.landmark, RIGHT_EYE_INDICES, w, h, eye_model)
                left_pred, l_box = process_eye(frame, face_landmarks.landmark, LEFT_EYE_INDICES, w, h, eye_model)
                
                if right_pred is not None and left_pred is not None:
                    avg_eye_pred = (right_pred + left_pred) / 2.0
                    eye_history.append(avg_eye_pred)
                    if len(eye_history) > HISTORY_LENGTH: eye_history.pop(0)
                    smooth_eye_pred = np.mean(eye_history)

                    last_ui['r_box'], last_ui['l_box'] = r_box, l_box
                    last_ui['eye_val'] = smooth_eye_pred

                    if smooth_eye_pred > EYE_CLOSED_THRESHOLD:
                        last_ui['eye_state'], last_ui['eye_color'] = "CLOSED", (0, 0, 255)
                        consecutive_closed_frames += 1
                    else:
                        last_ui['eye_state'], last_ui['eye_color'] = "OPEN", (0, 255, 0)
                        consecutive_closed_frames = 0

                # 3. PROCESS MOUTH
                mouth_crop, mouth_box = get_square_crop(frame, face_landmarks.landmark, MOUTH_INDICES, w, h, pad_ratio=0.2)
                if mouth_crop is not None:
                    mouth_resized = cv2.resize(mouth_crop, (96, 96))
                    yawn_pred = yawn_model.predict(np.expand_dims(mouth_resized, axis=0), verbose=0)[0][0]
                    
                    yawn_history.append(yawn_pred)
                    if len(yawn_history) > HISTORY_LENGTH: yawn_history.pop(0)
                    smooth_yawn_pred = np.mean(yawn_history)

                    last_ui['m_box'] = mouth_box
                    last_ui['yawn_val'] = smooth_yawn_pred

                    if smooth_yawn_pred > YAWN_THRESHOLD:
                        last_ui['yawn_state'], last_ui['yawn_color'] = "YAWNING", (0, 0, 255)
                    else:
                        last_ui['yawn_state'], last_ui['yawn_color'] = "NORMAL", (0, 255, 0)

        # ==========================================
        # 4. FATIGUE ENGINE (FINAL VERSION)
        # ==========================================
        # Safeguard: Only update fatigue if we actually tracked a face recently
        if last_ui['eye_state'] != "Tracking...":
            
            # Absolute override (~2 sec)
            if consecutive_closed_frames >= 30:
                fatigue_score = 10.0
                critical_alert = True
                
            # Micro-sleep (~0.6 sec)
            elif consecutive_closed_frames > 10:
                fatigue_score += 0.3  
                
            # Yawning
            elif last_ui['yawn_state'] == "YAWNING":
                fatigue_score += 0.15 
                
            # Head nodding (gated by eyes)
            elif last_ui['head_state'] == "NODDING OFF" and consecutive_closed_frames > 2:
                fatigue_score += 0.1  
                
            # Cooldown
            else:
                fatigue_score = max(0.0, fatigue_score - 0.3) 
            
            fatigue_score = min(fatigue_score, 10.0)
            
            if fatigue_score >= 7.0:
                critical_alert = True

    # ==========================================
    # 5. DRAW UI AND PLAY ALARM
    # ==========================================
    if last_ui['r_box']:
        x1, y1, x2, y2 = last_ui['r_box']
        cv2.rectangle(frame, (x1, y1), (x2, y2), last_ui['eye_color'], 2)
        
    if last_ui['l_box']:
        x1, y1, x2, y2 = last_ui['l_box']
        cv2.rectangle(frame, (x1, y1), (x2, y2), last_ui['eye_color'], 2)
        cv2.putText(frame, f"Eye: {last_ui['eye_state']} ({last_ui['eye_val']:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, last_ui['eye_color'], 2)

    if last_ui['m_box']:
        x1, y1, x2, y2 = last_ui['m_box']
        cv2.rectangle(frame, (x1, y1), (x2, y2), last_ui['yawn_color'], 2)
        cv2.putText(frame, f"Mouth: {last_ui['yawn_state']} ({last_ui['yawn_val']:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, last_ui['yawn_color'], 2)

    head_color = (0, 0, 255) if last_ui['head_state'] == "NODDING OFF" else (255, 165, 0)
    cv2.putText(frame, f"Head: {last_ui['head_state']} (Pitch: {last_ui['pitch_val']:.1f})", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, head_color, 2)

    cv2.putText(frame, f"Fatigue Score: {fatigue_score:.1f}/10.0", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # AUDIO & VISUAL ALARM TRIGGER
    if critical_alert:
        cv2.putText(frame, "CRITICAL ALARM: WAKE UP!", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
        if alarm_sound and not pygame.mixer.get_busy():
            alarm_sound.play()
    else:
        if alarm_sound and pygame.mixer.get_busy():
            alarm_sound.stop()

    cv2.imshow('Driver Drowsiness System - Master', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()