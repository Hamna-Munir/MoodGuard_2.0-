import pickle
import numpy as np
import cv2
import mediapipe as mp

# Tumhara RandomForest model load karo
with open("focus_model.pkl", "rb") as f:
    focus_model = pickle.load(f)

mp_face_mesh  = mp.solutions.face_mesh
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def _ear(landmarks, indices, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in indices]
    v1  = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2  = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    hz  = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (v1 + v2) / (2.0 * hz + 1e-6)

class FocusDetector:
    def __init__(self):
        self.mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.blink_count = 0
        self._prev_ear   = 0.3

    def detect(self, frame):
        h, w = frame.shape[:2]
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = self.mesh.process(rgb)

        if not res.multi_face_landmarks:
            return {"focus_score": 0, "state": "No face", "prediction": 0, "blinks": self.blink_count}

        lm        = res.multi_face_landmarks[0].landmark
        left_ear  = _ear(lm, LEFT_EYE,  w, h)
        right_ear = _ear(lm, RIGHT_EYE, w, h)
        avg_ear   = (left_ear + right_ear) / 2.0

        lw = np.linalg.norm(np.array([lm[LEFT_EYE[0]].x  * w, lm[LEFT_EYE[0]].y  * h]) -
                            np.array([lm[LEFT_EYE[3]].x  * w, lm[LEFT_EYE[3]].y  * h]))
        rw = np.linalg.norm(np.array([lm[RIGHT_EYE[0]].x * w, lm[RIGHT_EYE[0]].y * h]) -
                            np.array([lm[RIGHT_EYE[3]].x * w, lm[RIGHT_EYE[3]].y * h]))
        ed = np.linalg.norm(np.array([lm[33].x  * w, lm[33].y  * h]) -
                            np.array([lm[362].x * w, lm[362].y * h]))

        features = np.array([[left_ear, right_ear, avg_ear, lw, rw, ed]])

        if self._prev_ear > 0.25 and avg_ear < 0.20:
            self.blink_count += 1
        self._prev_ear = avg_ear

        prediction  = int(focus_model.predict(features)[0])
        proba       = focus_model.predict_proba(features)[0]
        focus_score = int(proba[1] * 100)
        state       = "Focused" if prediction == 1 else "Distracted"

        return {
            "focus_score": focus_score,
            "state":       state,
            "prediction":  prediction,
            "blinks":      self.blink_count
        }