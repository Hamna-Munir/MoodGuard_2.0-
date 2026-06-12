import numpy as np
import cv2
import onnxruntime as ort

# ONNX model load karo — TensorFlow nahi chahiye ab
session     = ort.InferenceSession("moodguard_model.onnx")
input_name  = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_emotion(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return {
            "emotion":    "Neutral",
            "confidence": 0.0,
            "all_scores": {},
            "face_box":   None
        }

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi   = gray[y:y+h, x:x+w]
    face_roi   = cv2.resize(face_roi, (48, 48))
    face_roi   = face_roi.astype("float32") / 255.0
    face_roi   = np.expand_dims(face_roi, axis=-1)
    face_roi   = np.expand_dims(face_roi, axis=0)

    preds        = session.run([output_name], {input_name: face_roi})[0][0]
    dominant_idx = int(np.argmax(preds))
    confidence   = float(round(preds[dominant_idx] * 100, 2))
    all_scores   = {EMOTIONS[i]: round(float(preds[i]) * 100, 1) for i in range(len(EMOTIONS))}

    return {
        "emotion":    EMOTIONS[dominant_idx],
        "confidence": confidence,
        "all_scores": all_scores,
        "face_box":   (x, y, w, h)
    }