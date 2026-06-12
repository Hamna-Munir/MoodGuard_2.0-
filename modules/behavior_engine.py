def analyze(emotion: str, focus_score: int) -> dict:
    e = emotion.lower()

    if e == "happy" and focus_score >= 70:
        return {"state": "Highly Productive",  "tip": "You're in the zone — keep going!",              "alert": False, "color": "#10b981"}
    if e == "happy" and focus_score >= 40:
        return {"state": "Good Mood",           "tip": "Solid focus. Try a Pomodoro timer.",            "alert": False, "color": "#7c5cfc"}
    if e == "neutral" and focus_score >= 60:
        return {"state": "Steady Focus",        "tip": "Consistent work mode. Keep it up!",            "alert": False, "color": "#7c5cfc"}
    if e in ["sad", "fear"] and focus_score < 30:
        return {"state": "Low Motivation",      "tip": "Take a 5-min walk — resets your brain.",       "alert": True,  "color": "#3b82f6"}
    if e == "angry":
        return {"state": "Stress Detected",     "tip": "Box breathing: 4s in, 4s hold, 4s out.",       "alert": True,  "color": "#ef4444"}
    if e == "disgust":
        return {"state": "Discomfort",          "tip": "Step away. Hydrate and stretch.",              "alert": True,  "color": "#f59e0b"}
    if e == "surprise":
        return {"state": "High Alertness",      "tip": "Great engagement! Channel into deep work.",    "alert": False, "color": "#10b981"}
    if focus_score < 30:
        return {"state": "Distracted",          "tip": "Close distracting tabs. Focus music help.",   "alert": True,  "color": "#f59e0b"}
    if focus_score < 60:
        return {"state": "Moderate Focus",      "tip": "Almost there — minimize distractions.",        "alert": False, "color": "#f59e0b"}

    return {"state": "Normal", "tip": "Keep going!", "alert": False, "color": "#7c5cfc"}