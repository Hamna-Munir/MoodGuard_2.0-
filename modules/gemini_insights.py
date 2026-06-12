import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

MODELS_TO_TRY = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
]

def _get_api_key():
    """Get API key from Streamlit secrets (cloud) or .env (local)"""
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")

def generate_session_insight(summary: dict) -> str:
    api_key = _get_api_key()

    if not api_key:
        return _fallback_insight(summary) + "\n\n_(No API key found — add GEMINI_API_KEY to .env or Streamlit secrets)_"

    prompt = f"""You are a warm, encouraging productivity and wellness coach.

A user just completed a work/study session. Here is their data:
- Duration: {summary.get('duration_min', '?')} minutes
- Emotion breakdown: {summary['emotion_breakdown']}
- Average focus score: {summary['avg_focus']}/100
- Dominant mental state: {summary['dominant_state']}
- Alerts triggered: {summary.get('top_alerts', [])}

Write exactly 3 sentences — no bullet points, no headers:
1. Acknowledge what went well (be specific about their actual data)
2. Point out one area to watch or improve
3. Give one concrete, actionable tip for their next session"""

    last_error = ""
    for model_name in MODELS_TO_TRY:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            last_error = str(e)
            if "404" in last_error or "not found" in last_error.lower():
                continue
            elif "429" in last_error or "QUOTA" in last_error.upper() or "EXHAUSTED" in last_error.upper():
                return _fallback_insight(summary) + "\n\n_(Gemini quota exceeded — showing smart fallback insight)_"
            else:
                continue

    # All models failed
    return _fallback_insight(summary) + f"\n\n_(All Gemini models unavailable — showing fallback insight)_"


def _fallback_insight(summary: dict) -> str:
    """Rule-based insight when Gemini is unavailable"""
    emotions  = list(summary['emotion_breakdown'].keys())
    top_emo   = emotions[0] if emotions else "Neutral"
    focus     = summary['avg_focus']
    state     = summary['dominant_state']
    duration  = summary.get('duration_min', 0)

    # Focus message
    if focus >= 61:
        focus_msg = f"Your average focus of {focus}/100 was strong — you maintained solid attention throughout the {duration}-minute session."
    elif focus >= 31:
        focus_msg = f"Your focus averaged {focus}/100 — moderate attention detected with clear room to improve consistency."
    else:
        focus_msg = f"Your focus score of {focus}/100 suggests significant distraction — try removing interruptions next session."

    # Emotion message
    if top_emo == "Happy":
        emo_msg = "Your positive emotional state is excellent for learning and creative work — keep this energy going."
    elif top_emo == "Neutral":
        emo_msg = "Your calm, neutral state reflects steady, consistent work — a reliable mode for deep tasks."
    elif top_emo in ["Angry", "Fear", "Disgust"]:
        emo_msg = "Some stress or discomfort was detected — short breathing breaks can significantly help regulate focus."
    elif top_emo == "Sad":
        emo_msg = "A lower mood was detected — consider a short walk or some fresh air before your next session."
    else:
        emo_msg = f"Your dominant emotion was {top_emo} — being mindful of your emotional state directly improves performance."

    # Action tip based on state
    if state in ["Distracted", "Low Motivation"]:
        tip = "For your next session, try the Pomodoro technique: 25 minutes focused work, then a 5-minute break."
    elif state in ["Stress Detected", "Discomfort"]:
        tip = "Before your next session, try 2 minutes of box breathing: 4 seconds in, hold 4, out 4, hold 4."
    elif state in ["Highly Productive", "Peak State"]:
        tip = "You were in peak state — schedule your hardest tasks during this same time window next session."
    else:
        tip = "For your next session, eliminate phone notifications and set a clear goal before you start."

    return f"{focus_msg} {emo_msg} {tip}"