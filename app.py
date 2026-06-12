import cv2, time
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

from modules.emotion_detector import detect_emotion
from modules.focus_detector   import FocusDetector
from modules.behavior_engine  import analyze
from modules.gemini_insights  import generate_session_insight
from utils.logger             import init_log, log_entry, read_log, get_session_summary

st.set_page_config(
    page_title="MoodGuard 2.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
init_log()

if "camera_on" not in st.session_state:
    st.session_state.camera_on = False
if "agent_log" not in st.session_state:
    st.session_state.agent_log = []

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], * { font-family: 'Inter', sans-serif !important; }

/* BG — soft blue-gray */
.stApp { background: #f0f4f8 !important; }
/* 3D Card Shadow */
div[data-testid="stVerticalBlock"] > div > div > div[style*="border-radius"] {
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.07),
                0 10px 15px -3px rgba(0,0,0,0.07),
                0 20px 25px -5px rgba(0,0,0,0.04) !important;
    transform: translateY(0);
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
div[data-testid="stVerticalBlock"] > div > div > div[style*="border-radius"]:hover {
    box-shadow: 0 8px 12px -2px rgba(0,0,0,0.1),
                0 20px 30px -5px rgba(0,0,0,0.1),
                0 30px 40px -8px rgba(0,0,0,0.06) !important;
    transform: translateY(-3px) !important;
}
[data-testid="stAppViewContainer"] { background: #f0f4f8 !important; }
[data-testid="stHeader"] { background: #ffffff !important; border-bottom: 1px solid #e2e8f0 !important; box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important; }
.main .block-container { padding-top: 28px !important; max-width: 1400px; }

/* ── PREMIUM SIDEBAR ── */
/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0f172a 60%, #0a0f1e 100%) !important;
    border-right: 1px solid rgba(56,189,248,0.1) !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.3) !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }


/* Radio as nav items */
[data-testid="stSidebar"] .stRadio > div {
    display: flex !important;
    flex-direction: column !important;
    gap: 4px !important;
    padding: 0 10px !important;
}

/* Hide the radio circle */
[data-testid="stSidebar"] .stRadio input[type="radio"] {
    width: 0 !important;
    height: 0 !important;
    opacity: 0 !important;
    position: absolute !important;
}

/* Each nav item label */
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    padding: 11px 16px !important;
    border-radius: 12px !important;
    cursor: pointer !important;
    border: 1px solid transparent !important;
    color: #94a3b8 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    margin: 0 !important;
    transition: all 0.2s ease !important;
    position: relative !important;
    overflow: hidden !important;
}

/* Hover effect */
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.06) !important;
    color: #e2e8f0 !important;
    border-color: rgba(255,255,255,0.08) !important;
    transform: translateX(4px) !important;
}

/* Active/selected item */
[data-testid="stSidebar"] [aria-checked="true"] + label {
    background: linear-gradient(135deg,rgba(14,165,233,0.2),rgba(56,189,248,0.1)) !important;
    border-color: rgba(56,189,248,0.4) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 
        0 4px 16px rgba(14,165,233,0.2),
        inset 0 1px 0 rgba(255,255,255,0.1) !important;
    transform: translateX(0) !important;
}

/* Active left border glow */
[data-testid="stSidebar"] [aria-checked="true"] + label::before {
    content: '' !important;
    position: absolute !important;
    left: 0 !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    width: 3px !important;
    height: 60% !important;
    background: linear-gradient(180deg, #38bdf8, #0ea5e9) !important;
    border-radius: 0 3px 3px 0 !important;
    box-shadow: 0 0 8px rgba(56,189,248,0.6) !important;
}

/* Hide small circle/dot that appears */
[data-testid="stSidebar"] .stRadio label > div:first-child {
    display: none !important;
}

/* Hide label text that says "label" */
[data-testid="stSidebar"] .stRadio > label {
    display: none !important;
}

/* ── 3D CARDS ── */
@keyframes card-float {
    0%, 100% {
        transform: translateY(0px);
        box-shadow:
            0 4px 6px rgba(0,0,0,0.05),
            0 10px 20px rgba(0,0,0,0.08),
            0 20px 40px rgba(0,0,0,0.04),
            inset 0 1px 0 rgba(255,255,255,0.9);
    }
    50% {
        transform: translateY(-2px);
        box-shadow:
            0 8px 12px rgba(0,0,0,0.07),
            0 16px 32px rgba(0,0,0,0.1),
            0 32px 64px rgba(0,0,0,0.05),
            inset 0 1px 0 rgba(255,255,255,0.9);
    }
}

@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(14,165,233,0.15); }
    50% { box-shadow: 0 0 20px 4px rgba(14,165,233,0.08); }
}

/* White cards — 3D floating */
div[style*="background:#ffffff"],
div[style*="background: #ffffff"] {
    animation: card-float 4s ease-in-out infinite !important;
    border: 1px solid rgba(226,232,240,0.8) !important;
    backdrop-filter: blur(8px) !important;
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    background: linear-gradient(145deg, #ffffff, #f8fafc) !important;
}

div[style*="background:#ffffff"]:hover,
div[style*="background: #ffffff"]:hover {
    transform: translateY(-6px) scale(1.01) !important;
    box-shadow:
        0 16px 32px rgba(0,0,0,0.12),
        0 32px 64px rgba(14,165,233,0.08),
        inset 0 1px 0 rgba(255,255,255,1) !important;
    animation: none !important;
    border-color: rgba(14,165,233,0.2) !important;
}

/* Blue state card — glow pulse */
div[style*="0c4a6e"],
div[style*="0369a1"] {
    animation: pulse-glow 3s ease-in-out infinite !important;
    border: 1px solid rgba(56,189,248,0.3) !important;
    transition: all 0.3s ease !important;
}

div[style*="0c4a6e"]:hover,
div[style*="0369a1"]:hover {
    transform: translateY(-4px) !important;
    animation: none !important;
    box-shadow: 0 20px 40px rgba(3,105,161,0.35) !important;
}

/* Stat cards top colored */
div[style*="position:relative;overflow:hidden"] {
    animation: card-float 4s ease-in-out infinite !important;
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
}
div[style*="position:relative;overflow:hidden"]:hover {
    transform: translateY(-8px) scale(1.02) !important;
    animation: none !important;
}

/* Metrics */
[data-testid="stMetric"] {
    animation: card-float 4s ease-in-out infinite !important;
    background: linear-gradient(145deg, #ffffff, #f8fafc) !important;
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-6px) scale(1.02) !important;
    animation: none !important;
}

/* TEXT */
/* Main content text — dark */
.main p { color: #334155 !important; }
.main label { color: #334155 !important; }

/* Plotly chart text — force dark */
.js-plotly-plot text { fill: #1e293b !important; }
.js-plotly-plot .gtitle { fill: #0f172a !important; }
.js-plotly-plot .xtick text { fill: #334155 !important; }
.js-plotly-plot .ytick text { fill: #334155 !important; }
.js-plotly-plot .legendtext { fill: #334155 !important; }
.js-plotly-plot .g-gtitle text { fill: #0f172a !important; }
.js-plotly-plot .xaxislayer-above text { fill: #334155 !important; }
.js-plotly-plot .yaxislayer-above text { fill: #334155 !important; }

h1 { color: #0f172a !important; font-size: 26px !important; font-weight: 800 !important; }
h2 { color: #0f172a !important; font-size: 20px !important; font-weight: 700 !important; }
h3 { color: #1e293b !important; font-size: 17px !important; font-weight: 600 !important; }

/* METRICS */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 20px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stMetricValue"] { color: #0ea5e9 !important; font-size: 30px !important; font-weight: 800 !important; }

/* BUTTONS — Floating Creative Style */
.stButton > button {
    background: linear-gradient(135deg, #7dd3fc, #38bdf8, #0ea5e9) !important;
    color: #0c4a6e !important;
    border: none !important;
    border-radius: 14px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    padding: 12px 24px !important;
    width: 100% !important;
    letter-spacing: 0.02em !important;
    box-shadow:
        0 4px 8px rgba(14,165,233,0.25),
        0 8px 16px rgba(14,165,233,0.15),
        inset 0 1px 0 rgba(255,255,255,0.4) !important;
    transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    animation: float-btn 3s ease-in-out infinite !important;
    position: relative !important;
}

@keyframes float-btn {
    0%, 100% {
        transform: translateY(0px);
        box-shadow:
            0 4px 8px rgba(14,165,233,0.25),
            0 8px 16px rgba(14,165,233,0.15),
            inset 0 1px 0 rgba(255,255,255,0.4);
    }
    50% {
        transform: translateY(-4px);
        box-shadow:
            0 8px 16px rgba(14,165,233,0.3),
            0 16px 32px rgba(14,165,233,0.2),
            inset 0 1px 0 rgba(255,255,255,0.4);
    }
}

.stButton > button:hover {
    transform: translateY(-6px) scale(1.02) !important;
    box-shadow:
        0 12px 24px rgba(14,165,233,0.35),
        0 24px 48px rgba(14,165,233,0.2),
        inset 0 1px 0 rgba(255,255,255,0.5) !important;
    animation: none !important;
    color: #082f49 !important;
}

.stButton > button:active {
    transform: translateY(-2px) scale(0.99) !important;
    box-shadow:
        0 4px 8px rgba(14,165,233,0.2),
        inset 0 1px 0 rgba(255,255,255,0.3) !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 2px dashed #bfdbfe !important;
    border-radius: 14px !important;
    padding: 20px !important;
}

/* ALERTS */
.stSuccess { background: rgba(16,185,129,0.08) !important; border: 1px solid rgba(16,185,129,0.25) !important; border-radius: 10px !important; }
.stSuccess p { color: #047857 !important; font-size: 15px !important; font-weight: 500 !important; }
.stWarning { background: rgba(245,158,11,0.08) !important; border: 1px solid rgba(245,158,11,0.25) !important; border-radius: 10px !important; }
.stWarning p { color: #b45309 !important; font-size: 15px !important; font-weight: 500 !important; }
.stInfo { background: rgba(14,165,233,0.08) !important; border: 1px solid rgba(14,165,233,0.25) !important; border-radius: 10px !important; }
.stInfo p { color: #0369a1 !important; font-size: 15px !important; }
.stError { background: rgba(239,68,68,0.08) !important; border: 1px solid rgba(239,68,68,0.25) !important; border-radius: 10px !important; }
.stError p { color: #dc2626 !important; font-size: 15px !important; }

/* DATAFRAME */
[data-testid="stDataFrame"] { border-radius: 12px !important; border: 1px solid #e2e8f0 !important; }

/* CAPTION */
.stCaption p { color: #64748b !important; font-size: 14px !important; }

/* SCROLLBAR */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f0f4f8; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
/* Blue cards text white */
div[style*="0c4a6e"] *, div[style*="0369a1"] *, div[style*="0284c7"] * {
    color: #ffffff !important;
}
div[style*="linear-gradient(145deg,#0c4a6e"] p,
div[style*="linear-gradient(145deg,#0c4a6e"] span,
div[style*="linear-gradient(145deg,#0c4a6e"] div {
    color: #ffffff !important;
}
/* Force white text in all blue/dark gradient cards */
div[style*="0c4a6e"] div,
div[style*="0c4a6e"] span,
div[style*="0369a1"] div,
div[style*="0369a1"] span,
div[style*="0284c7"] div,
div[style*="0284c7"] span {
    color: #ffffff !important;
}
/* NUCLEAR WHITE TEXT FIX */
[style*="0c4a6e"] { color: white !important; }
[style*="0369a1"] { color: white !important; }
[style*="0284c7"] { color: white !important; }
[style*="0c4a6e"] * { color: white !important; }
[style*="0369a1"] * { color: white !important; }
[style*="0284c7"] * { color: white !important; }
@keyframes emoji-float {
    0%, 100% {
        transform: perspective(200px) rotateX(5deg) translateY(0px);
        filter: drop-shadow(0 8px 16px rgba(0,0,0,0.3));
    }
    50% {
        transform: perspective(200px) rotateX(5deg) translateY(-8px);
        filter: drop-shadow(0 16px 24px rgba(0,0,0,0.4));
    }
}

</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────
EMOTION_COLORS = {
    "Happy":    "#0ea5e9",
    "Neutral":  "#94a3b8",
    "Sad":      "#818cf8",
    "Angry":    "#f87171",
    "Fear":     "#fb923c",
    "Surprise": "#34d399",
    "Disgust":  "#f472b6"
}
EMOJI_MAP = {
    "Happy":"😊","Sad":"😢","Angry":"😠",
    "Neutral":"😐","Fear":"😨","Surprise":"😲","Disgust":"🤢"
}

# WebRTC Config
RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Video Processor
class MoodGuardProcessor(VideoProcessorBase):
    def __init__(self):
        self.fd     = FocusDetector()
        self.emotion      = "Neutral"
        self.confidence   = 0.0
        self.focus_score  = 0
        self.focus_state  = "—"
        self.state_label  = "Waiting..."
        self.all_scores   = {}
        self.blinks       = 0
        self.alert        = False
        self.tip          = ""

    def recv(self, frame):
        img   = frame.to_ndarray(format="bgr24")
        emo   = detect_emotion(img)
        focus = self.fd.detect(img)
        state = analyze(emo["emotion"], focus["focus_score"])

        # Save results
        self.emotion     = emo["emotion"]
        self.confidence  = emo["confidence"]
        self.focus_score = focus["focus_score"]
        self.focus_state = focus["state"]
        self.state_label = state["state"]
        self.all_scores  = emo.get("all_scores", {})
        self.blinks      = focus["blinks"]
        self.alert       = state["alert"]
        self.tip         = state["tip"]

        # Draw on frame
        if emo["face_box"]:
            x,y,w,h = emo["face_box"]
            cv2.rectangle(img,(x,y),(x+w,y+h),(14,165,233),2)
            cv2.putText(img,f"{emo['emotion']} {emo['confidence']:.0f}%",
                       (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(14,165,233),2)
            cv2.putText(img,f"Focus:{focus['focus_score']} {focus['state']}",
                       (x,y+h+22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(5,150,105),2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:24px 24px 18px;border-bottom:1px solid rgba(56,189,248,0.1);'>
        <div style='display:flex;align-items:center;gap:10px;'>
            <div style='width:36px;height:36px;background:linear-gradient(135deg,#0ea5e9,#38bdf8);
                border-radius:10px;display:flex;align-items:center;justify-content:center;
                font-size:18px;box-shadow:0 4px 12px rgba(14,165,233,0.4);'>🧠</div>
            <div style='font-size:20px;font-weight:800;color:#f1f5f9;'>
                Mood<span style='color:#38bdf8;'>Guard</span>
            </div>
        </div>
        <div style='font-size:9px;font-weight:700;letter-spacing:0.2em;color:#475569;
            text-transform:uppercase;margin-top:6px;margin-left:46px;'>
            v2.0 · AI Mental State Monitor
        </div>
    </div>
    <div style='padding:16px 24px 4px;font-size:9px;font-weight:700;
        letter-spacing:0.2em;color:#334155;text-transform:uppercase;'>
        Navigation
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🧠  Dashboard",
        "📷  Photo Analysis",
        "🎥  Live Camera",
        "📊  Analytics",
        "✦   AI Insights",
        "🤖  Agent Log",
        "📁  History",
        "ℹ   About"
    ], label_visibility="collapsed")

    st.markdown("""
    <div style='margin:16px 12px 0;background:rgba(14,165,233,0.07);
        border:1px solid rgba(56,189,248,0.15);border-radius:12px;padding:14px;'>
        <div style='display:flex;align-items:center;gap:7px;margin-bottom:10px;'>
            <div style='width:8px;height:8px;background:#4ade80;border-radius:50%;
                box-shadow:0 0 6px rgba(74,222,128,0.6);'></div>
            <span style='font-size:12px;font-weight:700;color:#e2e8f0;'>System Ready</span>
        </div>
        <div style='font-size:11px;color:#64748b;line-height:1.8;'>
            ONNX Model · 66.6% accuracy<br>
            MediaPipe · RandomForest<br>
            Gemini 1.5 Flash
        </div>
    </div>
    <div style='margin:14px 12px 0;padding:14px;border-top:1px solid rgba(255,255,255,0.05);'>
        <div style='font-size:9px;color:#334155;text-transform:uppercase;
            letter-spacing:0.15em;margin-bottom:8px;'>Built by</div>
        <div style='display:flex;align-items:center;gap:9px;'>
            <div style='width:34px;height:34px;border-radius:50%;
                background:linear-gradient(135deg,#0ea5e9,#6366f1);
                display:flex;align-items:center;justify-content:center;
                font-size:13px;font-weight:800;color:white;flex-shrink:0;'>H</div>
            <div>
                <div style='font-size:13px;font-weight:700;color:#f1f5f9;'>Hamna Munir</div>
                <div style='font-size:10px;color:#475569;'>AI/ML Engineer</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
# ── HELPER FUNCTIONS ──────────────────────────────────────────────
def section_header(title, subtitle=""):
    sub = f"<div style='font-size:14px;color:#64748b;margin-top:5px;'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"""
    <div style='margin-bottom:24px;'>
        <div style='font-size:24px;font-weight:800;color:#0f172a;'>{title}</div>
        {sub}
    </div>""", unsafe_allow_html=True)

def white_card(content, extra_style=""):
    return f"""
    <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;
        padding:22px 24px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,0.04);{extra_style}'>
        {content}
    </div>"""

def card_title_html(title, color="#0ea5e9"):
    return f"""
    <div style='font-size:11px;font-weight:700;letter-spacing:0.12em;
        text-transform:uppercase;color:{color};margin-bottom:16px;
        display:flex;align-items:center;gap:8px;'>
        <div style='width:6px;height:6px;background:{color};border-radius:50%;flex-shrink:0;'></div>
        {title}
    </div>"""

def emotion_bars_html(all_scores):
    html = ""
    for emo, score in sorted(all_scores.items(), key=lambda x: -x[1]):
        color = EMOTION_COLORS.get(emo, "#0ea5e9")
        html += f"""
        <div style='display:flex;align-items:center;gap:12px;margin-bottom:11px;'>
            <div style='width:68px;font-size:13px;font-weight:500;color:#334155;'>{emo}</div>
            <div style='flex:1;height:8px;background:#f1f5f9;border-radius:4px;overflow:hidden;'>
                <div style='width:{score}%;height:100%;background:{color};border-radius:4px;'></div>
            </div>
            <div style='width:40px;font-size:13px;font-weight:600;color:#0f172a;text-align:right;'>
                {score:.0f}%
            </div>
        </div>"""
    return html

def state_card_html(emotion, confidence, focus_score, focus_state, blinks, state_label):
    emoji = EMOJI_MAP.get(emotion, "😐")
    if focus_score >= 61:   fc, fl = "#4ade80", "Focused"
    elif focus_score >= 31: fc, fl = "#fbbf24", "Moderate"
    else:                   fc, fl = "#93c5fd", "Distracted"
    return f"""
    <div style='background:linear-gradient(145deg,#0c4a6e,#0369a1);
        border-radius:16px;padding:24px;margin-bottom:16px;
        box-shadow:0 8px 24px rgba(3,105,161,0.25);'>
        <div style='font-size:10px;font-weight:700;letter-spacing:0.16em;
            text-transform:uppercase;color:#bae6fd;margin-bottom:14px;'>
            Current Mental State
        </div>
        <div style='font-size:80px;margin-bottom:10px;
            filter: drop-shadow(0 8px 16px rgba(0,0,0,0.3));
            transform: perspective(200px) rotateX(5deg);
            display:inline-block;
            animation: emoji-float 3s ease-in-out infinite;'>{emoji}</div>        <div style='font-size:26px;font-weight:800;color:#ffffff;letter-spacing:-0.02em;'>
            {emotion.upper()}
        </div>
        <div style='font-size:13px;color:#bae6fd;margin-top:6px;'>
            Confidence: <span style='color:#ffffff;font-weight:700;'>{confidence:.1f}%</span>
            &nbsp;·&nbsp; <span style='color:#e0f2fe;'>{state_label}</span>
        </div>
        <div style='height:1px;background:rgba(255,255,255,0.15);margin:18px 0;'></div>
        <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;'>
            <div style='background:rgba(255,255,255,0.1);border-radius:10px;padding:12px;'>
                <div style='font-size:10px;color:#bae6fd;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:5px;'>Focus</div>
                <div style='font-size:22px;font-weight:800;color:#ffffff;'>{focus_score}</div>
                <div style='font-size:11px;color:#93c5fd;margin-top:2px;'>{fl}</div>
            </div>
            <div style='background:rgba(255,255,255,0.1);border-radius:10px;padding:12px;'>
                <div style='font-size:10px;color:#bae6fd;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:5px;'>State</div>
                <div style='font-size:13px;font-weight:700;color:#ffffff;margin-top:6px;'>
                    {state_label}
                </div>
            </div>
            <div style='background:rgba(255,255,255,0.1);border-radius:10px;padding:12px;'>
                <div style='font-size:10px;color:#bae6fd;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:5px;'>Blinks</div>
                <div style='font-size:22px;font-weight:800;color:#ffffff;'>{blinks}</div>
                <div style='font-size:11px;color:#93c5fd;margin-top:2px;'>session</div>
            </div>
        </div>
    </div>"""

def focus_meter_html(score):
    if score >= 61:   color, label, desc = "#059669", "Focused",    "Strong eye contact detected"
    elif score >= 31: color, label, desc = "#d97706", "Moderate",   "Moderate attention level"
    else:             color, label, desc = "#dc2626", "Distracted",  "Attention drifting"
    return f"""
    <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;
        padding:20px 22px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,0.04);'>
        {card_title_html("Focus & Attention Score")}
        <div style='display:flex;align-items:center;gap:20px;'>
            <div style='text-align:center;'>
                <div style='font-size:48px;font-weight:800;color:{color};line-height:1;'>{score}</div>
                <div style='font-size:12px;color:#64748b;margin-top:4px;'>/ 100</div>
            </div>
            <div style='flex:1;'>
                <div style='width:100%;height:10px;background:#f1f5f9;
                    border-radius:5px;overflow:hidden;margin-bottom:10px;'>
                    <div style='width:{score}%;height:100%;background:{color};border-radius:5px;'></div>
                </div>
                <div style='font-size:15px;font-weight:700;color:{color};'>{label}</div>
                <div style='font-size:13px;color:#64748b;margin-top:4px;'>{desc}</div>
            </div>
        </div>
    </div>"""

def empty_state_html(icon, title, subtitle):
    st.markdown(f"""
    <div style='text-align:center;padding:80px 40px;'>
        <div style='font-size:56px;margin-bottom:16px;'>{icon}</div>
        <div style='font-size:18px;font-weight:700;color:#1e293b;margin-bottom:8px;'>{title}</div>
        <div style='font-size:14px;color:#64748b;'>{subtitle}</div>
    </div>""", unsafe_allow_html=True)

CHART_LAYOUT = dict(
    paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
    font=dict(family="Inter", color="#334155", size=13),
    title_font=dict(color="#0f172a", size=15, family="Inter"),
    margin=dict(t=50, b=30, l=20, r=20)
)

PAGE_INFO = {
    "🏠  Dashboard":      ("🧠 Dashboard",      "Real-time emotion · focus · behavioral intelligence"),
    "📷  Photo Analysis": ("📷 Photo Analysis", "Upload any photo for instant emotion detection"),
    "🎥  Live Camera":    ("🎥 Live Camera",     "Dedicated real-time analysis view"),
    "📊  Analytics":      ("📊 Analytics",       "Session charts, trends & emotion timeline"),
    "✦   AI Insights":    ("✦ AI Insights",      "Google Gemini 1.5 Flash · Personalized advice"),
    "🤖  Agent Log":      ("🤖 Agent Log",       "Smart recommendations & behavioral decisions"),
    "📁  History":        ("📁 Session History", "All recorded sessions with CSV export"),
    "   About":          (" About",            "MoodGuard 2.0 · AI Mental State Intelligence Platform"),
}
title, subtitle = PAGE_INFO.get(page, ("MoodGuard 2.0", ""))
section_header(title, subtitle)

# ══════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════
if page == "🧠  Dashboard":

    # Banner
    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(14,165,233,0.08),rgba(56,189,248,0.05));
        border:1px solid rgba(14,165,233,0.2);border-radius:12px;
        padding:13px 20px;font-size:14px;color:#0369a1;margin-bottom:20px;
        display:flex;align-items:center;gap:10px;font-weight:500;'>
        <span style='font-size:16px;'>✅</span>
        MoodGuard 2.0 is ready ·
        Emotion Detection + Focus Analysis + Behavioral Intelligence + AI Insights
    </div>""", unsafe_allow_html=True)

    # 4 stat cards
    c1,c2,c3,c4 = st.columns(4)
    stats = [
        (c1,"WELLNESS SCORE","72%","#0ea5e9","▲ +5%","vs last session"),
        (c2,"STRESS LEVEL",  "18%","#f87171","▼ −3%","improving"),
        (c3,"POSITIVE MOOD", "64%","#34d399","▲ +8%","this week"),
        (c4,"DOMINANT MOOD", "😊",  "#fb923c","Happy","68% confidence"),
    ]
    for col,label,val,color,trend,sub in stats:
        with col:
            st.markdown(f"""
            <div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;
                padding:22px;box-shadow:0 2px 8px rgba(0,0,0,0.04);
                position:relative;overflow:hidden;'>
                <div style='position:absolute;top:0;left:0;right:0;height:3px;background:{color};
                    border-radius:16px 16px 0 0;'></div>
                <div style='font-size:10px;font-weight:700;letter-spacing:0.12em;
                    text-transform:uppercase;color:#94a3b8;margin-bottom:10px;'>{label}</div>
                <div style='font-size:34px;font-weight:800;color:{color};line-height:1;'>{val}</div>
                <div style='font-size:13px;color:#94a3b8;margin-top:10px;'>
                    <span style='color:#059669;font-weight:600;'>{trend}</span> {sub}
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown(white_card(
            card_title_html("Real-Time Emotion Analysis") +
            "<div id='cam-area'></div>"
        ), unsafe_allow_html=True)

        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("▶  Start Camera", key="start_d"):
                st.session_state.camera_on = True
        with bc2:
            if st.button("⏹  Stop Camera", key="stop_d"):
                st.session_state.camera_on = False

        frame_slot = st.empty()
        if not st.session_state.camera_on:
            frame_slot.markdown("""
            <div style='background:#f8fafc;border:2px dashed #e2e8f0;border-radius:12px;
                height:260px;display:flex;flex-direction:column;align-items:center;
                justify-content:center;gap:12px;margin-top:12px;'>
                <div style='font-size:44px;'>📷</div>
                <div style='font-size:15px;font-weight:600;color:#94a3b8;'>
                    Press Start Camera to begin
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(white_card(
            card_title_html("Emotion Probabilities") +
            "<div id='emo-area'></div>"
        ), unsafe_allow_html=True)
        emo_slot = st.empty()
        emo_slot.markdown(
            "<div style='color:#94a3b8;font-size:14px;padding:4px 0;'>Start camera to see live probabilities</div>",
            unsafe_allow_html=True
        )

    with col_r:
        state_slot = st.empty()
        state_slot.markdown(state_card_html("Neutral",0,0,"—",0,"Waiting..."), unsafe_allow_html=True)
        focus_slot = st.empty()
        focus_slot.markdown(focus_meter_html(0), unsafe_allow_html=True)

        st.markdown(white_card(
            card_title_html("Mental Health Tips", "#059669") + """
            <div style='display:flex;flex-direction:column;gap:9px;'>
                <div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
                    padding:10px 14px;font-size:13px;color:#166534;'>
                    ⚡ Great energy! Perfect time to focus on goals.
                </div>
                <div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
                    padding:10px 14px;font-size:13px;color:#1d4ed8;'>
                    🎯 Channel positivity into creative tasks.
                </div>
                <div style='background:#fef3c7;border:1px solid #fde68a;border-radius:10px;
                    padding:10px 14px;font-size:13px;color:#92400e;'>
                    💬 Share your good mood with someone today.
                </div>
            </div>"""
        ), unsafe_allow_html=True)

        tip_slot = st.empty()

    if st.session_state.camera_on:
        fd       = FocusDetector()
        cap      = cv2.VideoCapture(0)
        last_log = time.time()

        if not cap.isOpened():
            st.error("❌ Webcam nahi mila.")
            st.session_state.camera_on = False
        else:
            while st.session_state.camera_on:
                ret, frame = cap.read()
                if not ret:
                    break

                emo   = detect_emotion(frame)
                focus = fd.detect(frame)
                state = analyze(emo["emotion"], focus["focus_score"])

                if emo["face_box"]:
                    x,y,w,h = emo["face_box"]
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(14,165,233),2)
                    cv2.putText(frame,f"{emo['emotion']} {emo['confidence']:.0f}%",
                               (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(14,165,233),2)
                    cv2.putText(frame,f"Focus:{focus['focus_score']} {focus['state']}",
                               (x,y+h+22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(5,150,105),2)

                frame_slot.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB), use_column_width=True)
                state_slot.markdown(
                    state_card_html(emo["emotion"],emo["confidence"],
                                   focus["focus_score"],focus["state"],
                                   focus["blinks"],state["state"]),
                    unsafe_allow_html=True
                )
                focus_slot.markdown(focus_meter_html(focus["focus_score"]), unsafe_allow_html=True)

                if emo["all_scores"]:
                    emo_slot.markdown(
                        f"<div style='padding:4px 0;'>{emotion_bars_html(emo['all_scores'])}</div>",
                        unsafe_allow_html=True
                    )

                if state["alert"]:
                    tip_slot.warning(f"⚠️  {state['tip']}")
                    if not st.session_state.agent_log or st.session_state.agent_log[-1]["tip"] != state["tip"]:
                        st.session_state.agent_log.append({
                            "time":  time.strftime("%H:%M:%S"),
                            "state": state["state"],
                            "tip":   state["tip"]
                        })
                else:
                    tip_slot.success(f"💡  {state['tip']}")

                if time.time()-last_log >= 5:
                    log_entry(emo["emotion"],emo["confidence"],
                             focus["focus_score"],state["state"],
                             state["alert"],state["tip"])
                    last_log = time.time()

                time.sleep(0.04)

            cap.release()

# ══════════════════════════════════════════════════════════════════
# PHOTO ANALYSIS
# ══════════════════════════════════════════════════════════════════
elif page == "📷  Photo Analysis":
    uploaded = st.file_uploader(
        "Upload a photo for instant emotion analysis",
        type=["jpg","jpeg","png"]
    )

    if not uploaded:
        empty_state_html("📷","No photo uploaded","Drop any JPG or PNG to analyze emotion instantly")
    else:
        img    = Image.open(uploaded)
        frame  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        result = detect_emotion(frame)
        bstate = analyze(result["emotion"], 50)

        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown(white_card(card_title_html("Uploaded Photo")), unsafe_allow_html=True)
            st.image(img, use_column_width=True)

        with col2:
            emoji = EMOJI_MAP.get(result["emotion"], "😐")
            color = EMOTION_COLORS.get(result["emotion"], "#0ea5e9")
            st.markdown(f"""
            <div style='background:linear-gradient(145deg,#0c4a6e,#0369a1);
                border-radius:16px;padding:28px;text-align:center;margin-bottom:16px;
                box-shadow:0 8px 24px rgba(3,105,161,0.2);'>
                <div style='font-size:10px;font-weight:700;letter-spacing:0.16em;
                    text-transform:uppercase;color:rgba(255,255,255,0.4);margin-bottom:14px;'>
                    Detected Emotion
                </div>
                <div style='font-size:90px;margin-bottom:12px;
                    filter: drop-shadow(0 10px 20px rgba(0,0,0,0.35));
                    display:inline-block;
                    animation: emoji-float 3s ease-in-out infinite;'>{emoji}</div>
                <div style='font-size:30px;font-weight:800;color:#ffffff;'>{result["emotion"].upper()}</div>
                <div style='font-size:14px;color:rgba(255,255,255,0.5);margin-top:8px;'>
                    Confidence:
                    <span style='color:#7dd3fc;font-weight:700;'>{result["confidence"]:.1f}%</span>
                </div>
                <div style='margin-top:16px;background:rgba(255,255,255,0.08);
                    border-radius:10px;padding:12px 16px;'>
                    <div style='font-size:13px;color:rgba(255,255,255,0.75);'>💡 {bstate["tip"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if result["all_scores"]:
                st.markdown(
                    white_card(card_title_html("Emotion Probabilities") + emotion_bars_html(result["all_scores"])),
                    unsafe_allow_html=True
                )

# ══════════════════════════════════════════════════════════════════
# LIVE CAMERA
# ══════════════════════════════════════════════════════════════════
elif page == "🎥  Live Camera":
    col_cam, col_side = st.columns([3,2])

    with col_cam:
        st.markdown(
            white_card(card_title_html("Live Feed") +
            "<div style='font-size:13px;color:#64748b;margin-bottom:12px;'>"
            "Click START — camera will open in browser</div>"),
            unsafe_allow_html=True
        )

        ctx = webrtc_streamer(
            key="moodguard-live",
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=MoodGuardProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_side:
        state2_slot = st.empty()
        focus2_slot = st.empty()
        emo2_slot   = st.empty()
        tip2_slot   = st.empty()

        state2_slot.markdown(
            state_card_html("Neutral",0,0,"—",0,"Press START"),
            unsafe_allow_html=True
        )
        focus2_slot.markdown(focus_meter_html(0), unsafe_allow_html=True)

    # Live update loop
    if ctx.video_processor:
        last_log = time.time()
        while True:
            p = ctx.video_processor

            state2_slot.markdown(
                state_card_html(p.emotion, p.confidence,
                               p.focus_score, p.focus_state,
                               p.blinks, p.state_label),
                unsafe_allow_html=True
            )
            focus2_slot.markdown(focus_meter_html(p.focus_score), unsafe_allow_html=True)

            if p.all_scores:
                emo2_slot.markdown(
                    white_card(card_title_html("Emotion Probabilities") +
                              emotion_bars_html(p.all_scores)),
                    unsafe_allow_html=True
                )

            if p.alert:
                tip2_slot.warning(f"⚠️  {p.tip}")
            elif p.tip:
                tip2_slot.success(f"💡  {p.tip}")

            if time.time() - last_log >= 5 and p.emotion != "Neutral":
                log_entry(p.emotion, p.confidence,
                         p.focus_score, p.state_label,
                         p.alert, p.tip)
                last_log = time.time()

            time.sleep(0.1)
# ══════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════
elif page == "📊  Analytics":
    df = read_log()
    if df.empty:
        empty_state_html("📊","No session data yet","Run a live session from Dashboard or Live Camera first")
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Readings",  len(df))
        c2.metric("Avg Focus",       f"{df['focus_score'].mean():.1f}")
        c3.metric("Top Emotion",     df["emotion"].mode()[0])
        c4.metric("Alerts Fired",    int(df["alert"].sum()))

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            emo_df = df["emotion"].value_counts().reset_index()
            emo_df.columns = ["emotion","count"]
            fig = px.pie(emo_df, names="emotion", values="count",
                        title="Emotion Distribution", hole=0.4,
                        color="emotion", color_discrete_map=EMOTION_COLORS)
            fig.update_layout(**CHART_LAYOUT)
            fig.update_traces(textfont_size=13)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            fig2 = px.line(df, x="timestamp", y="focus_score",
                          title="Focus Score Timeline",
                          color_discrete_sequence=["#0ea5e9"])
            fig2.add_hrect(y0=61,y1=100,fillcolor="#059669",opacity=0.06,line_width=0)
            fig2.add_hrect(y0=0, y1=30, fillcolor="#dc2626",opacity=0.06,line_width=0)
            fig2.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

       # Emotion Timeline Strip
        HEIGHTS = {"Happy":100,"Neutral":65,"Sad":55,"Angry":80,"Fear":62,"Surprise":75,"Disgust":60}
        emotions_list = df["emotion"].tolist()[-60:]
        total = len(emotions_list)
        strip_bars = ""
        for i, e in enumerate(emotions_list):
            h     = HEIGHTS.get(e, 60)
            color = EMOTION_COLORS.get(e, "#0ea5e9")
            op    = 0.4 if i < total - 6 else 1
            strip_bars += f"<div style='flex:1;height:{h}%;background:{color};opacity:{op};border-radius:3px 3px 0 0;' title='{e}'></div>"

        strip_html = (
            white_card(
                card_title_html("Emotion Timeline — Last 60 Readings") +
                "<div style='display:flex;gap:3px;align-items:flex-end;height:60px;'>" +
                strip_bars +
                "</div>"
            )
        )
        st.markdown(strip_html, unsafe_allow_html=True)

        state_df = df["state"].value_counts().reset_index()
        state_df.columns = ["state","count"]
        fig3 = px.bar(state_df, x="state", y="count",
                     title="Mental States Frequency",
                     color="state",
                     color_discrete_sequence=["#0ea5e9","#059669","#f87171","#fbbf24","#818cf8","#f472b6"])
        fig3.update_layout(**CHART_LAYOUT, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# AI INSIGHTS
# ══════════════════════════════════════════════════════════════════
elif page == "✦   AI Insights":
    df = read_log()
    if df.empty:
        empty_state_html("✦","No session data yet","Run a live session first to generate AI insights")
    else:
        summary = get_session_summary(df)
        c1,c2,c3 = st.columns(3)
        c1.metric("Dominant State", summary["dominant_state"])
        c2.metric("Avg Focus",      f"{summary['avg_focus']}/100")
        c3.metric("Duration",       f"{summary['duration_min']} min")

        st.markdown("""
        <div style='background:linear-gradient(135deg,rgba(14,165,233,0.08),rgba(56,189,248,0.04));
            border:1px solid rgba(14,165,233,0.2);border-radius:14px;padding:22px;margin:20px 0;'>
            <div style='display:flex;align-items:center;gap:12px;margin-bottom:10px;'>
                <div style='width:36px;height:36px;background:rgba(14,165,233,0.15);
                    border:1px solid rgba(14,165,233,0.3);border-radius:10px;
                    display:flex;align-items:center;justify-content:center;font-size:16px;'>✦</div>
                <div>
                    <div style='font-size:15px;font-weight:700;color:#0f172a;'>AI Insight Generator</div>
                    <div style='font-size:12px;color:#64748b;'>Google Gemini 1.5 Flash</div>
                </div>
            </div>
            <div style='font-size:14px;color:#475569;line-height:1.7;'>
                Gemini will analyze your emotion patterns, focus trends, and behavioral states
                to generate personalized productivity and wellness advice.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✨  Generate AI Insight", type="primary"):
            with st.spinner("Gemini is analyzing your session..."):
                try:
                    insight = generate_session_insight(summary)
                    st.markdown(f"""
                    <div style='background:#ffffff;border:1px solid #e2e8f0;
                        border-left:4px solid #0ea5e9;border-radius:14px;
                        padding:24px 28px;box-shadow:0 4px 12px rgba(14,165,233,0.1);'>
                        <div style='font-size:11px;font-weight:700;letter-spacing:0.12em;
                            text-transform:uppercase;color:#0ea5e9;margin-bottom:14px;'>
                            ✦ AI Generated Insight
                        </div>
                        <div style='font-size:15px;color:#1e293b;line-height:1.9;'>{insight}</div>
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Gemini Error: {str(e)[:200]}\n\nCheck .env mein GEMINI_API_KEY aur free tier quota.")

# ══════════════════════════════════════════════════════════════════
# AGENT LOG
# ══════════════════════════════════════════════════════════════════
elif page == "🤖  Agent Log":
    col1, col2 = st.columns(2)

    with col1:
        rules_html = card_title_html("Agent Rules")
        rules = [
            ("#f87171","rgba(248,113,113,0.1)","⚠","Stress Detected",    "emotion == Angry → breathing exercise"),
            ("#fb923c","rgba(251,146,60,0.1)", "↓","Low Focus",          "focus < 30 → Pomodoro technique"),
            ("#818cf8","rgba(129,140,248,0.1)","😢","Low Motivation",     "sad + focus < 30 → take a walk"),
            ("#34d399","rgba(52,211,153,0.1)", "✓","Peak State",         "happy + focus > 70 → deep work"),
            ("#0ea5e9","rgba(14,165,233,0.1)", "→","Steady Focus",       "neutral + focus > 60 → keep going"),
        ]
        items_html = ""
        for color,bg,icon,state_lbl,rule in rules:
            items_html += f"""
            <div style='display:flex;align-items:center;gap:12px;padding:13px 0;
                border-bottom:1px solid #f1f5f9;'>
                <div style='width:34px;height:34px;background:{bg};border-radius:9px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:14px;color:{color};flex-shrink:0;'>{icon}</div>
                <div>
                    <div style='font-size:14px;font-weight:600;color:#0f172a;'>{state_lbl}</div>
                    <div style='font-size:12px;color:#64748b;margin-top:2px;'>{rule}</div>
                </div>
            </div>"""
        st.markdown(white_card(rules_html + items_html), unsafe_allow_html=True)

    with col2:
        actions_html = card_title_html("Recent Agent Actions")
        if not st.session_state.agent_log:
            actions_html += "<div style='color:#94a3b8;font-size:14px;padding:20px 0;text-align:center;'>No actions yet.<br>Start a live session.</div>"
        else:
            for entry in reversed(st.session_state.agent_log[-10:]):
                actions_html += f"""
                <div style='display:flex;align-items:flex-start;gap:12px;padding:12px 0;
                    border-bottom:1px solid #f1f5f9;'>
                    <div style='width:34px;height:34px;background:rgba(251,146,60,0.1);
                        border-radius:9px;display:flex;align-items:center;justify-content:center;
                        color:#fb923c;font-size:14px;flex-shrink:0;'>⚠</div>
                    <div style='flex:1;'>
                        <div style='font-size:14px;font-weight:600;color:#0f172a;'>{entry["state"]}</div>
                        <div style='font-size:12px;color:#64748b;margin-top:3px;'>{entry["tip"]}</div>
                    </div>
                    <div style='font-size:12px;color:#94a3b8;flex-shrink:0;'>{entry["time"]}</div>
                </div>"""
        st.markdown(white_card(actions_html), unsafe_allow_html=True)

    df = read_log()
    if not df.empty and df["alert"].sum() > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Alerts", int(df["alert"].sum()))
        c2.metric("Alert Rate",   f"{df['alert'].sum()/len(df)*100:.1f}%")
        alerts_df = df[df["alert"]==True]
        c3.metric("Top Trigger",  alerts_df["state"].mode()[0] if not alerts_df.empty else "—")

# ══════════════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════════════
elif page == "📁  History":
    df = read_log()
    if df.empty:
        empty_state_html("📁","No history yet","Run a live session to start recording data")
    else:
        c1,c2 = st.columns([3,1])
        with c1:
            st.markdown(f"<div style='font-size:15px;color:#334155;font-weight:500;'>{len(df)} total readings recorded</div>", unsafe_allow_html=True)
        with c2:
            st.download_button("⬇  Download CSV",
                              df.to_csv(index=False).encode("utf-8"),
                              "moodguard_sessions.csv","text/csv")
        st.dataframe(df.sort_values("timestamp",ascending=False),
                    use_container_width=True, height=500)

# ══════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════
elif page == "ℹ   About":
    # Hero
    st.markdown("""
    <div style='background:linear-gradient(145deg,#0c4a6e,#0369a1,#0284c7);
        border-radius:20px;padding:48px;text-align:center;margin-bottom:28px;
        box-shadow:0 12px 40px rgba(3,105,161,0.25);position:relative;overflow:hidden;'>
        <div style='position:absolute;top:-60px;left:-60px;width:200px;height:200px;
            background:radial-gradient(circle,rgba(56,189,248,0.3),transparent 70%);'></div>
        <div style='position:absolute;bottom:-60px;right:-60px;width:200px;height:200px;
            background:radial-gradient(circle,rgba(14,165,233,0.2),transparent 70%);'></div>
        <div style='font-size:42px;font-weight:800;color:#ffffff;letter-spacing:-0.03em;
            position:relative;z-index:1;'>
            Mood<span style='color:#7dd3fc;'>Guard</span>
            <span style='color:rgba(255,255,255,0.35);font-weight:300;'> 2.0</span>
        </div>
        <div style='font-size:12px;color:rgba(255,255,255,0.4);margin-top:10px;
            letter-spacing:0.2em;text-transform:uppercase;position:relative;z-index:1;'>
            AI Mental State Intelligence Platform
        </div>
        <div style='font-size:15px;color:rgba(255,255,255,0.65);margin-top:18px;
            max-width:540px;margin-left:auto;margin-right:auto;line-height:1.8;
            position:relative;z-index:1;'>
            A complete human state understanding system combining Computer Vision,
            Behavioral Intelligence, and Generative AI.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Developer card
        dev_html = card_title_html("Developer")
        dev_html += """
        <div style='font-size:24px;font-weight:800;color:#0369a1;margin-bottom:6px;'>Hamna Munir</div>
        <div style='font-size:14px;color:#64748b;'>AI/ML Engineer</div>
        <div style='font-size:14px;color:#64748b;margin-bottom:14px;'>Software Engineering Student</div>
        <div style='display:flex;gap:8px;flex-wrap:wrap;'>"""
        for tag in ["AI/ML","Computer Vision","GenAI","Deep Learning"]:
            dev_html += f"<span style='background:#eff6ff;color:#1d4ed8;padding:5px 12px;border-radius:6px;font-size:12px;font-weight:600;'>{tag}</span>"
        dev_html += "</div>"
        st.markdown(white_card(dev_html), unsafe_allow_html=True)

        # Tech Stack
        tech_html = card_title_html("Tech Stack")
        for tech in ["Python 3.10","Streamlit","OpenCV","MediaPipe","ONNX Runtime",
                     "Google Gemini 1.5","scikit-learn","Plotly","FER2013","RandomForest"]:
            tech_html += f"<span style='display:inline-block;background:#f1f5f9;color:#334155;padding:6px 14px;border-radius:8px;font-size:13px;font-weight:500;margin:3px;'>{tech}</span>"
        st.markdown(white_card(tech_html), unsafe_allow_html=True)

        # Architecture
        arch_html = card_title_html("System Architecture")
        arch_html += """
        <div style='font-size:13px;color:#475569;line-height:2.2;font-family:monospace;
            background:#f8fafc;padding:16px;border-radius:10px;border:1px solid #e2e8f0;'>
            <span style='color:#0ea5e9;font-weight:600;'>Webcam / Photo Input</span><br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            <span style='color:#059669;font-weight:600;'>Face Detection</span> (OpenCV)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            <span style='color:#818cf8;font-weight:600;'>Emotion</span> (ONNX)&nbsp;&nbsp;&nbsp;<span style='color:#fb923c;font-weight:600;'>Focus</span> (RF)<br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            <span style='color:#f472b6;font-weight:600;'>Behavioral Intelligence</span><br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            <span style='color:#0ea5e9;font-weight:600;'>Gemini AI Insights</span><br>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br>
            <span style='color:#0f172a;font-weight:700;'>Streamlit Dashboard</span>
        </div>"""
        st.markdown(white_card(arch_html), unsafe_allow_html=True)

    with col2:
        # Features
        features_html = card_title_html("Features")
        for feat in [
            "Real-time emotion detection — 7 emotions",
            "Focus & attention scoring via eye tracking (0–100)",
            "Behavioral intelligence & mental state analysis",
            "Photo analysis — upload any image instantly",
            "Dedicated live camera page",
            "AI insights via Google Gemini 1.5 Flash",
            "Smart agent with automatic recommendations",
            "Session analytics — charts & emotion timeline",
            "Complete session history with CSV export",
            "Blink detection & attention monitoring"
        ]:
            features_html += f"""
            <div style='display:flex;gap:10px;padding:9px 0;border-bottom:1px solid #f1f5f9;
                font-size:14px;color:#334155;'>
                <span style='color:#0ea5e9;font-weight:700;flex-shrink:0;'>✓</span>{feat}
            </div>"""
        st.markdown(white_card(features_html), unsafe_allow_html=True)

        # Models
        models_html = card_title_html("AI Models")
        models_html += """
        <div style='background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;
            padding:16px;margin-bottom:12px;'>
            <div style='font-size:14px;font-weight:700;color:#0369a1;margin-bottom:8px;'>
                🧠 Emotion Model (ONNX)
            </div>
            <div style='font-size:13px;color:#64748b;line-height:1.8;'>
                Architecture: CNN · Dataset: FER2013<br>
                Accuracy: 66.6% · Output: 7 emotions
            </div>
        </div>
        <div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:16px;'>
            <div style='font-size:14px;font-weight:700;color:#059669;margin-bottom:8px;'>
                👁️ Focus Model (RandomForest)
            </div>
            <div style='font-size:13px;color:#64748b;line-height:1.8;'>
                Architecture: RandomForestClassifier<br>
                Input: 6 MediaPipe eye landmarks<br>
                Output: Focused / Distracted
            </div>
        </div>"""
        st.markdown(white_card(models_html), unsafe_allow_html=True)

        # Version
        version_html = card_title_html("Version Info")
        version_html += "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>"
        for label, val in [("Version","2.0"),("Year","2026"),("Status","Active"),("License","MIT")]:
            version_html += f"""
            <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px;'>
                <div style='font-size:10px;color:#94a3b8;text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:6px;'>{label}</div>
                <div style='font-size:18px;font-weight:700;color:#0ea5e9;'>{val}</div>
            </div>"""
        version_html += "</div>"
        st.markdown(white_card(version_html), unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:20px;color:#94a3b8;font-size:13px;
    border-top:1px solid #e2e8f0;margin-top:32px;'>
    MoodGuard 2.0 &nbsp;·&nbsp; Built by
    <span style='color:#0ea5e9;font-weight:600;'>Hamna Munir</span>
    &nbsp;·&nbsp; OpenCV · MediaPipe · ONNX · Gemini 1.5 · Streamlit
    &nbsp;·&nbsp; v2.0 · 2026
</div>
""", unsafe_allow_html=True)