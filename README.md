# 🧠 MoodGuard 2.0 — AI Mental State Intelligence System

> Real-time emotion detection + focus tracking + behavioral intelligence + AI insights

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Live Demo
[MoodGuard 2.0 on Streamlit Cloud](#) 

## 💡 What It Does
MoodGuard 2.0 is a complete AI Mental State Intelligence System that:
- 👁️ Detects 7 emotions in real-time using CNN (FER2013)
- 🎯 Tracks focus/attention score (0-100) using eye landmarks
- 🧠 Analyzes behavioral state (stress, distraction, peak focus)
- ✦ Generates AI insights using Google Gemini 1.5 Flash
- 📊 Shows session analytics — emotion timeline & focus trends

## 🔧 Tech Stack
| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| Emotion Model | CNN → ONNX Runtime |
| Focus Model | RandomForest (scikit-learn) |
| Face Tracking | MediaPipe |
| AI Insights | Google Gemini 1.5 Flash |
| Charts | Plotly |
| Camera | streamlit-webrtc |

## ⚙️ Run Locally

```bash
# Clone karo
git clone https://github.com/Hamna-Munir/MoodGuard_2.0.git
cd MoodGuard_2.0

# Virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install
pip install -r requirements.txt

# .env file banao
echo GEMINI_API_KEY=your_key_here > .env

# Run karo
streamlit run app.py
```

## 👩‍💻 Developer
**Hamna Munir** — AI/ML Engineer · Software Engineering Student

