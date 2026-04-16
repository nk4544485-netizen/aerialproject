# 🛸 AeroGuard: Drone vs. Bird Classification System

A professional-grade aerial surveillance system using Deep Learning (MobileNetV2) to distinguish between drones and birds in real-time.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 🎯 Features
- **Real-time Classification** — Upload any aerial image for instant drone/bird detection
- **MobileNetV2 Backbone** — Transfer learning with ImageNet weights for high accuracy
- **Confidence Scoring** — Visual confidence meter with percentage readout
- **Detection History** — Automatic logging of all classification results
- **Premium Dark UI** — Glassmorphism-inspired Streamlit dashboard

## 🏗️ Project Architecture
```
AeroGuard_Project/
├── .streamlit/
│   └── config.toml          # Streamlit theme & server config
├── models/
│   └── final_model.keras    # Trained MobileNetV2 classifier
├── project6/                # Training dataset (not in repo)
│   ├── Bird/
│   └── Drone/
├── app.py                   # Streamlit dashboard
├── train_model.py           # Model training script
├── requirements.txt         # Python dependencies
├── packages.txt             # System dependencies (Streamlit Cloud)
├── .gitignore
└── README.md
```

## 📊 Preprocessing Pipeline
| Step | Detail |
|------|--------|
| Resize | 224 × 224 pixels |
| Normalization | 1/255.0 scaling |
| Channel | RGB conversion |

## ⚖️ Prediction Logic
- `Score > 0.5` → 🚨 **DRONE DETECTED** (High-priority alert)
- `Score ≤ 0.5` → 🐦 **BIRD IDENTIFIED** (Low-priority detection)

---

## 🚀 Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/AeroGuard_Project.git
cd AeroGuard_Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Train the model — requires dataset in ./project6
python train_model.py

# 4. Launch the dashboard
streamlit run app.py
```

## ☁️ Streamlit Cloud Deployment

1. **Push to GitHub** — Make sure `models/final_model.keras` is included
2. **Go to** [share.streamlit.io](https://share.streamlit.io)
3. **Click** "New app"
4. **Select** your GitHub repo → Branch: `main` → Main file: `app.py`
5. **Click** "Deploy" — Done! 🎉

> **Note:** The dataset (`project6/`) is excluded from the repo via `.gitignore`.  
> The pre-trained model (`models/final_model.keras`) is included so the app works without retraining.

---

## 🛠️ Tech Stack
- **Framework:** Streamlit
- **Model:** MobileNetV2 (Transfer Learning)
- **Backend:** TensorFlow / Keras
- **Language:** Python 3.10+

---
*Developed as Senior AI System Architecture Project*
