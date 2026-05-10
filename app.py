import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time
from collections import Counter
from datetime import datetime
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ==========================================
# CONFIG
# ==========================================
CLASS_NAMES = [
    "DangerousDriving",
    "Distracted",
    "Drinking",
    "SafeDriving",
    "SleepyDriving",
    "Yawn"
]

COLORS_BGR = {
    "DangerousDriving" : (  56,  56, 255),
    "Distracted"       : (   0, 165, 255),
    "Drinking"         : ( 255,  56,  56),
    "SafeDriving"      : (  56, 200,  56),
    "SleepyDriving"    : ( 200,   0, 200),
    "Yawn"             : ( 200, 200,  56),
}

ALERT_CONFIG = {
    "DangerousDriving" : {"level": "CRITICAL"},
    "SleepyDriving"    : {"level": "CRITICAL"},
    "Yawn"             : {"level": "WARNING"},
    "Distracted"       : {"level": "WARNING"},
    "Drinking"         : {"level": "CRITICAL"},
    "SafeDriving"      : {"level": "SAFE"},
}

DANGER_CLASSES  = ["DangerousDriving", "SleepyDriving", "Drinking"]
WARNING_CLASSES = ["Distracted", "Yawn"]
MODEL_PATH      = "best.pt"

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background-color: #0d0f14; color: #e8eaf0; }

header[data-testid="stHeader"] { background: transparent; }

.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    background: linear-gradient(135deg, #f0f0f0 0%, #a0c4ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: -1px;
}
.hero p { color: #6b7280; font-size: 1.05rem; font-weight: 300; margin-top: 0; }

.card {
    background: #161a23;
    border: 1px solid #1f2535;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}

.badge-critical {
    display: inline-block;
    background: rgba(255,56,56,0.15);
    color: #ff6b6b;
    border: 1px solid rgba(255,56,56,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-warning {
    display: inline-block;
    background: rgba(255,165,0,0.15);
    color: #ffaa40;
    border: 1px solid rgba(255,165,0,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-safe {
    display: inline-block;
    background: rgba(56,200,56,0.15);
    color: #4ade80;
    border: 1px solid rgba(56,200,56,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
}

.det-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0;
    border-bottom: 1px solid #1f2535;
}
.det-row:last-child { border-bottom: none; }
.det-conf { color: #6b7280; font-size: 0.85rem; }

.alert-critical {
    background: rgba(255,56,56,0.1);
    border-left: 4px solid #ff3838;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    color: #ff6b6b;
    font-weight: 500;
}
.alert-warning {
    background: rgba(255,165,0,0.1);
    border-left: 4px solid #ffa500;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    color: #ffaa40;
    font-weight: 500;
}

.metric-box {
    background: #161a23;
    border: 1px solid #1f2535;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #a0c4ff;
}
.metric-lbl { font-size: 0.78rem; color: #6b7280; margin-top: 2px; }

.divider { border: none; border-top: 1px solid #1f2535; margin: 2rem 0; }

.img-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.5rem;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #e8eaf0;
    margin-bottom: 1rem;
}

[data-testid="stFileUploader"] {
    background: #161a23;
    border: 2px dashed #1f2535;
    border-radius: 16px;
    padding: 1rem;
}

.stButton > button {
    background: #1f2535;
    color: #e8eaf0;
    border: 1px solid #2d3448;
    border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #2d3448;
    border-color: #a0c4ff;
    color: #a0c4ff;
}

[data-testid="stSidebar"] {
    background: #0d0f14;
    border-right: 1px solid #1f2535;
}

.footer {
    text-align: center;
    color: #374151;
    font-size: 0.8rem;
    padding: 2rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HERO
# ==========================================
st.markdown("""
<div class="hero">
    <h1>🚗 Driver Drowsiness Detection</h1>
    <p>Upload one or more driver images — the model will analyze behavior and flag risks instantly.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    conf_threshold = st.slider("Confidence Threshold", 0.10, 1.00, 0.40, 0.05)
    st.markdown("<hr style='border-color:#1f2535'>", unsafe_allow_html=True)
    st.markdown("### 🏷️ Class Legend")
    for name in CLASS_NAMES:
        level = ALERT_CONFIG[name]["level"]
        if level == "CRITICAL":
            st.markdown(f'<span class="badge-critical">🔴 {name}</span><br><br>', unsafe_allow_html=True)
        elif level == "WARNING":
            st.markdown(f'<span class="badge-warning">🟡 {name}</span><br><br>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge-safe">🟢 {name}</span><br><br>', unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1f2535'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#374151;font-size:0.78rem'>Driver Safety AI · YOLO11s</div>", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# ==========================================
# DETECTION
# ==========================================
def detect(frame, conf):
    results    = model.predict(frame, imgsz=640, conf=conf, verbose=False)
    boxes      = results[0].boxes
    output     = frame.copy()
    detections = []
    if boxes is not None:
        for box in boxes:
            cls_id       = int(box.cls[0].item())
            conf_v       = float(box.conf[0].item())
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            name         = CLASS_NAMES[cls_id]
            color        = COLORS_BGR.get(name, (200,200,200))
            cv2.rectangle(output, (x1,y1), (x2,y2), color, 2)
            cv2.putText(output, f"{name} {conf_v:.2f}", (x1, max(y1-10,20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            detections.append((name, conf_v))
    return output, detections

# ==========================================
# SESSION STATE
# ==========================================
for key, default in [
    ("alert_log",  []),
    ("counts",     {n:0 for n in CLASS_NAMES}),
    ("last_alert", {}),
    ("results",    []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def check_alerts(detections):
    detected_now = [d[0] for d in detections]
    alerts = []
    for name in CLASS_NAMES:
        if name not in detected_now:
            st.session_state.counts[name] = 0
    for name, conf in detections:
        st.session_state.counts[name] += 1
        level = ALERT_CONFIG[name]["level"]
        if st.session_state.counts[name] >= 2:
            now = time.time()
            if now - st.session_state.last_alert.get(name, 0) > 5:
                alert = {"timestamp": now, "class": name, "level": level, "confidence": conf}
                st.session_state.alert_log.append(alert)
                st.session_state.last_alert[name] = now
                alerts.append(alert)
    return alerts

# ==========================================
# UPLOAD SECTION
# ==========================================
st.markdown('<div class="section-title">📤 Upload Images</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drop one or more driver images here",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

col_analyze, col_clear, _ = st.columns([1, 1, 5])

with col_analyze:
    analyze_btn = st.button("🔍 Analyze", use_container_width=True)
with col_clear:
    clear_btn = st.button("🗑️ Clear All", use_container_width=True)

if clear_btn:
    st.session_state.alert_log  = []
    st.session_state.counts     = {n:0 for n in CLASS_NAMES}
    st.session_state.last_alert = {}
    st.session_state.results    = []
    st.rerun()

# ==========================================
# RUN ANALYSIS
# ==========================================
if analyze_btn and uploaded_files:
    st.session_state.results = []
    with st.spinner("Analyzing images..."):
        for uploaded in uploaded_files:
            img   = Image.open(uploaded).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            output_frame, detections = detect(frame, conf_threshold)
            output_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            alerts = check_alerts(detections)
            st.session_state.results.append({
                "name"       : uploaded.name,
                "original"   : img,
                "output"     : output_rgb,
                "detections" : detections,
                "alerts"     : alerts,
            })
elif analyze_btn and not uploaded_files:
    st.warning("Please upload at least one image first.")

# ==========================================
# SHOW RESULTS
# ==========================================
if st.session_state.results:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Detection Results</div>', unsafe_allow_html=True)

    for res in st.session_state.results:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-family:Syne,sans-serif;font-size:0.85rem;color:#6b7280;margin-bottom:0.8rem;'>📎 {res['name']}</div>",
            unsafe_allow_html=True
        )

        col_orig, col_det, col_info = st.columns([2, 2, 1.5])

        with col_orig:
            st.markdown("<div class='img-label'>Original</div>", unsafe_allow_html=True)
            st.image(res["original"], use_column_width=True)

        with col_det:
            st.markdown("<div class='img-label'>Detection</div>", unsafe_allow_html=True)
            st.image(res["output"], use_column_width=True)

        with col_info:
            st.markdown("<div class='img-label'>Findings</div>", unsafe_allow_html=True)
            if res["detections"]:
                for name, conf in res["detections"]:
                    level = ALERT_CONFIG[name]["level"]
                    badge = (
                        f'<span class="badge-critical">{name}</span>' if level == "CRITICAL" else
                        f'<span class="badge-warning">{name}</span>'  if level == "WARNING"  else
                        f'<span class="badge-safe">{name}</span>'
                    )
                    st.markdown(
                        f'<div class="det-row">{badge}<span class="det-conf">{conf:.0%}</span></div>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown("<div style='color:#6b7280;font-size:0.85rem;'>No detections</div>", unsafe_allow_html=True)

            if res["alerts"]:
                st.markdown("<br>", unsafe_allow_html=True)
                for alert in res["alerts"]:
                    css_class = "alert-critical" if alert["level"] == "CRITICAL" else "alert-warning"
                    icon      = "🚨" if alert["level"] == "CRITICAL" else "⚠️"
                    st.markdown(f'<div class="{css_class}">{icon} {alert["class"]}</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# SESSION REPORT
# ==========================================
if st.session_state.alert_log:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Session Report</div>', unsafe_allow_html=True)

    log      = st.session_state.alert_log
    counts   = Counter(a["class"] for a in log)
    critical = sum(1 for a in log if a["level"] == "CRITICAL")
    warning  = sum(1 for a in log if a["level"] == "WARNING")
    score    = max(0, 100 - critical * 10 - warning * 4)
    verdict  = (
        "✅ SAFE DRIVER"     if score >= 85 else
        "⚠️ NEEDS ATTENTION" if score >= 60 else
        "🚨 DANGEROUS DRIVER"
    )
    score_color = "#4ade80" if score >= 85 else "#ffaa40" if score >= 60 else "#ff6b6b"

    m1, m2, m3, m4 = st.columns(4)
    for col, val, lbl, clr in zip(
        [m1, m2, m3, m4],
        [f"{score}/100", len(log), critical, warning],
        ["Safety Score", "Total Alerts", "Critical", "Warnings"],
        [score_color, "#a0c4ff", "#ff6b6b", "#ffaa40"]
    ):
        with col:
            st.markdown(
                f'<div class="metric-box"><div class="metric-val" style="color:{clr}">{val}</div>'
                f'<div class="metric-lbl">{lbl}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="text-align:center;font-family:Syne,sans-serif;font-size:1.4rem;font-weight:700;margin:1rem 0;">{verdict}</div>',
        unsafe_allow_html=True
    )

    if counts:
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor("#161a23")
        ax.set_facecolor("#161a23")
        bar_colors = [
            "#ff6b6b" if c in DANGER_CLASSES else
            "#ffaa40" if c in WARNING_CLASSES else
            "#4ade80"
            for c in counts.keys()
        ]
        ax.bar(counts.keys(), counts.values(), color=bar_colors, width=0.5, zorder=3)
        ax.set_ylabel("Alerts", color="#6b7280")
        ax.tick_params(colors="#9ca3af")
        ax.set_title("Alert Breakdown", color="#e8eaf0", pad=10)
        plt.xticks(rotation=25, ha="right")
        for spine in ax.spines.values():
            spine.set_color("#1f2535")
        ax.yaxis.grid(True, color="#1f2535", zorder=0)
        plt.tight_layout()
        st.pyplot(fig)

    report_data = {
        "date"         : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_alerts" : len(log),
        "critical"     : critical,
        "warning"      : warning,
        "score"        : score,
        "verdict"      : verdict,
        "breakdown"    : dict(counts),
    }
    st.download_button(
        label     = "⬇️ Download Report (JSON)",
        data      = json.dumps(report_data, indent=4),
        file_name = "driver_report.json",
        mime      = "application/json",
    )

# ==========================================
# FOOTER
# ==========================================
st.markdown(
    "<div class='footer'>Driver Drowsiness Detection System · YOLO11s · Built with Streamlit</div>",
    unsafe_allow_html=True
)
