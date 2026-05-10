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

COLORS = {
    "DangerousDriving" : (255,  56,  56),
    "Distracted"       : (255, 165,   0),
    "Drinking"         : ( 56,  56, 255),
    "SafeDriving"      : ( 56, 200,  56),
    "SleepyDriving"    : (200,   0, 200),
    "Yawn"             : ( 56, 220, 220),
}

ALERT_CONFIG = {
    "DangerousDriving" : {"level": "CRITICAL", "threshold": 2},
    "SleepyDriving"    : {"level": "CRITICAL", "threshold": 3},
    "Yawn"             : {"level": "WARNING",  "threshold": 4},
    "Distracted"       : {"level": "WARNING",  "threshold": 4},
    "Drinking"         : {"level": "CRITICAL", "threshold": 2},
    "SafeDriving"      : {"level": "SAFE",     "threshold": 99},
}

DANGER_CLASSES  = ["DangerousDriving", "SleepyDriving", "Drinking"]
WARNING_CLASSES = ["Distracted", "Yawn"]

MODEL_PATH = "best.pt"

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title = "Driver Drowsiness Detection",
    page_icon  = "🚗",
    layout     = "wide",
)

st.title("🚗 Driver Drowsiness Detection System")
st.markdown("---")


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("⚙️ Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value = 0.1,
    max_value = 1.0,
    value     = 0.40,
    step      = 0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏷️ Classes")
for name in CLASS_NAMES:
    level = ALERT_CONFIG[name]["level"]
    icon  = "🔴" if level == "CRITICAL" else "🟡" if level == "WARNING" else "🟢"
    st.sidebar.markdown(f"{icon} {name}")


# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
    st.sidebar.success("✅ Model Loaded")
except Exception as e:
    st.sidebar.error(f"❌ Model Error: {e}")
    st.stop()


# ==========================================
# DETECTION FUNCTION
# ==========================================
def detect(frame, conf):

    results = model.predict(
        frame,
        imgsz   = 640,
        conf    = conf,
        verbose = False,
    )

    boxes      = results[0].boxes
    output     = frame.copy()
    detections = []

    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf_v = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            name  = CLASS_NAMES[cls_id]
            color = COLORS.get(name, (255, 255, 255))

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            label = f"{name} {conf_v:.2f}"
            cv2.putText(
                output, label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2
            )

            detections.append((name, conf_v))

    return output, detections


# ==========================================
# ALERT CHECK
# ==========================================
if "alert_log"  not in st.session_state:
    st.session_state.alert_log  = []

if "counts" not in st.session_state:
    st.session_state.counts = {n: 0 for n in CLASS_NAMES}

if "last_alert" not in st.session_state:
    st.session_state.last_alert = {}


def check_alerts(detections):

    detected_now = [d[0] for d in detections]
    alerts = []

    for name in CLASS_NAMES:
        if name not in detected_now:
            st.session_state.counts[name] = 0

    for name, conf in detections:

        st.session_state.counts[name] += 1
        config    = ALERT_CONFIG[name]
        threshold = config["threshold"]
        level     = config["level"]

        if st.session_state.counts[name] >= threshold:

            now       = time.time()
            last_time = st.session_state.last_alert.get(name, 0)

            if now - last_time > 5:
                alert = {
                    "timestamp"  : now,
                    "class"      : name,
                    "level"      : level,
                    "confidence" : conf,
                }
                st.session_state.alert_log.append(alert)
                st.session_state.last_alert[name] = now
                alerts.append(alert)

    return alerts


# ==========================================
# REPORT FUNCTION
# ==========================================
def generate_report(alert_log):

    if not alert_log:
        return None

    counts   = Counter(a["class"] for a in alert_log)
    critical = sum(1 for a in alert_log if a["level"] == "CRITICAL")
    warning  = sum(1 for a in alert_log if a["level"] == "WARNING")

    deductions = critical * 10 + warning * 4
    score      = max(0, 100 - deductions)

    verdict = (
        "✅ SAFE DRIVER"     if score >= 85 else
        "⚠️ NEEDS ATTENTION" if score >= 60 else
        "🚨 DANGEROUS DRIVER"
    )

    return {
        "date"         : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_alerts" : len(alert_log),
        "critical"     : critical,
        "warning"      : warning,
        "score"        : score,
        "verdict"      : verdict,
        "breakdown"    : dict(counts),
    }


# ==========================================
# MAIN — UPLOAD IMAGE
# ==========================================
st.subheader("📷 Upload Driver Image")

uploaded = st.file_uploader(
    "Upload an image to analyze",
    type=["jpg", "jpeg", "png"]
)

if uploaded:

    img   = Image.open(uploaded).convert("RGB")
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📥 Original**")
        st.image(img, use_column_width=True)

    output, detections = detect(frame, conf_threshold)
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    with col2:
        st.markdown("**🔍 Detection Result**")
        st.image(output_rgb, use_column_width=True)

    # Detections
    if detections:

        st.markdown("### 📊 Detections")

        for name, conf in detections:
            level = ALERT_CONFIG[name]["level"]
            icon  = "🔴" if level == "CRITICAL" else "🟡" if level == "WARNING" else "🟢"
            st.write(f"{icon} **{name}** — Confidence: `{conf:.2f}`")

        alerts = check_alerts(detections)

        for alert in alerts:
            if alert["level"] == "CRITICAL":
                st.error(f"🚨 CRITICAL ALERT: {alert['class']}")
            elif alert["level"] == "WARNING":
                st.warning(f"⚠️ WARNING: {alert['class']}")

    else:
        st.info("No detections found. Try lowering the confidence threshold.")


# ==========================================
# DRIVER REPORT
# ==========================================
st.markdown("---")
st.subheader("📋 Driver Session Report")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Generate Report"):

        report = generate_report(st.session_state.alert_log)

        if report:

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Safety Score",    f"{report['score']}/100")
            c2.metric("Total Alerts",    report["total_alerts"])
            c3.metric("Critical Alerts", report["critical"])
            c4.metric("Warning Alerts",  report["warning"])

            st.markdown(f"### Verdict: {report['verdict']}")

            if report["breakdown"]:

                fig, ax = plt.subplots(figsize=(7, 4))

                ax.bar(
                    report["breakdown"].keys(),
                    report["breakdown"].values(),
                    color=[
                        "#FF3838" if c in DANGER_CLASSES
                        else "#FFA500" if c in WARNING_CLASSES
                        else "#38C838"
                        for c in report["breakdown"].keys()
                    ]
                )

                ax.set_xlabel("Class")
                ax.set_ylabel("Alert Count")
                ax.set_title("Alerts by Class")
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

            report_json = json.dumps(report, indent=4)
            st.download_button(
                label     = "⬇️ Download Report (JSON)",
                data      = report_json,
                file_name = "driver_report.json",
                mime      = "application/json"
            )

        else:
            st.info("No alerts recorded yet. Upload images first.")

with col2:
    if st.button("🗑️ Clear Session"):
        st.session_state.alert_log  = []
        st.session_state.counts     = {n: 0 for n in CLASS_NAMES}
        st.session_state.last_alert = {}
        st.success("Session cleared.")
