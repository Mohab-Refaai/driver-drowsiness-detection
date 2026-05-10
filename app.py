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
    "DangerousDriving": (56,  56,  255),
    "Distracted":       (0,   165, 255),
    "Drinking":         (255, 56,  56),
    "SafeDriving":      (56,  200, 56),
    "SleepyDriving":    (200, 0,   200),
    "Yawn":             (200, 200, 56),
}

ALERT_CONFIG = {
    "DangerousDriving": {"level": "CRITICAL"},
    "SleepyDriving":    {"level": "CRITICAL"},
    "Yawn":             {"level": "WARNING"},
    "Distracted":       {"level": "WARNING"},
    "Drinking":         {"level": "CRITICAL"},
    "SafeDriving":      {"level": "SAFE"},      # ← SAFE = لا يطلع alert أبداً
}

DANGER_CLASSES  = ["DangerousDriving", "SleepyDriving", "Drinking"]
WARNING_CLASSES = ["Distracted", "Yawn"]
MODEL_PATH      = "best.pt"

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
)

# ==========================================
# CSS
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

.stApp { background-color: #080b12; color: #dde1ea; }

header[data-testid="stHeader"] { background: transparent; }

/* ── Hero ── */
.hero {
    padding: 3.5rem 0 2.5rem;
    text-align: center;
    position: relative;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 50%; transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse at center, rgba(99,179,237,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-tag {
    display: inline-block;
    background: rgba(99,179,237,0.1);
    border: 1px solid rgba(99,179,237,0.25);
    color: #63b3ed;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700;
    font-size: 2.6rem;
    color: #f0f4ff;
    margin: 0 0 0.6rem;
    letter-spacing: -0.5px;
    line-height: 1.2;
}
.hero p {
    color: #64748b;
    font-size: 1rem;
    font-weight: 400;
    margin: 0;
}

/* ── Section title ── */
.sec-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 1rem;
}

/* ── Card ── */
.card {
    background: #0e1420;
    border: 1px solid #1a2236;
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1rem;
}

/* ── img label ── */
.img-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 6px;
}

/* ── Badges ── */
.badge-critical {
    display: inline-block;
    background: rgba(239,68,68,0.12);
    color: #f87171;
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.76rem;
    font-weight: 600;
}
.badge-warning {
    display: inline-block;
    background: rgba(245,158,11,0.12);
    color: #fbbf24;
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.76rem;
    font-weight: 600;
}
.badge-safe {
    display: inline-block;
    background: rgba(52,211,153,0.12);
    color: #34d399;
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 0.76rem;
    font-weight: 600;
}

/* ── Detection row ── */
.det-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 7px 0;
    border-bottom: 1px solid #1a2236;
}
.det-row:last-child { border-bottom: none; }
.det-conf {
    font-family: 'JetBrains Mono', monospace !important;
    color: #475569;
    font-size: 0.8rem;
}

/* ── Alert banners ── */
.alert-critical {
    background: rgba(239,68,68,0.08);
    border-left: 3px solid #ef4444;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 5px 0;
    color: #f87171;
    font-size: 0.85rem;
    font-weight: 500;
}
.alert-warning {
    background: rgba(245,158,11,0.08);
    border-left: 3px solid #f59e0b;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 5px 0;
    color: #fbbf24;
    font-size: 0.85rem;
    font-weight: 500;
}

/* ── Metric boxes ── */
.metric-box {
    background: #0e1420;
    border: 1px solid #1a2236;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 2.2rem;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-lbl {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #475569;
}

/* ── Verdict ── */
.verdict {
    text-align: center;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 0.8rem;
    border-radius: 10px;
    margin: 1.2rem 0;
}
.verdict-safe     { background: rgba(52,211,153,0.08); color: #34d399; border: 1px solid rgba(52,211,153,0.2); }
.verdict-warning  { background: rgba(245,158,11,0.08); color: #fbbf24; border: 1px solid rgba(245,158,11,0.2); }
.verdict-critical { background: rgba(239,68,68,0.08);  color: #f87171; border: 1px solid rgba(239,68,68,0.2); }

/* ── Buttons ── */
.stButton > button {
    background: #0e1420 !important;
    color: #94a3b8 !important;
    border: 1px solid #1a2236 !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.45rem 1.2rem !important;
    white-space: nowrap !important;
    transition: all 0.15s !important;
    min-width: 110px !important;
}
.stButton > button:hover {
    background: #1a2236 !important;
    border-color: #63b3ed !important;
    color: #63b3ed !important;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #0e1420;
    border: 1.5px dashed #1a2236;
    border-radius: 14px;
    padding: 0.5rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #080b12 !important;
    border-right: 1px solid #1a2236;
}
[data-testid="stSidebar"] * { color: #94a3b8; }

/* ── Divider ── */
.divider { border: none; border-top: 1px solid #1a2236; margin: 2rem 0; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: #1e293b;
    font-size: 0.75rem;
    padding: 2.5rem 0 1rem;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HERO
# ==========================================
st.markdown("""
<div class="hero">
    <div class="hero-tag">AI Safety System</div>
    <h1>Driver Drowsiness Detection</h1>
    <p>Upload driver images — the model analyzes behavior and flags risks in real time.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("<div style='padding:1rem 0 0.5rem;font-size:0.72rem;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#475569;'>Settings</div>", unsafe_allow_html=True)

    conf_threshold = st.slider("Confidence Threshold", 0.10, 1.00, 0.40, 0.05)

    st.markdown("<hr style='border-color:#1a2236;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.72rem;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#475569;margin-bottom:0.8rem;'>Class Legend</div>", unsafe_allow_html=True)

    legend = {
        "CRITICAL": (["DangerousDriving","SleepyDriving","Drinking"], "badge-critical"),
        "WARNING":  (["Distracted","Yawn"],                            "badge-warning"),
        "SAFE":     (["SafeDriving"],                                  "badge-safe"),
    }
    for level, (names, cls) in legend.items():
        for n in names:
            st.markdown(f'<div style="margin-bottom:6px"><span class="{cls}">{n}</span></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1a2236;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#1e293b;font-size:0.72rem;letter-spacing:0.5px;'>YOLO11s · Driver Safety AI</div>", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
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
            cv2.putText(output, f"{name} {conf_v:.2f}",
                        (x1, max(y1-10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            detections.append((name, conf_v))
    return output, detections

# ==========================================
# SESSION STATE
# ==========================================
for key, default in [
    ("alert_log",  []),
    ("counts",     {n: 0 for n in CLASS_NAMES}),
    ("last_alert", {}),
    ("results",    []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

def check_alerts(detections):
    """فقط CRITICAL و WARNING يطلع ليهم alert — SAFE مش بيطلعش"""
    detected_now = [d[0] for d in detections]
    alerts = []

    for name in CLASS_NAMES:
        if name not in detected_now:
            st.session_state.counts[name] = 0

    for name, conf in detections:
        level = ALERT_CONFIG[name]["level"]

        # SafeDriving لا يطلع alert أبداً
        if level == "SAFE":
            continue

        st.session_state.counts[name] += 1

        if st.session_state.counts[name] >= 2:
            now = time.time()
            if now - st.session_state.last_alert.get(name, 0) > 5:
                alert = {"timestamp": now, "class": name,
                         "level": level, "confidence": conf}
                st.session_state.alert_log.append(alert)
                st.session_state.last_alert[name] = now
                alerts.append(alert)
    return alerts

# ==========================================
# UPLOAD SECTION
# ==========================================
st.markdown('<div class="sec-title">Upload Images</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drop driver images here",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

col_a, col_b, _ = st.columns([1.1, 1.1, 6])
with col_a:
    analyze_btn = st.button("🔍  Analyze", use_container_width=True)
with col_b:
    clear_btn = st.button("🗑  Clear All", use_container_width=True)

if clear_btn:
    st.session_state.alert_log  = []
    st.session_state.counts     = {n: 0 for n in CLASS_NAMES}
    st.session_state.last_alert = {}
    st.session_state.results    = []
    st.rerun()

# ==========================================
# RUN ANALYSIS
# ==========================================
if analyze_btn and uploaded_files:
    st.session_state.results = []
    with st.spinner("Analyzing..."):
        for f in uploaded_files:
            img   = Image.open(f).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            out_frame, dets = detect(frame, conf_threshold)
            out_rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
            alerts  = check_alerts(dets)
            st.session_state.results.append({
                "name":       f.name,
                "original":   img,
                "output":     out_rgb,
                "detections": dets,
                "alerts":     alerts,
            })
elif analyze_btn:
    st.warning("Please upload at least one image first.")

# ==========================================
# RESULTS
# ==========================================
if st.session_state.results:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Detection Results</div>', unsafe_allow_html=True)

    for res in st.session_state.results:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:0.78rem;color:#475569;margin-bottom:0.8rem;'>📎 {res['name']}</div>",
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns([2, 2, 1.4])

        with c1:
            st.markdown("<div class='img-label'>Original</div>", unsafe_allow_html=True)
            st.image(res["original"], use_column_width=True)

        with c2:
            st.markdown("<div class='img-label'>Detection</div>", unsafe_allow_html=True)
            st.image(res["output"], use_column_width=True)

        with c3:
            st.markdown("<div class='img-label'>Findings</div>", unsafe_allow_html=True)
            if res["detections"]:
                for name, conf in res["detections"]:
                    lvl   = ALERT_CONFIG[name]["level"]
                    badge = (
                        f'<span class="badge-critical">{name}</span>' if lvl == "CRITICAL" else
                        f'<span class="badge-warning">{name}</span>'  if lvl == "WARNING"  else
                        f'<span class="badge-safe">{name}</span>'
                    )
                    st.markdown(
                        f'<div class="det-row">{badge}'
                        f'<span class="det-conf">{conf:.0%}</span></div>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown("<div style='color:#475569;font-size:0.82rem;'>No detections</div>",
                            unsafe_allow_html=True)

            # Alerts — SafeDriving مش بيطلعش هنا
            if res["alerts"]:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                for alert in res["alerts"]:
                    css = "alert-critical" if alert["level"] == "CRITICAL" else "alert-warning"
                    ico = "🚨" if alert["level"] == "CRITICAL" else "⚠️"
                    st.markdown(f'<div class="{css}">{ico} {alert["class"]}</div>',
                                unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# SESSION REPORT
# ==========================================
if st.session_state.alert_log:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Session Report</div>', unsafe_allow_html=True)

    log      = st.session_state.alert_log
    counts   = Counter(a["class"] for a in log)
    critical = sum(1 for a in log if a["level"] == "CRITICAL")
    warning  = sum(1 for a in log if a["level"] == "WARNING")
    score    = max(0, 100 - critical * 10 - warning * 4)

    verdict_text  = "✅  SAFE DRIVER"     if score >= 85 else \
                    "⚠️  NEEDS ATTENTION" if score >= 60 else \
                    "🚨  DANGEROUS DRIVER"
    verdict_class = "verdict-safe"     if score >= 85 else \
                    "verdict-warning"  if score >= 60 else \
                    "verdict-critical"
    score_color   = "#34d399" if score >= 85 else "#fbbf24" if score >= 60 else "#f87171"

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    for col, val, lbl, clr in zip(
        [m1, m2, m3, m4],
        [f"{score}/100", str(len(log)), str(critical), str(warning)],
        ["Safety Score", "Total Alerts", "Critical", "Warnings"],
        [score_color, "#63b3ed", "#f87171", "#fbbf24"]
    ):
        with col:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-val" style="color:{clr}">{val}</div>'
                f'<div class="metric-lbl">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # Verdict
    st.markdown(f'<div class="verdict {verdict_class}">{verdict_text}</div>',
                unsafe_allow_html=True)

    # Chart — أصغر وأنيق
    if counts:
        fig, ax = plt.subplots(figsize=(6, 2.5))
        fig.patch.set_facecolor("#0e1420")
        ax.set_facecolor("#0e1420")

        clrs = ["#f87171" if c in DANGER_CLASSES else
                "#fbbf24" if c in WARNING_CLASSES else
                "#34d399" for c in counts.keys()]

        bars = ax.barh(list(counts.keys()), list(counts.values()),
                       color=clrs, height=0.5)

        ax.set_xlabel("Alerts", color="#475569", fontsize=8)
        ax.tick_params(colors="#64748b", labelsize=8)
        ax.set_title("Alert Breakdown", color="#94a3b8",
                     fontsize=9, pad=8, loc="left")

        for spine in ax.spines.values():
            spine.set_color("#1a2236")
        ax.xaxis.grid(True, color="#1a2236", zorder=0)
        ax.set_axisbelow(True)

        # Value labels
        for bar, val in zip(bars, counts.values()):
            ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                    str(val), va='center', color="#94a3b8", fontsize=8)

        plt.tight_layout(pad=1)
        st.pyplot(fig)

    # Download
    report_data = {
        "date":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_alerts":  len(log),
        "critical":      critical,
        "warning":       warning,
        "score":         score,
        "verdict":       verdict_text,
        "breakdown":     dict(counts),
    }
    st.download_button(
        label="⬇️  Download Report (JSON)",
        data=json.dumps(report_data, indent=4),
        file_name="driver_report.json",
        mime="application/json",
    )

# ==========================================
# FOOTER
# ==========================================
st.markdown(
    "<div class='footer'>Driver Drowsiness Detection · YOLO11s · Streamlit Cloud</div>",
    unsafe_allow_html=True
)
