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
    "SafeDriving":      {"level": "SAFE"},
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
# CSS — Cinematic Dark
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: #080a0e;
    color: #9aa3b2;
}
header[data-testid="stHeader"] { background: transparent; }

/* ── Hero ── */
.hero {
    padding: 4rem 0 3rem;
    text-align: center;
    position: relative;
}
.hero-glow {
    position: absolute;
    top: 0; left: 50%; transform: translateX(-50%);
    width: 500px; height: 200px;
    background: radial-gradient(ellipse, rgba(220,38,38,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #dc2626;
    margin-bottom: 1rem;
    display: block;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: clamp(3.5rem, 8vw, 6.5rem);
    font-weight: 400;
    color: #f5f6f8;
    line-height: 0.92;
    letter-spacing: 2px;
    margin: 0 0 1.2rem;
}
.hero-title em {
    font-style: normal;
    color: #dc2626;
}
.hero-sub {
    font-size: 0.88rem;
    color: #4b5563;
    font-weight: 400;
    letter-spacing: 0.3px;
}

/* ── Section label ── */
.sec-label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #374151;
    margin-bottom: 0.8rem;
    margin-top: 2rem;
}

/* ── Upload ── */
[data-testid="stFileUploader"] {
    background: #0c0f16;
    border: 1px solid #161c28;
    border-radius: 10px;
    padding: 0.4rem;
}
[data-testid="stFileUploaderDropzoneInstructions"] div span:first-child {
    display: none;
}

/* ── Buttons ── */
.stButton > button {
    background: #0c0f16 !important;
    color: #4b5563 !important;
    border: 1px solid #161c28 !important;
    border-radius: 5px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500 !important;
    font-size: 0.75rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    padding: 0.45rem 1.1rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #161c28 !important;
    border-color: #dc2626 !important;
    color: #dc2626 !important;
}

/* ── Result card ── */
.r-card {
    background: #0c0f16;
    border: 1px solid #161c28;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.r-filename {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem;
    color: #374151;
    margin-bottom: 0.8rem;
    letter-spacing: 0.5px;
}
.img-lbl {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.55rem;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #1f2937;
    margin-bottom: 5px;
}

/* ── Badges ── */
.badge-critical {
    display: inline-block;
    background: rgba(220,38,38,0.1);
    color: #f87171;
    border: 1px solid rgba(220,38,38,0.2);
    border-radius: 3px;
    padding: 2px 8px;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.5px;
}
.badge-warning {
    display: inline-block;
    background: rgba(234,179,8,0.1);
    color: #fbbf24;
    border: 1px solid rgba(234,179,8,0.2);
    border-radius: 3px;
    padding: 2px 8px;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.5px;
}
.badge-safe {
    display: inline-block;
    background: rgba(34,197,94,0.1);
    color: #4ade80;
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 3px;
    padding: 2px 8px;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.5px;
}

/* ── Detection rows ── */
.det-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 5px 0;
    border-bottom: 1px solid #0f131c;
}
.det-row:last-child { border-bottom: none; }
.det-conf {
    font-family: 'JetBrains Mono', monospace !important;
    color: #2d3748;
    font-size: 0.7rem;
}
.no-det {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem;
    color: #1f2937;
    letter-spacing: 0.5px;
}

/* ── Alert inline ── */
.alert-crit {
    background: rgba(220,38,38,0.06);
    border-left: 2px solid #dc2626;
    border-radius: 4px;
    padding: 5px 8px;
    margin: 3px 0;
    color: #f87171;
    font-size: 0.72rem;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace !important;
}
.alert-warn {
    background: rgba(234,179,8,0.06);
    border-left: 2px solid #eab308;
    border-radius: 4px;
    padding: 5px 8px;
    margin: 3px 0;
    color: #fbbf24;
    font-size: 0.72rem;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Divider ── */
.div-line {
    border: none;
    border-top: 1px solid #0f131c;
    margin: 2.5rem 0;
}

/* ── Report ── */
.report-title {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 3.5rem;
    color: #f5f6f8;
    letter-spacing: 3px;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.report-sub {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem;
    color: #374151;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ── Score ── */
.score-wrap {
    text-align: center;
    padding: 1.5rem 0;
}
.score-num {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 6rem;
    line-height: 1;
    letter-spacing: 2px;
}
.score-denom {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem;
    color: #374151;
    letter-spacing: 2px;
    margin-top: 4px;
    text-transform: uppercase;
}

/* ── Metric box ── */
.mbox {
    background: #0c0f16;
    border: 1px solid #161c28;
    border-radius: 8px;
    padding: 0.9rem 0.8rem;
    text-align: center;
}
.mval {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2.4rem;
    line-height: 1;
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.mlbl {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.55rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #374151;
}

/* ── Verdict ── */
.verdict {
    font-family: 'Bebas Neue', sans-serif !important;
    text-align: center;
    font-size: 1rem;
    letter-spacing: 4px;
    padding: 0.6rem 1.5rem;
    border-radius: 5px;
    margin: 1rem auto;
    max-width: 260px;
    display: block;
}
.v-safe     { background: rgba(34,197,94,0.07); color: #4ade80; border: 1px solid rgba(34,197,94,0.15); }
.v-warning  { background: rgba(234,179,8,0.07); color: #fbbf24; border: 1px solid rgba(234,179,8,0.15); }
.v-critical { background: rgba(220,38,38,0.07); color: #f87171; border: 1px solid rgba(220,38,38,0.15); }

/* ── Timeline ── */
.tl-wrap {
    background: #0c0f16;
    border: 1px solid #161c28;
    border-radius: 8px;
    padding: 0.8rem 1rem;
}
.tl-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 7px 0;
    border-bottom: 1px solid #0f131c;
}
.tl-item:last-child { border-bottom: none; }
.tl-dot-c { width:7px; height:7px; background:#dc2626; border-radius:50%; margin-top:5px; flex-shrink:0; }
.tl-dot-w { width:7px; height:7px; background:#eab308; border-radius:50%; margin-top:5px; flex-shrink:0; }
.tl-cls { font-weight:600; font-size:0.78rem; color:#9aa3b2; letter-spacing:0.3px; }
.tl-meta {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.62rem;
    color: #374151;
    margin-top: 1px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #080a0e !important;
    border-right: 1px solid #0f131c;
}
[data-testid="stSidebar"] * { color: #4b5563; }

/* ── Footer ── */
.footer {
    text-align: center;
    font-family: 'JetBrains Mono', monospace !important;
    color: #0f131c;
    font-size: 0.62rem;
    padding: 3rem 0 1rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HERO
# ==========================================
st.markdown("""
<div class="hero">
    <div class="hero-glow"></div>
    <span class="hero-eyebrow">// AI Safety System</span>
    <div class="hero-title">Driver<br><em>Drowsiness</em><br>Detection</div>
    <p class="hero-sub">Upload driver images — the model analyzes behavior and flags risks.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style='padding:1.5rem 0 1rem;'>
        <div style='font-family:"JetBrains Mono",monospace;font-size:0.58rem;letter-spacing:3px;
                    text-transform:uppercase;color:#1f2937;margin-bottom:1.2rem;'>// Settings</div>
    </div>
    """, unsafe_allow_html=True)

    conf_threshold = st.slider(
        "Confidence Threshold", 0.10, 1.00, 0.15, 0.05,
        help="Lower = more detections. Recommended: 0.15–0.25"
    )

    st.markdown("<hr style='border-color:#0f131c;margin:1.2rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace;font-size:0.58rem;letter-spacing:3px;
                text-transform:uppercase;color:#1f2937;margin-bottom:1rem;'>// Classes</div>
    """, unsafe_allow_html=True)

    legend_items = [
        ("DangerousDriving", "badge-critical"),
        ("SleepyDriving",    "badge-critical"),
        ("Drinking",         "badge-critical"),
        ("Distracted",       "badge-warning"),
        ("Yawn",             "badge-warning"),
        ("SafeDriving",      "badge-safe"),
    ]
    for name, cls in legend_items:
        st.markdown(f'<div style="margin-bottom:7px"><span class="{cls}">{name}</span></div>',
                    unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#0f131c;margin:1.2rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace;color:#1f2937;
                font-size:0.58rem;letter-spacing:1.5px;'>YOLO11s · v1.0</div>
    """, unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model `{MODEL_PATH}`: {e}")
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
    detected_now = [d[0] for d in detections]
    alerts = []

    for name in CLASS_NAMES:
        if name not in detected_now:
            st.session_state.counts[name] = 0

    for name, conf in detections:
        level = ALERT_CONFIG[name]["level"]
        if level == "SAFE":
            continue

        st.session_state.counts[name] += 1

        if st.session_state.counts[name] >= 1:
            now = time.time()
            if now - st.session_state.last_alert.get(name, 0) > 2:
                alert = {
                    "timestamp":  now,
                    "class":      name,
                    "level":      level,
                    "confidence": conf,
                    "time_str":   datetime.now().strftime("%H:%M:%S"),
                }
                st.session_state.alert_log.append(alert)
                st.session_state.last_alert[name] = now
                alerts.append(alert)
    return alerts

# ==========================================
# UPLOAD
# ==========================================
st.markdown('<div class="sec-label">// Upload Images</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "drop images",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

col_a, col_b, _ = st.columns([1, 1, 6])
with col_a:
    analyze_btn = st.button("ANALYZE", use_container_width=True)
with col_b:
    clear_btn   = st.button("CLEAR", use_container_width=True)

if clear_btn:
    st.session_state.alert_log   = []
    st.session_state.counts      = {n: 0 for n in CLASS_NAMES}
    st.session_state.last_alert  = {}
    st.session_state.results     = []
    st.rerun()

# ==========================================
# RUN ANALYSIS
# ==========================================
if analyze_btn and uploaded_files:
    st.session_state.results    = []
    st.session_state.alert_log  = []
    st.session_state.counts     = {n: 0 for n in CLASS_NAMES}
    st.session_state.last_alert = {}

    progress = st.progress(0)
    for i, f in enumerate(uploaded_files):
        img       = Image.open(f).convert("RGB")
        frame     = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        out_frame, dets = detect(frame, conf_threshold)
        out_rgb   = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
        alerts    = check_alerts(dets)
        st.session_state.results.append({
            "name":       f.name,
            "original":   img,
            "output":     out_rgb,
            "detections": dets,
            "alerts":     alerts,
        })
        progress.progress((i+1)/len(uploaded_files))
    progress.empty()

elif analyze_btn:
    st.warning("Upload at least one image first.")

# ==========================================
# RESULTS
# ==========================================
if st.session_state.results:
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">// Detection Results</div>', unsafe_allow_html=True)

    for res in st.session_state.results:
        st.markdown('<div class="r-card">', unsafe_allow_html=True)
        st.markdown(f"<div class='r-filename'>📎 {res['name']}</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([2, 2, 1.3])
        with c1:
            st.markdown("<div class='img-lbl'>Original</div>", unsafe_allow_html=True)
            st.image(res["original"], use_column_width=True)
        with c2:
            st.markdown("<div class='img-lbl'>Detection</div>", unsafe_allow_html=True)
            st.image(res["output"], use_column_width=True)
        with c3:
            st.markdown("<div class='img-lbl'>Findings</div>", unsafe_allow_html=True)
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
                st.markdown("<div class='no-det'>— no detections —</div>", unsafe_allow_html=True)

            if res["alerts"]:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                for alert in res["alerts"]:
                    css = "alert-crit" if alert["level"] == "CRITICAL" else "alert-warn"
                    ico = "!" if alert["level"] == "CRITICAL" else "~"
                    st.markdown(f'<div class="{css}">[{ico}] {alert["class"]}</div>',
                                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# REPORT
# ==========================================
if st.session_state.alert_log:
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)

    log      = st.session_state.alert_log
    counts   = Counter(a["class"] for a in log)
    critical = sum(1 for a in log if a["level"] == "CRITICAL")
    warning  = sum(1 for a in log if a["level"] == "WARNING")
    score    = max(0, 100 - critical * 10 - warning * 4)

    v_text  = "SAFE DRIVER"      if score >= 85 else \
              "NEEDS ATTENTION"  if score >= 60 else \
              "DANGEROUS DRIVER"
    v_emoji = "✓" if score >= 85 else "!" if score >= 60 else "✕"
    v_cls   = "v-safe" if score >= 85 else "v-warning" if score >= 60 else "v-critical"
    s_color = "#4ade80" if score >= 85 else "#fbbf24" if score >= 60 else "#f87171"

    # Header
    st.markdown(f"""
    <div style='margin-bottom:2rem;'>
        <div class='sec-label' style='margin-top:0;'>// Session Report</div>
        <div class='report-title'>Safety<br>Analysis</div>
        <div class='report-sub'>{datetime.now().strftime("%Y · %m · %d — %H:%M:%S")} · {len(st.session_state.results)} frame(s)</div>
    </div>
    """, unsafe_allow_html=True)

    # Score + metrics
    col_score, col_right = st.columns([1, 2.2])

    with col_score:
        st.markdown(f"""
        <div class='score-wrap'>
            <div class='score-num' style='color:{s_color}'>{score}</div>
            <div class='score-denom'>/ 100 — Safety Score</div>
        </div>
        <div class='verdict {v_cls}'>{v_emoji}  {v_text}</div>
        """, unsafe_allow_html=True)

    with col_right:
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        for col, val, lbl, clr in [
            (r1c1, len(log),      "Total",    "#9aa3b2"),
            (r1c2, critical,      "Critical", "#f87171"),
            (r1c3, warning,       "Warnings", "#fbbf24"),
            (r1c4, len(st.session_state.results), "Frames", "#374151"),
        ]:
            with col:
                st.markdown(
                    f'<div class="mbox">'
                    f'<div class="mval" style="color:{clr}">{val}</div>'
                    f'<div class="mlbl">{lbl}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        cls_cols = st.columns(len(CLASS_NAMES))
        for col, cls_name in zip(cls_cols, CLASS_NAMES):
            cnt   = counts.get(cls_name, 0)
            level = ALERT_CONFIG[cls_name]["level"]
            clr   = "#f87171" if level == "CRITICAL" else \
                    "#fbbf24" if level == "WARNING"  else "#374151"
            with col:
                st.markdown(
                    f'<div class="mbox" style="padding:0.6rem 0.4rem">'
                    f'<div class="mval" style="color:{clr};font-size:1.6rem">{cnt}</div>'
                    f'<div class="mlbl" style="font-size:0.48rem">{cls_name}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # Chart + Timeline
    if counts:
        col_chart, col_tl = st.columns([1.3, 1])

        with col_chart:
            st.markdown('<div class="sec-label" style="margin-top:0">// Alert Breakdown</div>',
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, max(2, len(counts) * 0.75)))
            fig.patch.set_facecolor("#0c0f16")
            ax.set_facecolor("#0c0f16")

            clrs = ["#dc2626" if c in DANGER_CLASSES else
                    "#eab308" if c in WARNING_CLASSES else
                    "#22c55e" for c in counts.keys()]

            bars = ax.barh(list(counts.keys()), list(counts.values()),
                           color=clrs, height=0.42, edgecolor="none")

            ax.set_xlabel("Alerts", color="#374151", fontsize=7,
                          fontfamily="monospace")
            ax.tick_params(colors="#374151", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#0f131c")
            ax.xaxis.grid(True, color="#0f131c", zorder=0,
                          linestyle="--", linewidth=0.5)
            ax.set_axisbelow(True)

            for bar, val in zip(bars, counts.values()):
                ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                        str(val), va='center', color="#4b5563",
                        fontsize=7, fontfamily="monospace")

            plt.tight_layout(pad=0.8)
            st.pyplot(fig)
            plt.close()

        with col_tl:
            st.markdown('<div class="sec-label" style="margin-top:0">// Alert Log</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="tl-wrap">', unsafe_allow_html=True)
            for alert in log[-12:]:
                dot = "tl-dot-c" if alert["level"] == "CRITICAL" else "tl-dot-w"
                t   = alert.get("time_str", "—")
                c   = f"{alert['confidence']:.0%}"
                st.markdown(f"""
                <div class="tl-item">
                    <div class="{dot}"></div>
                    <div>
                        <div class="tl-cls">{alert['class']}</div>
                        <div class="tl-meta">{t}  ·  conf {c}  ·  {alert['level']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Download
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    report_data = {
        "generated_at":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "frames_analyzed": len(st.session_state.results),
        "total_alerts":    len(log),
        "critical_count":  critical,
        "warning_count":   warning,
        "safety_score":    score,
        "verdict":         v_text,
        "class_breakdown": dict(counts),
        "alert_log": [
            {"class": a["class"], "level": a["level"],
             "confidence": round(a["confidence"], 3), "time": a.get("time_str", "—")}
            for a in log
        ],
    }

    col_dl, col_clr, _ = st.columns([1.3, 1.1, 5])
    with col_dl:
        st.download_button(
            label="DOWNLOAD REPORT",
            data=json.dumps(report_data, indent=4),
            file_name=f"driver_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_clr:
        if st.button("CLEAR REPORT", use_container_width=True):
            st.session_state.alert_log  = []
            st.session_state.counts     = {n: 0 for n in CLASS_NAMES}
            st.session_state.last_alert = {}
            st.session_state.results    = []
            st.rerun()

elif st.session_state.results:
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;padding:3rem 0;'>
        <div style='font-family:"Bebas Neue",sans-serif;font-size:3rem;color:#4ade80;letter-spacing:3px;'>ALL CLEAR</div>
        <div style='font-family:"JetBrains Mono",monospace;font-size:0.65rem;color:#374151;
                    letter-spacing:2px;margin-top:0.5rem;text-transform:uppercase;'>No dangerous behavior detected</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================
st.markdown(
    "<div class='footer'>Driver Drowsiness Detection · YOLO11s · Streamlit</div>",
    unsafe_allow_html=True
)
