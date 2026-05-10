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
# CSS
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
}

.stApp { background-color: #07090f; color: #c8cdd8; }
header[data-testid="stHeader"] { background: transparent; }

/* ── Hide default Streamlit file uploader label duplication ── */
[data-testid="stFileUploaderDropzoneInstructions"] div span:first-child {
    display: none;
}

/* ── Hero ── */
.hero {
    padding: 3rem 0 2rem;
    text-align: center;
}
.hero-tag {
    display: inline-block;
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.2);
    color: #fbbf24;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 4px;
    margin-bottom: 1.2rem;
}
.hero h1 {
    font-weight: 800;
    font-size: 2.8rem;
    color: #f1f3f8;
    margin: 0 0 0.5rem;
    letter-spacing: -1px;
    line-height: 1.15;
}
.hero h1 span { color: #fbbf24; }
.hero p {
    color: #4a5568;
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
}

/* ── Section label ── */
.sec-label {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #374151;
    margin-bottom: 0.8rem;
    margin-top: 2rem;
}

/* ── Upload box ── */
[data-testid="stFileUploader"] {
    background: #0d1017;
    border: 1.5px dashed #1c2333;
    border-radius: 12px;
    padding: 0.4rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #374151;
}

/* ── Custom upload label ── */
.upload-hint {
    text-align: center;
    padding: 1.2rem 0 0.4rem;
    color: #374151;
    font-size: 0.82rem;
}
.upload-hint span {
    color: #fbbf24;
    font-weight: 600;
    cursor: pointer;
}

/* ── Action buttons ── */
.stButton > button {
    background: #0d1017 !important;
    color: #6b7280 !important;
    border: 1px solid #1c2333 !important;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 0.4rem 1.1rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #1c2333 !important;
    border-color: #fbbf24 !important;
    color: #fbbf24 !important;
}

/* ── Confidence info ── */
.conf-note {
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.15);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 0.78rem;
    color: #92400e;
    margin-bottom: 0.5rem;
    line-height: 1.5;
}

/* ── Result card ── */
.r-card {
    background: #0d1017;
    border: 1px solid #1c2333;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.r-filename {
    font-size: 0.72rem;
    color: #374151;
    margin-bottom: 0.8rem;
    font-family: 'DM Mono', monospace !important;
}
.img-lbl {
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #374151;
    margin-bottom: 5px;
}

/* ── Badges ── */
.badge-critical {
    display: inline-block;
    background: rgba(239,68,68,0.1);
    color: #f87171;
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 4px;
    padding: 2px 9px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.badge-warning {
    display: inline-block;
    background: rgba(245,158,11,0.1);
    color: #fbbf24;
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 4px;
    padding: 2px 9px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.badge-safe {
    display: inline-block;
    background: rgba(52,211,153,0.1);
    color: #34d399;
    border: 1px solid rgba(52,211,153,0.2);
    border-radius: 4px;
    padding: 2px 9px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.det-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid #111827;
}
.det-row:last-child { border-bottom: none; }
.det-conf {
    font-family: 'DM Mono', monospace !important;
    color: #374151;
    font-size: 0.75rem;
}
.no-det {
    font-size: 0.78rem;
    color: #374151;
    font-style: italic;
}

/* ── Alert inline ── */
.alert-crit {
    background: rgba(239,68,68,0.07);
    border-left: 2px solid #ef4444;
    border-radius: 6px;
    padding: 6px 10px;
    margin: 4px 0;
    color: #f87171;
    font-size: 0.78rem;
    font-weight: 600;
}
.alert-warn {
    background: rgba(245,158,11,0.07);
    border-left: 2px solid #f59e0b;
    border-radius: 6px;
    padding: 6px 10px;
    margin: 4px 0;
    color: #fbbf24;
    font-size: 0.78rem;
    font-weight: 600;
}

/* ── Divider ── */
.div-line { border: none; border-top: 1px solid #111827; margin: 2.5rem 0; }

/* ── Report section ── */
.report-header {
    text-align: center;
    margin-bottom: 2rem;
}
.report-header h2 {
    font-weight: 800;
    font-size: 1.6rem;
    color: #f1f3f8;
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
}
.report-header p { color: #374151; font-size: 0.82rem; }

/* ── Score gauge ── */
.score-ring {
    text-align: center;
    margin: 1.5rem auto;
}
.score-num {
    font-family: 'DM Mono', monospace !important;
    font-size: 3.5rem;
    font-weight: 500;
    line-height: 1;
}
.score-lbl {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #374151;
    margin-top: 4px;
}

/* ── Metric cards ── */
.mbox {
    background: #0d1017;
    border: 1px solid #1c2333;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.mval {
    font-family: 'DM Mono', monospace !important;
    font-size: 1.8rem;
    font-weight: 500;
    line-height: 1;
    margin-bottom: 5px;
}
.mlbl {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #374151;
}

/* ── Verdict pill ── */
.verdict {
    text-align: center;
    font-weight: 800;
    font-size: 1rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 0.7rem 1.5rem;
    border-radius: 8px;
    margin: 1.5rem auto;
    max-width: 300px;
}
.v-safe     { background: rgba(52,211,153,0.08); color: #34d399; border: 1px solid rgba(52,211,153,0.2); }
.v-warning  { background: rgba(245,158,11,0.08); color: #fbbf24; border: 1px solid rgba(245,158,11,0.2); }
.v-critical { background: rgba(239,68,68,0.08);  color: #f87171; border: 1px solid rgba(239,68,68,0.2); }

/* ── Timeline ── */
.timeline-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid #111827;
}
.timeline-item:last-child { border-bottom: none; }
.tl-dot-crit {
    width: 8px; height: 8px;
    background: #ef4444;
    border-radius: 50%;
    margin-top: 5px;
    flex-shrink: 0;
}
.tl-dot-warn {
    width: 8px; height: 8px;
    background: #f59e0b;
    border-radius: 50%;
    margin-top: 5px;
    flex-shrink: 0;
}
.tl-class {
    font-weight: 700;
    font-size: 0.82rem;
    color: #c8cdd8;
}
.tl-meta {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem;
    color: #374151;
    margin-top: 2px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #07090f !important;
    border-right: 1px solid #111827;
}
[data-testid="stSidebar"] * { color: #6b7280; }
[data-testid="stSidebar"] .sec-label { color: #1c2333; }

/* ── Slider ── */
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    font-family: 'DM Mono', monospace !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #111827;
    font-size: 0.7rem;
    padding: 3rem 0 1rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HERO
# ==========================================
st.markdown("""
<div class="hero">
    <div class="hero-tag">⚡ AI Safety System</div>
    <h1>Driver <span>Drowsiness</span><br>Detection</h1>
    <p>Upload driver images — the model analyzes behavior and flags risks instantly.</p>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style='padding:1.5rem 0 0.5rem;'>
        <div style='font-size:0.62rem;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:#1c2333;margin-bottom:1.2rem;'>Settings</div>
    </div>
    """, unsafe_allow_html=True)

    conf_threshold = st.slider("Confidence Threshold", 0.10, 1.00, 0.25, 0.05,
                               help="اخفضه لو النتايج مش بتطلع — جرب 0.15 أو 0.20")

    st.markdown("""
    <div style='font-size:0.72rem;color:#374151;margin-top:0.5rem;line-height:1.6;
                background:rgba(251,191,36,0.04);border:1px solid rgba(251,191,36,0.1);
                border-radius:6px;padding:8px 10px;'>
        💡 لو الصورة مش بتتطلعلها نتيجة، اخفض الـ threshold لـ 0.15–0.20
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#111827;margin:1.5rem 0'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.62rem;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:#1c2333;margin-bottom:1rem;'>Class Legend</div>", unsafe_allow_html=True)

    legend_items = [
        ("DangerousDriving", "badge-critical", "CRITICAL"),
        ("SleepyDriving",    "badge-critical", "CRITICAL"),
        ("Drinking",         "badge-critical", "CRITICAL"),
        ("Distracted",       "badge-warning",  "WARNING"),
        ("Yawn",             "badge-warning",  "WARNING"),
        ("SafeDriving",      "badge-safe",     "SAFE"),
    ]
    for name, cls, lvl in legend_items:
        st.markdown(f'<div style="margin-bottom:7px"><span class="{cls}">{name}</span></div>',
                    unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#111827;margin:1.5rem 0'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#1c2333;font-size:0.65rem;letter-spacing:1px;text-transform:uppercase;'>YOLO11s · Driver Safety AI</div>", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Could not load model `{MODEL_PATH}`: {e}")
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

        if st.session_state.counts[name] >= 1:   # كل detection تسجل في الـ report
            now = time.time()
            if now - st.session_state.last_alert.get(name, 0) > 3:
                alert = {"timestamp": now, "class": name,
                         "level": level, "confidence": conf,
                         "time_str": datetime.now().strftime("%H:%M:%S")}
                st.session_state.alert_log.append(alert)
                st.session_state.last_alert[name] = now
                alerts.append(alert)
    return alerts

# ==========================================
# UPLOAD SECTION
# ==========================================
st.markdown('<div class="sec-label">Upload Images</div>', unsafe_allow_html=True)

# Confidence hint
st.markdown("""
<div class="conf-note">
    ⚠️ <strong>لو النتايج مش بتطلع:</strong> اخفض الـ Confidence Threshold من الـ sidebar —
    جرب <strong>0.15</strong> أو <strong>0.20</strong> بدل 0.40.
    النموذج ممكن يكون شايف الكلاس بـ confidence أقل من الـ threshold.
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "اختار صور السائق",      # ← label واضحة ومش بتتكرر
    type=["jpg","jpeg","png"],
    accept_multiple_files=True,
    label_visibility="collapsed",   # ← إخفاء الـ label من فوق عشان ما تتكررش
)

# Custom hint under uploader
st.markdown("""
<div class="upload-hint">
    📁 اسحب الصور هنا أو <span>Browse</span> — JPG · PNG · حتى 200MB
</div>
""", unsafe_allow_html=True)

col_a, col_b, _ = st.columns([1, 1, 5])
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
    progress = st.progress(0, text="Analyzing images...")
    for i, f in enumerate(uploaded_files):
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
        progress.progress((i+1)/len(uploaded_files),
                          text=f"Analyzing {i+1}/{len(uploaded_files)}...")
    progress.empty()

elif analyze_btn:
    st.warning("⚠️ Please upload at least one image first.")

# ==========================================
# RESULTS
# ==========================================
if st.session_state.results:
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Detection Results</div>', unsafe_allow_html=True)

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
                st.markdown("<div class='no-det'>No detections — try lowering confidence</div>",
                            unsafe_allow_html=True)

            # Alerts
            if res["alerts"]:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                for alert in res["alerts"]:
                    css = "alert-crit" if alert["level"] == "CRITICAL" else "alert-warn"
                    ico = "🚨" if alert["level"] == "CRITICAL" else "⚠️"
                    st.markdown(f'<div class="{css}">{ico} {alert["class"]}</div>',
                                unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# SESSION REPORT  ← الجزء المحسّن
# ==========================================
if st.session_state.alert_log:
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)

    log      = st.session_state.alert_log
    counts   = Counter(a["class"] for a in log)
    critical = sum(1 for a in log if a["level"] == "CRITICAL")
    warning  = sum(1 for a in log if a["level"] == "WARNING")
    score    = max(0, 100 - critical * 10 - warning * 4)

    verdict_text  = "SAFE DRIVER"      if score >= 85 else \
                    "NEEDS ATTENTION"  if score >= 60 else \
                    "DANGEROUS DRIVER"
    verdict_emoji = "✅"  if score >= 85 else "⚠️"  if score >= 60 else "🚨"
    verdict_cls   = "v-safe"    if score >= 85 else \
                    "v-warning" if score >= 60 else "v-critical"
    score_color   = "#34d399" if score >= 85 else "#fbbf24" if score >= 60 else "#f87171"

    # Header
    st.markdown(f"""
    <div class="report-header">
        <div class="sec-label" style="margin-top:0">Session Safety Report</div>
        <h2>Analysis Complete</h2>
        <p>{datetime.now().strftime("%Y-%m-%d · %H:%M:%S")} · {len(st.session_state.results)} image(s) analyzed</p>
    </div>
    """, unsafe_allow_html=True)

    # Score + verdict
    col_score, col_meta = st.columns([1, 2])

    with col_score:
        st.markdown(f"""
        <div class="score-ring">
            <div class="score-num" style="color:{score_color}">{score}</div>
            <div class="score-lbl">Safety Score / 100</div>
        </div>
        <div class="verdict {verdict_cls}">{verdict_emoji} {verdict_text}</div>
        """, unsafe_allow_html=True)

    with col_meta:
        m1, m2 = st.columns(2)
        m3, m4 = st.columns(2)

        for col, val, lbl, clr in [
            (m1, str(len(log)),      "Total Alerts",    "#c8cdd8"),
            (m2, str(critical),      "Critical",        "#f87171"),
            (m3, str(warning),       "Warnings",        "#fbbf24"),
            (m4, str(len(st.session_state.results)), "Images", "#6b7280"),
        ]:
            with col:
                st.markdown(
                    f'<div class="mbox">'
                    f'<div class="mval" style="color:{clr}">{val}</div>'
                    f'<div class="mlbl">{lbl}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Chart ──
    if counts:
        col_chart, col_timeline = st.columns([1.2, 1])

        with col_chart:
            st.markdown('<div class="sec-label" style="margin-top:0">Alert Breakdown</div>',
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, max(2, len(counts) * 0.7)))
            fig.patch.set_facecolor("#0d1017")
            ax.set_facecolor("#0d1017")

            clrs = ["#ef4444" if c in DANGER_CLASSES else
                    "#f59e0b" if c in WARNING_CLASSES else
                    "#34d399" for c in counts.keys()]

            bars = ax.barh(list(counts.keys()), list(counts.values()),
                           color=clrs, height=0.45, edgecolor="none")

            ax.set_xlabel("Number of Alerts", color="#374151", fontsize=8)
            ax.tick_params(colors="#4b5563", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#111827")
            ax.xaxis.grid(True, color="#111827", zorder=0, linestyle="--", linewidth=0.5)
            ax.set_axisbelow(True)

            for bar, val in zip(bars, counts.values()):
                ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                        str(val), va='center', color="#6b7280", fontsize=8)

            plt.tight_layout(pad=1)
            st.pyplot(fig)
            plt.close()

        with col_timeline:
            st.markdown('<div class="sec-label" style="margin-top:0">Alert Timeline</div>',
                        unsafe_allow_html=True)

            st.markdown('<div style="background:#0d1017;border:1px solid #111827;border-radius:10px;padding:0.8rem 1rem;">', unsafe_allow_html=True)
            for alert in log[-15:]:   # آخر 15 alert
                dot_cls = "tl-dot-crit" if alert["level"] == "CRITICAL" else "tl-dot-warn"
                ico = "🚨" if alert["level"] == "CRITICAL" else "⚠️"
                t = alert.get("time_str", "—")
                conf_pct = f"{alert['confidence']:.0%}"
                st.markdown(f"""
                <div class="timeline-item">
                    <div class="{dot_cls}"></div>
                    <div>
                        <div class="tl-class">{ico} {alert['class']}</div>
                        <div class="tl-meta">{t} · conf {conf_pct} · {alert['level']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Class summary ──
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label" style="margin-top:0">Class Summary</div>',
                unsafe_allow_html=True)

    summary_cols = st.columns(len(CLASS_NAMES))
    for col, cls_name in zip(summary_cols, CLASS_NAMES):
        cnt   = counts.get(cls_name, 0)
        level = ALERT_CONFIG[cls_name]["level"]
        clr   = "#f87171" if level == "CRITICAL" else \
                "#fbbf24" if level == "WARNING"  else "#34d399"
        with col:
            st.markdown(
                f'<div class="mbox">'
                f'<div class="mval" style="color:{clr};font-size:1.5rem">{cnt}</div>'
                f'<div class="mlbl" style="font-size:0.55rem">{cls_name}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── Download ──
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    report_data = {
        "generated_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "images_analyzed": len(st.session_state.results),
        "total_alerts":  len(log),
        "critical_count": critical,
        "warning_count":  warning,
        "safety_score":  score,
        "verdict":       verdict_text,
        "class_breakdown": dict(counts),
        "alert_log": [
            {
                "class":      a["class"],
                "level":      a["level"],
                "confidence": round(a["confidence"], 3),
                "time":       a.get("time_str", "—"),
            }
            for a in log
        ],
    }

    col_dl, col_clr, _ = st.columns([1.2, 1.2, 5])
    with col_dl:
        st.download_button(
            label="⬇️  Download Report (JSON)",
            data=json.dumps(report_data, indent=4, ensure_ascii=False),
            file_name=f"driver_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_clr:
        if st.button("🗑  Clear Report", use_container_width=True):
            st.session_state.alert_log  = []
            st.session_state.counts     = {n: 0 for n in CLASS_NAMES}
            st.session_state.last_alert = {}
            st.session_state.results    = []
            st.rerun()

elif st.session_state.results:
    # Results موجودة بس مفيش alerts (كل حاجة Safe)
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;padding:2rem;'>
        <div style='font-size:2rem;margin-bottom:0.5rem;'>✅</div>
        <div style='font-weight:700;color:#34d399;font-size:1.1rem;letter-spacing:1px;'>ALL CLEAR — SAFE DRIVING</div>
        <div style='color:#374151;font-size:0.82rem;margin-top:0.5rem;'>No dangerous behavior detected in this session.</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================
st.markdown(
    "<div class='footer'>Driver Drowsiness Detection · YOLO11s · Built with Streamlit</div>",
    unsafe_allow_html=True
)
