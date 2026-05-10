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

CLASS_INFO = {
    "DangerousDriving": {
        "ar": "قيادة خطرة",
        "desc": "تجاوز السرعة أو التغيير المفاجئ للمسارات",
        "icon": "⚡",
        "level": "CRITICAL",
        "color_bgr": (56, 56, 255),
    },
    "Distracted": {
        "ar": "قيادة مشتتة",
        "desc": "استخدام الهاتف أو الأكل أو التحدث",
        "icon": "📵",
        "level": "WARNING",
        "color_bgr": (0, 165, 255),
    },
    "Drinking": {
        "ar": "شرب أثناء القيادة",
        "desc": "تناول المشروبات الكحولية خلف المقود",
        "icon": "🚫",
        "level": "CRITICAL",
        "color_bgr": (255, 56, 56),
    },
    "SafeDriving": {
        "ar": "قيادة آمنة",
        "desc": "التزام بقواعد الطريق والمسافة الآمنة",
        "icon": "✅",
        "level": "SAFE",
        "color_bgr": (56, 200, 56),
    },
    "SleepyDriving": {
        "ar": "قيادة نعسان",
        "desc": "علامات النعاس والإرهاق خلف المقود",
        "icon": "😴",
        "level": "CRITICAL",
        "color_bgr": (200, 0, 200),
    },
    "Yawn": {
        "ar": "تثاؤب",
        "desc": "التثاؤب مؤشر على التعب وضعف الانتباه",
        "icon": "🥱",
        "level": "WARNING",
        "color_bgr": (200, 200, 56),
    },
}

ALERT_CONFIG    = {k: {"level": v["level"]} for k, v in CLASS_INFO.items()}
COLORS_BGR      = {k: v["color_bgr"] for k, v in CLASS_INFO.items()}
DANGER_CLASSES  = ["DangerousDriving", "SleepyDriving", "Drinking"]
WARNING_CLASSES = ["Distracted", "Yawn"]
MODEL_PATH      = "best.pt"

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Driver Safety Monitor",
    page_icon="🛡️",
    layout="wide",
)

# ==========================================
# CSS
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

*, html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}
.stApp {
    background: #0b1120;
    color: #94a3b8;
}
header[data-testid="stHeader"] { background: transparent; }

/* ── Hero ── */
.hero { padding: 3.5rem 0 2rem; position: relative; overflow: hidden; }
.hero-bg-line {
    position: absolute; top:0; left:0; right:0; bottom:0;
    background:
        linear-gradient(90deg, transparent 49.5%, rgba(251,191,36,0.03) 49.5%,
            rgba(251,191,36,0.03) 50.5%, transparent 50.5%),
        linear-gradient(0deg, transparent 0%, rgba(14,165,233,0.02) 50%, transparent 100%);
    pointer-events: none;
}
.hero-inner { position: relative; z-index: 1; }
.hero-chip {
    display: inline-block;
    background: rgba(251,191,36,0.08);
    border: 1px solid rgba(251,191,36,0.2);
    color: #fbbf24;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.6rem; letter-spacing: 3px; text-transform: uppercase;
    padding: 4px 12px; border-radius: 2px; margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(2.8rem, 7vw, 5.5rem);
    font-weight: 800; color: #f1f5f9;
    line-height: 1; letter-spacing: -1px; margin: 0 0 0.5rem;
}
.hero-title .accent  { color: #0ea5e9; }
.hero-title .accent2 { color: #fbbf24; }
.hero-divider {
    width: 48px; height: 3px;
    background: linear-gradient(90deg, #0ea5e9, #fbbf24);
    border-radius: 2px; margin: 1rem 0;
}
.hero-sub { font-size: 0.9rem; color: #475569; max-width: 480px; }

/* ── Section label ── */
.sec-label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.58rem; font-weight: 500; letter-spacing: 3px;
    text-transform: uppercase; color: #1e3a5f;
    margin-bottom: 0.8rem; margin-top: 2rem;
    display: flex; align-items: center; gap: 8px;
}
.sec-label::before {
    content: ''; display: inline-block;
    width: 16px; height: 1px; background: #0ea5e9;
}

/* ── Class cards ── */
.class-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 8px; margin-bottom: 1.5rem;
}
.class-card {
    background: #0d1829; border: 1px solid #0f2040;
    border-radius: 8px; padding: 0.9rem 1rem;
    position: relative; overflow: hidden;
}
.class-card::before {
    content: ''; position: absolute;
    left:0; top:0; bottom:0; width:3px;
}
.class-card.critical::before { background: #ef4444; }
.class-card.warning::before  { background: #fbbf24; }
.class-card.safe::before     { background: #22c55e; }
.cc-icon  { font-size: 1.3rem; margin-bottom: 0.4rem; }
.cc-en    { font-family: 'DM Mono', monospace !important; font-size: 0.62rem; color: #334155; margin-bottom: 2px; }
.cc-ar    { font-family: 'Syne', sans-serif !important; font-size: 0.9rem; font-weight: 700; color: #e2e8f0; margin-bottom: 4px; }
.cc-desc  { font-size: 0.72rem; color: #334155; line-height: 1.4; }

/* ── Upload area ── */
.upload-box {
    background: #0d1829;
    border: 1px dashed #0f2040;
    border-radius: 10px;
    padding: 1.8rem;
    text-align: center;
    margin-bottom: 8px;
}
.upload-box-icon { font-size: 2rem; margin-bottom: 8px; }
.upload-box-text {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem; color: #1e3a5f;
    letter-spacing: 2px; text-transform: uppercase;
}
.upload-box-sub {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.55rem; color: #0f2040;
    margin-top: 4px; letter-spacing: 1px;
}

/* ── Hide native uploader chrome, keep functional ── */
[data-testid="stFileUploader"] > div {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
    min-height: unset !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: none !important;
}
/* Kill ALL dropzone buttons then show only first */
[data-testid="stFileUploaderDropzone"] button {
    display: none !important;
}
[data-testid="stFileUploaderDropzone"] button:first-of-type {
    display: inline-flex !important;
    background: #0d1829 !important;
    color: #0ea5e9 !important;
    border: 1px solid #0f2040 !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 1px !important;
    padding: 0.45rem 1.4rem !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
}
[data-testid="stFileUploaderDropzone"] button:first-of-type:hover {
    border-color: #0ea5e9 !important;
    background: #0f2040 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #0d1829 !important; color: #475569 !important;
    border: 1px solid #0f2040 !important; border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important; letter-spacing: 1.5px !important;
    text-transform: uppercase !important; padding: 0.5rem 1.2rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #0f2040 !important;
    border-color: #0ea5e9 !important; color: #0ea5e9 !important;
}

/* ── Divider ── */
.div-line { border:none; border-top:1px solid #0f1e35; margin:2.5rem 0; }

/* ── Result card ── */
.r-card {
    background: #0d1829; border: 1px solid #0f2040;
    border-radius: 10px; padding: 1.2rem 1.4rem; margin-bottom: 0.8rem;
}
.r-filename {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.62rem; color: #1e3a5f; margin-bottom: 0.8rem;
}
.img-lbl {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.52rem; letter-spacing: 3px;
    text-transform: uppercase; color: #1e3a5f; margin-bottom: 5px;
}

/* ── Badges ── */
.badge-critical {
    display:inline-block; background:rgba(239,68,68,0.08);
    color:#f87171; border:1px solid rgba(239,68,68,0.18);
    border-radius:3px; padding:2px 8px;
    font-family:'DM Mono',monospace !important; font-size:0.62rem;
}
.badge-warning {
    display:inline-block; background:rgba(251,191,36,0.08);
    color:#fbbf24; border:1px solid rgba(251,191,36,0.18);
    border-radius:3px; padding:2px 8px;
    font-family:'DM Mono',monospace !important; font-size:0.62rem;
}
.badge-safe {
    display:inline-block; background:rgba(34,197,94,0.08);
    color:#4ade80; border:1px solid rgba(34,197,94,0.18);
    border-radius:3px; padding:2px 8px;
    font-family:'DM Mono',monospace !important; font-size:0.62rem;
}

/* ── Detection rows ── */
.det-row {
    display:flex; align-items:center; justify-content:space-between;
    padding:5px 0; border-bottom:1px solid #0a1525;
}
.det-row:last-child { border-bottom:none; }
.det-conf { font-family:'DM Mono',monospace !important; color:#1e3a5f; font-size:0.68rem; }
.no-det { font-family:'DM Mono',monospace !important; font-size:0.62rem; color:#1e3a5f; }

/* ── Alerts ── */
.alert-crit {
    background:rgba(239,68,68,0.05); border-left:2px solid #ef4444;
    border-radius:4px; padding:5px 8px; margin:3px 0;
    color:#f87171; font-size:0.7rem;
    font-family:'DM Mono',monospace !important;
}
.alert-warn {
    background:rgba(251,191,36,0.05); border-left:2px solid #fbbf24;
    border-radius:4px; padding:5px 8px; margin:3px 0;
    color:#fbbf24; font-size:0.7rem;
    font-family:'DM Mono',monospace !important;
}

/* ── Report ── */
.report-title {
    font-family:'Syne',sans-serif !important;
    font-size:3rem; font-weight:800; color:#f1f5f9;
    letter-spacing:-1px; line-height:1; margin-bottom:0.3rem;
}
.report-sub {
    font-family:'DM Mono',monospace !important;
    font-size:0.6rem; color:#1e3a5f; letter-spacing:2px; text-transform:uppercase;
}
.score-wrap { text-align:center; padding:1.5rem 0; }
.score-num {
    font-family:'Syne',sans-serif !important;
    font-size:5.5rem; font-weight:800; line-height:1; letter-spacing:-2px;
}
.score-denom {
    font-family:'DM Mono',monospace !important;
    font-size:0.6rem; color:#1e3a5f; letter-spacing:2px;
    margin-top:4px; text-transform:uppercase;
}
.mbox {
    background:#0d1829; border:1px solid #0f2040;
    border-radius:8px; padding:0.9rem 0.8rem; text-align:center;
}
.mval {
    font-family:'Syne',sans-serif !important;
    font-size:2.2rem; font-weight:800; line-height:1;
    letter-spacing:-1px; margin-bottom:4px;
}
.mlbl {
    font-family:'DM Mono',monospace !important;
    font-size:0.52rem; letter-spacing:2px;
    text-transform:uppercase; color:#1e3a5f;
}
.verdict {
    font-family:'Syne',sans-serif !important;
    text-align:center; font-size:0.95rem; font-weight:700;
    letter-spacing:3px; padding:0.6rem 1.5rem; border-radius:6px;
    margin:1rem auto; max-width:260px; display:block; text-transform:uppercase;
}
.v-safe     { background:rgba(34,197,94,0.06);  color:#4ade80; border:1px solid rgba(34,197,94,0.12); }
.v-warning  { background:rgba(251,191,36,0.06); color:#fbbf24; border:1px solid rgba(251,191,36,0.12); }
.v-critical { background:rgba(239,68,68,0.06);  color:#f87171; border:1px solid rgba(239,68,68,0.12); }

/* ── Timeline ── */
.tl-wrap {
    background:#0d1829; border:1px solid #0f2040;
    border-radius:8px; padding:0.8rem 1rem;
}
.tl-item {
    display:flex; align-items:flex-start; gap:10px;
    padding:7px 0; border-bottom:1px solid #0a1525;
}
.tl-item:last-child { border-bottom:none; }
.tl-dot-c { width:7px;height:7px;background:#ef4444;border-radius:50%;margin-top:5px;flex-shrink:0; }
.tl-dot-w { width:7px;height:7px;background:#fbbf24;border-radius:50%;margin-top:5px;flex-shrink:0; }
.tl-cls { font-weight:600; font-size:0.78rem; color:#94a3b8; }
.tl-meta { font-family:'DM Mono',monospace !important; font-size:0.6rem; color:#1e3a5f; margin-top:1px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background:#080e1a !important; border-right:1px solid #0a1525; }
[data-testid="stSidebar"] * { color:#334155; }

/* ── Footer ── */
.footer {
    text-align:center; font-family:'DM Mono',monospace !important;
    color:#0a1525; font-size:0.6rem; padding:3rem 0 1rem;
    letter-spacing:2px; text-transform:uppercase;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HERO
# ==========================================
st.markdown("""
<div class="hero">
    <div class="hero-bg-line"></div>
    <div class="hero-inner">
        <div class="hero-chip">🛡️ AI Safety System · YOLO11s</div>
        <div class="hero-title">
            Driver<br><span class="accent">Safety</span> <span class="accent2">Monitor</span>
        </div>
        <div class="hero-divider"></div>
        <p class="hero-sub">رفع صور السائق — النظام يحلل السلوك ويكتشف المخاطر تلقائياً</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# CLASS CARDS
# ==========================================
st.markdown('<div class="sec-label">Classes — ما يكتشفه النظام</div>', unsafe_allow_html=True)

st.markdown('<div class="class-grid">', unsafe_allow_html=True)
for name, info in CLASS_INFO.items():
    lvl_cls = "critical" if info["level"] == "CRITICAL" else \
              "warning"  if info["level"] == "WARNING"  else "safe"
    st.markdown(f"""
    <div class="class-card {lvl_cls}">
        <div class="cc-icon">{info['icon']}</div>
        <div class="cc-en">{name}</div>
        <div class="cc-ar">{info['ar']}</div>
        <div class="cc-desc">{info['desc']}</div>
    </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style='padding:1.5rem 0 1rem;'>
        <div style='font-family:"DM Mono",monospace;font-size:0.58rem;letter-spacing:3px;
                    text-transform:uppercase;color:#1e3a5f;margin-bottom:1.2rem;'>// Settings</div>
    </div>
    """, unsafe_allow_html=True)

    conf_threshold = st.slider(
        "Confidence Threshold", 0.10, 1.00, 0.15, 0.05,
        help="كلما قل — زادت الاكتشافات. يُنصح: 0.15–0.25"
    )

    st.markdown("<hr style='border-color:#0a1525;margin:1.2rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:"DM Mono",monospace;font-size:0.58rem;letter-spacing:3px;
                text-transform:uppercase;color:#1e3a5f;margin-bottom:1rem;'>// Levels</div>
    """, unsafe_allow_html=True)

    for lvl, cls, names in [
        ("CRITICAL", "badge-critical", "DangerousDriving · SleepyDriving · Drinking"),
        ("WARNING",  "badge-warning",  "Distracted · Yawn"),
        ("SAFE",     "badge-safe",     "SafeDriving"),
    ]:
        st.markdown(f"""
        <div style="margin-bottom:12px">
            <span class="{cls}">{lvl}</span>
            <div style="font-family:'DM Mono',monospace;font-size:0.52rem;color:#1e3a5f;
                        margin-top:4px;">{names}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#0a1525;margin:1.2rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:"DM Mono",monospace;color:#1e3a5f;
                font-size:0.52rem;letter-spacing:1.5px;'>MODEL: YOLO11s · v1.0</div>
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
                    "timestamp":  now, "class": name, "level": level,
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
st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
st.markdown('<div class="sec-label">Upload — رفع الصور</div>', unsafe_allow_html=True)

st.markdown("""
<div class="upload-box">
    <div class="upload-box-icon">📂</div>
    <div class="upload-box-text">اسحب الصور هنا أو اضغط Browse</div>
    <div class="upload-box-sub">JPG · JPEG · PNG</div>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    label="",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

col_a, col_b, _ = st.columns([1, 1, 6])
with col_a:
    analyze_btn = st.button("▶ ANALYZE", use_container_width=True)
with col_b:
    clear_btn   = st.button("✕ CLEAR",   use_container_width=True)

if clear_btn:
    st.session_state.alert_log  = []
    st.session_state.counts     = {n: 0 for n in CLASS_NAMES}
    st.session_state.last_alert = {}
    st.session_state.results    = []
    st.rerun()

# ==========================================
# ANALYSIS
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
            "name": f.name, "original": img,
            "output": out_rgb, "detections": dets, "alerts": alerts,
        })
        progress.progress((i+1)/len(uploaded_files))
    progress.empty()
elif analyze_btn:
    st.warning("ارفع صورة واحدة على الأقل أولاً.")

# ==========================================
# RESULTS
# ==========================================
if st.session_state.results:
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Results — نتائج الفحص</div>', unsafe_allow_html=True)

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
                st.markdown("<div class='no-det'>— لا يوجد اكتشافات —</div>", unsafe_allow_html=True)
            if res["alerts"]:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                for alert in res["alerts"]:
                    css = "alert-crit" if alert["level"] == "CRITICAL" else "alert-warn"
                    ico = "⚠" if alert["level"] == "CRITICAL" else "~"
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

    v_text  = "SAFE DRIVER"     if score >= 85 else \
              "NEEDS ATTENTION" if score >= 60 else "DANGEROUS DRIVER"
    v_emoji = "✓" if score >= 85 else "!" if score >= 60 else "✕"
    v_cls   = "v-safe" if score >= 85 else "v-warning" if score >= 60 else "v-critical"
    s_color = "#4ade80" if score >= 85 else "#fbbf24" if score >= 60 else "#f87171"

    st.markdown(f"""
    <div style='margin-bottom:2rem;'>
        <div class='sec-label' style='margin-top:0;'>Report — تقرير الجلسة</div>
        <div class='report-title'>Safety<br>Analysis</div>
        <div class='report-sub'>{datetime.now().strftime("%Y · %m · %d — %H:%M:%S")} · {len(st.session_state.results)} frame(s)</div>
    </div>
    """, unsafe_allow_html=True)

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
            (r1c1, len(log),      "Total",    "#94a3b8"),
            (r1c2, critical,      "Critical", "#f87171"),
            (r1c3, warning,       "Warnings", "#fbbf24"),
            (r1c4, len(st.session_state.results), "Frames", "#1e3a5f"),
        ]:
            with col:
                st.markdown(
                    f'<div class="mbox"><div class="mval" style="color:{clr}">{val}</div>'
                    f'<div class="mlbl">{lbl}</div></div>',
                    unsafe_allow_html=True
                )

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        cls_cols = st.columns(len(CLASS_NAMES))
        for col, cls_name in zip(cls_cols, CLASS_NAMES):
            cnt  = counts.get(cls_name, 0)
            info = CLASS_INFO[cls_name]
            clr  = "#f87171" if info["level"] == "CRITICAL" else \
                   "#fbbf24" if info["level"] == "WARNING"  else "#1e3a5f"
            with col:
                st.markdown(
                    f'<div class="mbox" style="padding:0.6rem 0.4rem">'
                    f'<div style="font-size:1rem;margin-bottom:2px">{info["icon"]}</div>'
                    f'<div class="mval" style="color:{clr};font-size:1.5rem">{cnt}</div>'
                    f'<div class="mlbl" style="font-size:0.44rem">{info["ar"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    if counts:
        col_chart, col_tl = st.columns([1.3, 1])
        with col_chart:
            st.markdown('<div class="sec-label" style="margin-top:0">Alert Breakdown</div>',
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, max(2, len(counts) * 0.75)))
            fig.patch.set_facecolor("#0d1829")
            ax.set_facecolor("#0d1829")
            clrs = ["#ef4444" if c in DANGER_CLASSES else
                    "#fbbf24" if c in WARNING_CLASSES else
                    "#22c55e" for c in counts.keys()]
            bars = ax.barh(list(counts.keys()), list(counts.values()),
                           color=clrs, height=0.42, edgecolor="none")
            ax.set_xlabel("Alerts", color="#1e3a5f", fontsize=7, fontfamily="monospace")
            ax.tick_params(colors="#1e3a5f", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#0a1525")
            ax.xaxis.grid(True, color="#0a1525", zorder=0, linestyle="--", linewidth=0.5)
            ax.set_axisbelow(True)
            for bar, val in zip(bars, counts.values()):
                ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                        str(val), va='center', color="#334155",
                        fontsize=7, fontfamily="monospace")
            plt.tight_layout(pad=0.8)
            st.pyplot(fig)
            plt.close()

        with col_tl:
            st.markdown('<div class="sec-label" style="margin-top:0">Alert Log</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="tl-wrap">', unsafe_allow_html=True)
            for alert in log[-12:]:
                dot = "tl-dot-c" if alert["level"] == "CRITICAL" else "tl-dot-w"
                t   = alert.get("time_str", "—")
                c   = f"{alert['confidence']:.0%}"
                ico = CLASS_INFO[alert["class"]]["icon"]
                st.markdown(f"""
                <div class="tl-item">
                    <div class="{dot}"></div>
                    <div>
                        <div class="tl-cls">{ico} {alert['class']}</div>
                        <div class="tl-meta">{t}  ·  conf {c}  ·  {alert['level']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

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
            label="⬇ DOWNLOAD REPORT",
            data=json.dumps(report_data, indent=4),
            file_name=f"driver_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_clr:
        if st.button("✕ CLEAR REPORT", use_container_width=True):
            st.session_state.alert_log  = []
            st.session_state.counts     = {n: 0 for n in CLASS_NAMES}
            st.session_state.last_alert = {}
            st.session_state.results    = []
            st.rerun()

elif st.session_state.results:
    st.markdown("<div class='div-line'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;padding:3rem 0;'>
        <div style='font-size:3rem;margin-bottom:0.5rem;'>✅</div>
        <div style='font-family:"Syne",sans-serif;font-size:2.5rem;font-weight:800;
                    color:#4ade80;letter-spacing:-1px;'>ALL CLEAR</div>
        <div style='font-family:"DM Mono",monospace;font-size:0.62rem;color:#1e3a5f;
                    letter-spacing:2px;margin-top:0.5rem;text-transform:uppercase;'>
            لا يوجد سلوك خطير
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================
st.markdown(
    "<div class='footer'>Driver Safety Monitor · YOLO11s · Streamlit · 2025</div>",
    unsafe_allow_html=True
)
