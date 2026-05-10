import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DrowsyGuard · AI Driver Monitor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CINEMATIC CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap');

/* ── RESET & BASE ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #030508 !important;
    color: #c8d8e8 !important;
    font-family: 'Rajdhani', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, #0a1628 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, #0d1f1a 0%, transparent 55%),
        #030508 !important;
    min-height: 100vh;
}

/* hide streamlit chrome */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { visibility: hidden !important; height: 0 !important; }

/* ── SCANLINE OVERLAY ── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.03) 2px,
        rgba(0,0,0,0.03) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── SIDEBAR STYLE ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06101e 0%, #030810 100%) !important;
    border-right: 1px solid rgba(62,207,175,0.15) !important;
}
[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #3ecfaf, transparent);
}
[data-testid="stSidebar"] * {
    color: #c8d8e8 !important;
}

/* ── HIDE X BUTTON ON FILE UPLOADER ── */
[data-testid="stFileUploaderDeleteBtn"],
button[title="Remove file"],
[data-testid="fileDeleteBtn"],
.st-emotion-cache-1erivf3,
[aria-label="Remove file"] {
    display: none !important;
    visibility: hidden !important;
}

/* also target by class patterns Streamlit uses */
[data-testid="stFileUploader"] button {
    display: none !important;
}

/* ── MAIN BLOCK ── */
.block-container {
    max-width: 1400px !important;
    padding: 2rem 3rem !important;
}

/* ── HERO TITLE ── */
.hero-wrap {
    text-align: center;
    padding: 3rem 0 2rem;
    position: relative;
}
.hero-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.4em;
    color: #3ecfaf;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    opacity: 0.8;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3.5rem, 8vw, 7rem);
    letter-spacing: 0.08em;
    line-height: 0.9;
    background: linear-gradient(135deg, #ffffff 0%, #8ecfff 50%, #3ecfaf 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 40px rgba(62,207,175,0.3));
}
.hero-sub {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    letter-spacing: 0.2em;
    color: #4a6fa5;
    margin-top: 0.75rem;
    text-transform: uppercase;
}
.hero-line {
    width: 120px; height: 1px;
    background: linear-gradient(90deg, transparent, #3ecfaf, transparent);
    margin: 1.5rem auto 0;
}

/* ── UPLOAD ZONE ── */
.upload-section {
    background: linear-gradient(135deg, rgba(10,22,40,0.9) 0%, rgba(8,18,32,0.95) 100%);
    border: 1px solid rgba(62,207,175,0.2);
    border-radius: 16px;
    padding: 2rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 2rem;
    box-shadow:
        0 0 60px rgba(62,207,175,0.05),
        inset 0 1px 0 rgba(255,255,255,0.05);
}
.upload-section::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #3ecfaf 50%, transparent);
}

/* ── STREAMLIT UPLOADER OVERRIDE ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploader"] > div {
    background: rgba(62,207,175,0.03) !important;
    border: 2px dashed rgba(62,207,175,0.25) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    padding: 2rem !important;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: rgba(62,207,175,0.6) !important;
    background: rgba(62,207,175,0.07) !important;
    box-shadow: 0 0 30px rgba(62,207,175,0.1) !important;
}
[data-testid="stFileUploader"] label {
    display: none !important;
}
[data-testid="stFileUploaderDropzone"] span {
    color: #4a6fa5 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* ── SECTION HEADERS ── */
.section-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    letter-spacing: 0.15em;
    color: #8ecfff;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(62,207,175,0.15);
}
.section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #3ecfaf;
    box-shadow: 0 0 12px #3ecfaf;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.7); }
}

/* ── IMAGE CARDS ── */
.img-card-wrap {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(62,207,175,0.15);
    background: rgba(10,22,40,0.8);
    transition: all 0.3s ease;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.img-card-wrap:hover {
    border-color: rgba(62,207,175,0.4);
    box-shadow: 0 8px 40px rgba(62,207,175,0.1);
    transform: translateY(-2px);
}

/* ── RESULT PAIR ROW ── */
.result-pair-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-bottom: 2rem;
    background: rgba(10,22,40,0.5);
    border: 1px solid rgba(62,207,175,0.12);
    border-radius: 14px;
    padding: 1.25rem;
}
.result-pair-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    color: #3ecfaf;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    opacity: 0.7;
}

/* ── PREDICTION CARDS ── */
.pred-card {
    background: linear-gradient(135deg, rgba(10,22,40,0.95) 0%, rgba(5,14,26,0.98) 100%);
    border: 1px solid rgba(62,207,175,0.2);
    border-radius: 14px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 30px rgba(0,0,0,0.5);
    height: 100%;
}
.pred-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent-color, #3ecfaf);
    box-shadow: 0 0 20px var(--accent-color, #3ecfaf);
}
.pred-filename {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #4a6fa5;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
}
.pred-class-name {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    letter-spacing: 0.1em;
    line-height: 1;
    color: var(--accent-color, #3ecfaf);
    filter: drop-shadow(0 0 20px var(--accent-color, #3ecfaf));
    margin-bottom: 0.25rem;
}
.pred-confidence {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.9rem;
    color: #6a8faf;
}
.pred-bar-bg {
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    height: 4px;
    margin-top: 1rem;
    overflow: hidden;
}
.pred-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: var(--accent-color, #3ecfaf);
    box-shadow: 0 0 10px var(--accent-color, #3ecfaf);
    transition: width 1s ease;
}
.pred-icon {
    position: absolute;
    top: 1.25rem;
    right: 1.25rem;
    font-size: 2.5rem;
    opacity: 0.15;
}

/* ── STATUS BADGE ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.2rem 0.8rem;
    border-radius: 100px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
}
.badge-safe    { background: rgba(62,207,175,0.12); color: #3ecfaf; border: 1px solid rgba(62,207,175,0.3); }
.badge-warning { background: rgba(255,190,60,0.12); color: #ffbe3c; border: 1px solid rgba(255,190,60,0.3); }
.badge-danger  { background: rgba(255,70,70,0.12);  color: #ff4646; border: 1px solid rgba(255,70,70,0.3); }

/* ── DASHBOARD ── */
.dashboard-wrap {
    background: linear-gradient(135deg, rgba(6,14,28,0.98) 0%, rgba(4,10,20,0.99) 100%);
    border: 1px solid rgba(62,207,175,0.15);
    border-radius: 16px;
    padding: 2rem;
    margin-top: 2.5rem;
    position: relative;
    overflow: hidden;
}
.dashboard-wrap::before {
    content: 'DRIVER STATUS MONITOR';
    position: absolute;
    top: 1.5rem; right: 1.5rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    color: rgba(62,207,175,0.25);
}
.dash-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
    margin-top: 1.25rem;
}
.dash-metric {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 1.25rem;
    text-align: center;
    transition: all 0.3s ease;
}
.dash-metric:hover {
    background: rgba(62,207,175,0.05);
    border-color: rgba(62,207,175,0.2);
}
.dash-metric-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.8rem;
    line-height: 1;
    letter-spacing: 0.05em;
    color: #ffffff;
    filter: drop-shadow(0 0 15px rgba(142,207,255,0.4));
}
.dash-metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    color: #4a6fa5;
    text-transform: uppercase;
    margin-top: 0.4rem;
}
.dash-metric-sub {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.85rem;
    color: var(--m-color, #3ecfaf);
    margin-top: 0.2rem;
}

/* ── ALERT BAR ── */
.alert-bar {
    border-radius: 10px;
    padding: 1rem 1.5rem;
    margin-top: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    letter-spacing: 0.05em;
    font-weight: 600;
}
.alert-safe    { background: rgba(62,207,175,0.08); border: 1px solid rgba(62,207,175,0.25); color: #3ecfaf; }
.alert-caution { background: rgba(255,190,60,0.08); border: 1px solid rgba(255,190,60,0.25); color: #ffbe3c; }
.alert-critical{ background: rgba(255,70,70,0.08);  border: 1px solid rgba(255,70,70,0.25);  color: #ff4646; }

/* ── BUTTONS ── */
[data-testid="stButton"] > button {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.2em !important;
    font-size: 1.1rem !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    border: none !important;
    padding: 0.7rem 2rem !important;
    cursor: pointer !important;
}

/* Primary – Analyze */
[data-testid="stButton"]:first-of-type > button {
    background: linear-gradient(135deg, #1a4a3a 0%, #0d2d24 100%) !important;
    color: #3ecfaf !important;
    box-shadow: 0 0 0 1px rgba(62,207,175,0.3), 0 4px 20px rgba(62,207,175,0.1) !important;
}
[data-testid="stButton"]:first-of-type > button:hover {
    box-shadow: 0 0 0 1px rgba(62,207,175,0.7), 0 8px 30px rgba(62,207,175,0.25) !important;
    transform: translateY(-1px) !important;
    color: #7fffd4 !important;
}

/* ── SLIDER (Threshold) ── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #3ecfaf, #8ecfff) !important;
}
[data-testid="stSlider"] [role="slider"] {
    background: #3ecfaf !important;
    border: 2px solid #0d2d24 !important;
    box-shadow: 0 0 12px rgba(62,207,175,0.6) !important;
    width: 20px !important;
    height: 20px !important;
}
[data-testid="stSlider"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.15em !important;
    color: #3ecfaf !important;
    text-transform: uppercase !important;
}

/* ── SPINNER OVERRIDE ── */
[data-testid="stSpinner"] > div {
    border-color: #3ecfaf transparent transparent transparent !important;
}

/* ── DIVIDER ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(62,207,175,0.1) !important;
    margin: 2rem 0 !important;
}

/* ── COLUMNS ── */
[data-testid="stHorizontalBlock"] { gap: 1.5rem !important; }

/* ── SIDEBAR CLASS CARDS ── */
.class-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(62,207,175,0.12);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
    border-left: 3px solid var(--cls-color, #3ecfaf);
}
.class-card-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem;
    letter-spacing: 0.1em;
    color: var(--cls-color, #3ecfaf);
    display: flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 0.3rem;
}
.class-card-desc {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.78rem;
    color: #6a8faf;
    line-height: 1.4;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
CLASS_NAMES = [
    "DangerousDriving",
    "Distracted",
    "Drinking",
    "SafeDriving",
    "SleepyDriving",
    "Yawn",
]

CLASS_META = {
    "DangerousDriving": {"icon": "⚡", "color": "#ff4646", "level": "critical",  "label": "CRITICAL"},
    "Distracted":       {"icon": "👁", "color": "#ffbe3c", "level": "caution",  "label": "CAUTION"},
    "Drinking":         {"icon": "🥤", "color": "#ff8c42", "level": "caution",  "label": "CAUTION"},
    "SafeDriving":      {"icon": "✅", "color": "#3ecfaf", "level": "safe",     "label": "SAFE"},
    "SleepyDriving":    {"icon": "😴", "color": "#b46fff", "level": "critical", "label": "CRITICAL"},
    "Yawn":             {"icon": "🥱", "color": "#ffbe3c", "level": "caution",  "label": "CAUTION"},
}

CLASS_DESCRIPTIONS = {
    "DangerousDriving": {
        "color": "#ff4646",
        "icon": "⚡",
        "desc": "Driver is behaving aggressively or recklessly — think sudden lane cuts, tailgating, or dangerously high speeds."
    },
    "Distracted": {
        "color": "#ffbe3c",
        "icon": "👁",
        "desc": "Driver's attention is off the road — could be on a phone, talking to a passenger, eating, or anything pulling focus away from driving."
    },
    "Drinking": {
        "color": "#ff8c42",
        "icon": "🥤",
        "desc": "Driver is consuming a drink while behind the wheel, which takes at least one hand off and diverts attention from the road."
    },
    "SafeDriving": {
        "color": "#3ecfaf",
        "icon": "✅",
        "desc": "Driver is focused, hands on the wheel, eyes on the road — no risky behaviour detected."
    },
    "SleepyDriving": {
        "color": "#b46fff",
        "icon": "😴",
        "desc": "Driver is showing signs of fatigue or drowsiness — one of the most dangerous states on the road as reaction time drops significantly."
    },
    "Yawn": {
        "color": "#ffbe3c",
        "icon": "🥱",
        "desc": "Driver is yawning, which is usually an early warning sign of tiredness and can lead to drowsy driving if ignored."
    },
}


# ─────────────────────────────────────────────
#  SIDEBAR — Class Descriptions + Threshold
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 0.5rem;">
        <div style="font-family:'Bebas Neue',sans-serif; font-size:1.5rem; letter-spacing:0.2em; color:#3ecfaf;">
            CLASS GUIDE
        </div>
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem; letter-spacing:0.2em; color:#4a6fa5; margin-top:0.2rem;">
            DETECTION CATEGORIES
        </div>
        <div style="height:1px; background:linear-gradient(90deg,transparent,#3ecfaf,transparent); margin:0.8rem 0;"></div>
    </div>
    """, unsafe_allow_html=True)

    for cls_name, cls_info in CLASS_DESCRIPTIONS.items():
        st.markdown(f"""
        <div class="class-card" style="--cls-color:{cls_info['color']}">
            <div class="class-card-title">
                {cls_info['icon']} {cls_name}
            </div>
            <div class="class-card-desc">{cls_info['desc']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="height:1px; background:linear-gradient(90deg,transparent,#3ecfaf,transparent); margin:1rem 0;"></div>
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; letter-spacing:0.2em; color:#3ecfaf; text-align:center; margin-bottom:0.5rem;">
        ⚙ DETECTION THRESHOLD
    </div>
    """, unsafe_allow_html=True)

    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.35,
        step=0.05,
        format="%.2f",
        label_visibility="collapsed",
    )

    st.markdown(f"""
    <div style="text-align:center; font-family:'Bebas Neue',sans-serif; font-size:1.6rem;
                color:#3ecfaf; filter:drop-shadow(0 0 10px rgba(62,207,175,0.5)); margin-top:0.2rem;">
        {int(threshold*100)}%
    </div>
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.6rem; color:#4a6fa5;
                text-align:center; letter-spacing:0.15em;">
        MIN CONFIDENCE TO ACCEPT
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  YOLO MODEL  (cached so it loads only once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("best.pt")


# ─────────────────────────────────────────────
#  REAL INFERENCE  (uses sidebar threshold)
# ─────────────────────────────────────────────
def run_inference(image: Image.Image, conf_threshold: float = 0.35) -> dict:
    import numpy as np
    model = load_model()

    # First try with the user threshold
    results = model.predict(image, imgsz=640, conf=conf_threshold, verbose=False)
    boxes = results[0].boxes

    if boxes is not None and len(boxes):
        confs    = boxes.conf
        best_idx = int(confs.argmax().item())
        cls      = int(boxes.cls[best_idx].item())
        conf     = float(boxes.conf[best_idx].item())
        # Draw bounding box using YOLO's built-in plot
        annotated = results[0].plot(
            line_width=2,
            font_size=12,
            labels=True,
            conf=True,
        )
        annotated_pil = Image.fromarray(annotated[..., ::-1])  # BGR→RGB
        return {"class": CLASS_NAMES[cls], "confidence": conf,
                "annotated": annotated_pil, "below_threshold": False}

    # Fallback: run with very low threshold to still get best guess
    results2 = model.predict(image, imgsz=640, conf=0.01, verbose=False)
    boxes2   = results2[0].boxes
    if boxes2 is not None and len(boxes2):
        confs2   = boxes2.conf
        best_idx2 = int(confs2.argmax().item())
        cls      = int(boxes2.cls[best_idx2].item())
        conf     = float(boxes2.conf[best_idx2].item())
        annotated = results2[0].plot(line_width=2, font_size=12, labels=True, conf=True)
        annotated_pil = Image.fromarray(annotated[..., ::-1])
        return {"class": CLASS_NAMES[cls], "confidence": conf,
                "annotated": annotated_pil, "below_threshold": True}

    # Truly no detection
    return {"class": "SafeDriving", "confidence": 0.0,
            "annotated": image, "below_threshold": True}


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "predictions" not in st.session_state:
    st.session_state.predictions = []
if "show_results" not in st.session_state:
    st.session_state.show_results = False
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-label">▸ Neural Vision System · v2.4</div>
    <div class="hero-title">DrowsyGuard</div>
    <div class="hero-sub">AI-Powered Driver Behaviour Detection</div>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  UPLOAD ZONE
# ─────────────────────────────────────────────
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown("""
<div class="section-header">
    <span class="section-dot"></span>
    IMAGE UPLOAD
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "upload_hidden",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    label_visibility="hidden",
    key=f"uploader_{st.session_state.uploader_key}",
)
st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PREVIEW UPLOADED IMAGES
# ─────────────────────────────────────────────
if uploaded_files:
    st.markdown("""
    <div class="section-header" style="margin-top:1rem;">
        <span class="section-dot"></span>
        LOADED FRAMES
    </div>
    """, unsafe_allow_html=True)

    # Always 7 columns — images stay small regardless of count
    THUMB_COLS = 7
    cols = st.columns(THUMB_COLS)
    for i, f in enumerate(uploaded_files):
        with cols[i % THUMB_COLS]:
            img = Image.open(f)
            st.markdown('<div class="img-card-wrap">', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(
                f'<p style="font-family:\'Share Tech Mono\',monospace;font-size:0.55rem;'
                f'color:#4a6fa5;text-align:center;margin-top:0.25rem;letter-spacing:0.03em;'
                f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'
                f'{f.name[:16]}{"…" if len(f.name)>16 else ""}</p>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ACTION BUTTONS ──────────────────────────
    col_a, col_b, col_c = st.columns([2, 1.2, 4])

    with col_a:
        analyze = st.button("⬡  ANALYZE FRAMES", use_container_width=True)

    with col_b:
        if st.button("✕  CLEAR ALL", use_container_width=True):
            st.session_state.predictions = []
            st.session_state.show_results = False
            st.session_state.uploader_key += 1   # resets the file uploader widget
            st.rerun()

    # ── RUN INFERENCE ────────────────────────────
    if analyze:
        preds = []
        with st.spinner("Running neural analysis…"):
            for f in uploaded_files:
                img = Image.open(f).convert("RGB")
                result = run_inference(img, conf_threshold=threshold)
                preds.append({
                    "filename": f.name,
                    "image":    img,
                    "annotated": result.get("annotated", img),
                    "class":    result["class"],
                    "confidence": result["confidence"],
                    "below_threshold": result.get("below_threshold", False),
                })
        st.session_state.predictions = preds
        st.session_state.show_results = True
        st.rerun()


# ─────────────────────────────────────────────
#  PREDICTIONS  — Original + Predicted side-by-side
# ─────────────────────────────────────────────
if st.session_state.show_results and st.session_state.predictions:
    preds = st.session_state.predictions

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-header">
        <span class="section-dot"></span>
        ANALYSIS RESULTS
    </div>
    """, unsafe_allow_html=True)

    for p in preds:
        meta = CLASS_META[p["class"]]
        conf_pct = int(p["confidence"] * 100)

        col_orig, col_pred = st.columns(2)

        # ── Original image ──
        with col_orig:
            st.markdown('<div class="result-pair-label">📷 ORIGINAL FRAME</div>', unsafe_allow_html=True)
            st.markdown('<div class="img-card-wrap">', unsafe_allow_html=True)
            st.image(p["image"], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(
                f'<p style="font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;'
                f'color:#4a6fa5;text-align:center;margin-top:0.3rem;letter-spacing:0.05em;">'
                f'{p["filename"][:32]}{"…" if len(p["filename"])>32 else ""}</p>',
                unsafe_allow_html=True,
            )

        # ── Prediction column: bounding box image + card ──
        with col_pred:
            st.markdown('<div class="result-pair-label">🤖 PREDICTION · BOUNDING BOX</div>', unsafe_allow_html=True)

            # Annotated image with bbox on top
            st.markdown('<div class="img-card-wrap" style="border-color:' + meta["color"] + '55;'
                        'box-shadow:0 0 20px ' + meta["color"] + '22;">', unsafe_allow_html=True)
            st.image(p["annotated"], use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Prediction info card below the image
            below = p.get("below_threshold", False)
            low_conf_badge = (
                '<div style="margin-top:0.5rem;font-family:\'Share Tech Mono\',monospace;'
                'font-size:0.65rem;color:#ff8c42;letter-spacing:0.08em;">'
                '⚠ BELOW THRESHOLD — LOW CONFIDENCE</div>'
            ) if below else ""
            st.markdown(
                '<div style="background:rgba(10,22,40,0.8);border:1px solid ' + meta["color"] + '44;'
                'border-radius:10px;padding:0.75rem 1rem;margin-top:0.5rem;">'
                '<div style="font-family:\'Bebas Neue\',sans-serif;font-size:1.6rem;'
                'letter-spacing:0.1em;color:' + meta["color"] + ';'
                'filter:drop-shadow(0 0 12px ' + meta["color"] + '88);">' + p["class"] + '</div>'
                '<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.85rem;color:#6a8faf;">'
                'Confidence: ' + str(conf_pct) + '%</div>'
                '<span class="status-badge badge-' + ('safe' if meta['level']=='safe' else 'warning' if meta['level']=='caution' else 'danger') + '">'
                '● ' + meta["label"] + '</span>'
                + low_conf_badge +
                '<div class="pred-bar-bg" style="margin-top:0.5rem;">'
                '<div class="pred-bar-fill" style="width:' + str(conf_pct) + '%;background:' + meta["color"] + ';'
                'box-shadow:0 0 8px ' + meta["color"] + ';"></div>'
                '</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr style="margin:1rem 0 1.5rem !important; opacity:0.3;">', unsafe_allow_html=True)


    # ─────────────────────────────────────────
    #  DASHBOARD
    # ─────────────────────────────────────────
    total       = len(preds)
    safe_count  = sum(1 for p in preds if CLASS_META[p["class"]]["level"] == "safe")
    caution_cnt = sum(1 for p in preds if CLASS_META[p["class"]]["level"] == "caution")
    danger_cnt  = sum(1 for p in preds if CLASS_META[p["class"]]["level"] == "critical")
    avg_conf    = round(sum(p["confidence"] for p in preds) / total * 100, 1)

    from collections import Counter
    class_counts  = Counter(p["class"] for p in preds)
    dominant_cls  = class_counts.most_common(1)[0][0]
    dominant_meta = CLASS_META[dominant_cls]
    dom_color     = dominant_meta["color"]
    dom_icon      = dominant_meta["icon"]
    dom_label     = dominant_meta["label"]
    dom_name      = dominant_cls.replace("Driving", "")

    if danger_cnt > 0:
        alert_class = "alert-critical"
        alert_icon  = "🚨"
        alert_msg   = "CRITICAL ALERT — " + str(danger_cnt) + " frame(s) show dangerous behaviour. Immediate intervention required."
    elif caution_cnt > 0:
        alert_class = "alert-caution"
        alert_icon  = "⚠️"
        alert_msg   = "CAUTION — " + str(caution_cnt) + " frame(s) show distracted or risky behaviour. Monitor closely."
    else:
        alert_class = "alert-safe"
        alert_icon  = "✅"
        alert_msg   = "ALL CLEAR — Driver behaviour appears normal across all analysed frames."

    # ── Dashboard wrapper open + header ──
    st.markdown("""
    <div class="dashboard-wrap">
        <div class="section-header" style="border-bottom:1px solid rgba(62,207,175,0.15);padding-bottom:0.5rem;">
            <span class="section-dot"></span>
            DRIVER STATUS DASHBOARD
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric tiles using st.columns (no complex f-string) ──
    m1, m2, m3, m4, m5, m6 = st.columns(6)

    def metric_html(value, label, sub, color="#8ecfff", big=False):
        size = "1.6rem" if big else "2.8rem"
        glow = color.replace("#", "")
        return f"""
        <div class="dash-metric">
            <div class="dash-metric-value" style="font-size:{size};color:{color};
                 filter:drop-shadow(0 0 15px {color}88)">{value}</div>
            <div class="dash-metric-label">{label}</div>
            <div class="dash-metric-sub" style="color:{color}">{sub}</div>
        </div>"""

    with m1:
        st.markdown(metric_html(total, "Frames Analysed", "TOTAL", "#8ecfff"), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_html(safe_count, "Safe Frames", "SAFE DRIVING", "#3ecfaf"), unsafe_allow_html=True)
    with m3:
        st.markdown(metric_html(caution_cnt, "Caution Frames", "NEEDS ATTENTION", "#ffbe3c"), unsafe_allow_html=True)
    with m4:
        st.markdown(metric_html(danger_cnt, "Critical Frames", "DANGEROUS", "#ff4646"), unsafe_allow_html=True)
    with m5:
        st.markdown(metric_html(str(avg_conf) + "%", "Avg Confidence", "MODEL CERTAINTY", "#8ecfff"), unsafe_allow_html=True)
    with m6:
        st.markdown(metric_html(dom_name, "Dominant State", dom_icon + " " + dom_label, dom_color, big=True), unsafe_allow_html=True)

    # ── Alert bar ──
    st.markdown(
        '<div class="alert-bar ' + alert_class + '">'
        '<span style="font-size:1.4rem">' + alert_icon + '</span>'
        '<span>' + alert_msg + '</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Class breakdown ──
    st.markdown("""
    <div style="margin-top:1.5rem;padding-top:1.25rem;border-top:1px solid rgba(62,207,175,0.1);">
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;letter-spacing:0.25em;
                    color:#4a6fa5;text-transform:uppercase;margin-bottom:1rem;">
            ▸ DETECTIONS PER CLASS
        </div>
    </div>
    """, unsafe_allow_html=True)

    cls_cols = st.columns(len(CLASS_NAMES))
    for col_el, cls_name in zip(cls_cols, CLASS_NAMES):
        cnt   = class_counts.get(cls_name, 0)
        meta  = CLASS_META[cls_name]
        c     = meta["color"]
        ico   = meta["icon"]
        pct   = int(cnt / total * 100) if total > 0 else 0
        opacity = "1" if cnt > 0 else "0.3"
        with col_el:
            st.markdown(
                '<div class="dash-metric" style="opacity:' + opacity + ';border-left:3px solid ' + c + ';">'
                '<div style="font-size:2rem;margin-bottom:0.2rem;">' + ico + '</div>'
                '<div class="dash-metric-value" style="font-size:2rem;color:' + c + ';'
                'filter:drop-shadow(0 0 12px ' + c + '88)">' + str(cnt) + '</div>'
                '<div class="dash-metric-label">' + cls_name.replace("Driving", " Drv") + '</div>'
                '<div style="background:rgba(255,255,255,0.05);border-radius:3px;height:3px;margin-top:0.5rem;overflow:hidden;">'
                '<div style="width:' + str(pct) + '%;height:100%;background:' + c + ';border-radius:3px;'
                'box-shadow:0 0 6px ' + c + ';"></div>'
                '</div>'
                '<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.6rem;color:#4a6fa5;margin-top:0.3rem;">' + str(pct) + '%</div>'
                '</div>',
                unsafe_allow_html=True,
            )

elif not uploaded_files:
    # Empty state
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;opacity:0.35;">
        <div style="font-size:4rem;margin-bottom:1rem;">🎬</div>
        <div style="font-family:'Bebas Neue',sans-serif;font-size:1.8rem;letter-spacing:0.2em;color:#4a6fa5;">
            UPLOAD IMAGES TO BEGIN ANALYSIS
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;letter-spacing:0.15em;color:#2a3f5a;margin-top:0.5rem;">
            SUPPORTS JPG · PNG · WEBP · MULTI-IMAGE
        </div>
    </div>
    """, unsafe_allow_html=True)
