import json
import streamlit as st
from retriever import find_best_matches
from llm_chain import generate_llm_explanation
from explanation_utils import generate_dynamic_explanation
from image_predictor import predict_disease_from_image, compute_image_weight
from image_predictor import get_loaded_model
from weather_utils import get_weather_data, calculate_risk
from graph_flow import graph
from gradcam_utils import generate_gradcam
from PIL import Image

st.set_page_config(
    page_title="Smart Crop Health Advisor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# APPLE-LEVEL / SAAS UI CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
:root {
    --bg-overlay-1: rgba(8, 12, 17, 0.76);
    --bg-overlay-2: rgba(12, 16, 22, 0.84);
    --panel: rgba(255,255,255,0.84);
    --panel-strong: rgba(255,255,255,0.92);
    --panel-soft: rgba(255,255,255,0.72);
    --stroke: rgba(255,255,255,0.30);
    --shadow: 0 18px 60px rgba(0,0,0,0.18);
    --shadow-soft: 0 10px 30px rgba(15,23,42,0.10);
    --text: #0f172a;
    --muted: #5b6472;
    --muted-2: #7a8290;
    --green: #1f8f52;
    --green-2: #31b86c;
    --gold: #d4af37;
    --gold-2: #f1d172;
    --red: #ef4444;
    --amber: #f59e0b;
    --blue: #0f766e;
    --radius-xl: 28px;
    --radius-lg: 22px;
    --radius-md: 18px;
    --radius-sm: 14px;
}

html, body, [class*="css"] {
    font-family: "Inter", "SF Pro Display", "Segoe UI", sans-serif;
}

.stApp {
    background:
        linear-gradient(135deg, var(--bg-overlay-1), var(--bg-overlay-2)),
        url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1800&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: var(--text);
}

.block-container {
    max-width: 1420px;
    padding-top: 0.7rem;
    padding-bottom: 2rem;
}

#MainMenu, footer, header {visibility: hidden;}

.main-shell {
    background: linear-gradient(180deg, rgba(255,255,255,0.56), rgba(255,255,255,0.42));
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 34px;
    padding: 1rem 1rem 1.2rem 1rem;
    box-shadow: 0 24px 80px rgba(0,0,0,0.18);
    backdrop-filter: blur(18px);
}

.hero-shell {
    position: relative;
    background: linear-gradient(135deg, rgba(11, 90, 71, 0.84), rgba(17, 33, 46, 0.78));
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 30px;
    padding: 1.25rem 1.25rem 1.15rem 1.25rem;
    box-shadow: 0 24px 60px rgba(0,0,0,0.18);
    backdrop-filter: blur(16px);
    margin-bottom: 1rem;
    animation: slideUp 0.55s ease;
}
.hero-grid {
    display: grid;
    grid-template-columns: 1.5fr 0.95fr;
    gap: 1rem;
    align-items: stretch;
}
.hero-left {
    padding: 0.3rem 0.25rem;
}
.hero-title {
    font-size: clamp(2.1rem, 4vw, 4rem);
    line-height: 0.98;
    font-weight: 900;
    color: #f8fafc;
    letter-spacing: -0.05em;
    margin-bottom: 0.9rem;
    text-shadow: 0 8px 30px rgba(0,0,0,0.18);
}
.hero-subtitle {
    font-size: 1.02rem;
    line-height: 1.72;
    color: rgba(255,255,255,0.88);
    max-width: 730px;
    margin-bottom: 0;
}
.hero-right {
    background: linear-gradient(180deg, rgba(255,255,255,0.16), rgba(255,255,255,0.08));
    border: 1px solid rgba(255,255,255,0.16);
    border-radius: 24px;
    padding: 1rem 1.05rem;
    display: flex;
    align-items: center;
    backdrop-filter: blur(12px);
}
.mini-note {
    color: rgba(255,255,255,0.94);
    font-size: 0.98rem;
    line-height: 1.75;
    margin: 0;
}

.section-title {
    font-size: 1.5rem;
    font-weight: 900;
    color: #f8fafc;
    letter-spacing: -0.03em;
    margin-bottom: 0.25rem;
}
.section-sub {
    font-size: 0.98rem;
    color: rgba(255,255,255,0.82);
    margin-bottom: 1rem;
}

.panel-card, .result-card, .metric-tile, .sub-block, .notice-box, .preview-card {
    background: linear-gradient(180deg, var(--panel), var(--panel-strong));
    border: 1px solid rgba(255,255,255,0.34);
    color: var(--text);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-soft);
    backdrop-filter: blur(16px);
    animation: fadeUp 0.45s ease;
}

.panel-card {
    padding: 1rem;
}

.input-panel {
    padding: 1rem;
    position: sticky;
    top: 1rem;
}

.preview-card {
    padding: 0.85rem;
    min-height: 100%;
}
.preview-title {
    font-size: 0.96rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.7rem;
}
.preview-empty {
    min-height: 120px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: #f8fafc;
    background: transparent;
    border-radius: 0;
    border: none;
    padding: 0.75rem 0.25rem;
}

.upload-shell {
    border: 1px solid rgba(15,23,42,0.08);
    background: rgba(255,255,255,0.44);
    border-radius: 22px;
    padding: 0.55rem;
}

.result-card {
    padding: 1.1rem;
    margin-bottom: 1rem;
}
.side-stack > div {
    margin-bottom: 1rem;
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 0.95rem;
    flex-wrap: wrap;
}
.result-name {
    font-size: clamp(1.6rem, 2.5vw, 2.2rem);
    font-weight: 900;
    color: var(--text);
    line-height: 1.05;
    letter-spacing: -0.03em;
    margin-bottom: 0.35rem;
}
.result-chip {
    display: inline-flex;
    align-items: center;
    padding: 0.4rem 0.82rem;
    border-radius: 999px;
    background: linear-gradient(135deg, var(--gold), var(--gold-2));
    color: #1a1408;
    font-size: 0.83rem;
    font-weight: 900;
    box-shadow: 0 10px 22px rgba(212,175,55,0.18);
}
.result-subline {
    color: #ffffff;
    font-size: 0.96rem;
    font-weight: 600;
}

.ring-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
}
.conf-ring {
    --pct: 50;
    width: 132px;
    height: 132px;
    border-radius: 50%;
    background: conic-gradient(#2fb36d calc(var(--pct) * 1%), #e8edf2 0);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: inset 0 0 0 1px rgba(15,23,42,0.06), 0 16px 36px rgba(31,143,82,0.16);
    animation: spinIn 0.9s ease;
}
.conf-ring::before {
    content: "";
    width: 96px;
    height: 96px;
    border-radius: 50%;
    background: rgba(255,255,255,0.96);
    position: absolute;
}
.conf-ring-inner {
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}
.conf-ring-value {
    color: var(--text);
    font-size: 1.65rem;
    font-weight: 900;
    line-height: 1;
}
.conf-ring-label {
    color: var(--muted);
    font-size: 0.76rem;
    font-weight: 700;
    margin-top: 0.18rem;
}

.conf-wrap {
    margin-top: 0.25rem;
}
.conf-top {
    display: flex;
    justify-content: space-between;
    color: var(--muted);
    font-size: 0.92rem;
    font-weight: 800;
    margin-bottom: 0.45rem;
    gap: 0.7rem;
    flex-wrap: wrap;
}
.conf-bar {
    width: 100%;
    height: 12px;
    border-radius: 999px;
    overflow: hidden;
    background: rgba(15,23,42,0.08);
}
.conf-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #2fb36d 0%, #9ed14d 55%, #f0c95c 100%);
}

.soft-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1rem;
}
.soft-chip {
    display: inline-flex;
    align-items: center;
    padding: 0.52rem 0.8rem;
    border-radius: 999px;
    background: rgba(15,23,42,0.05);
    border: 1px solid rgba(15,23,42,0.08);
    color: var(--text);
    font-size: 0.86rem;
    font-weight: 800;
}

.metric-grid-2 {
    display: grid;
    grid-template-columns: repeat(2, minmax(0,1fr));
    gap: 0.8rem;
}
.metric-grid-3 {
    display: grid;
    grid-template-columns: repeat(3, minmax(0,1fr));
    gap: 0.8rem;
}
.metric-tile {
    padding: 1rem;
    min-height: 128px;
}
.metric-kicker {
    color: var(--muted-2);
    font-size: 0.83rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.metric-big {
    color: var(--text);
    font-size: clamp(1.4rem, 2.4vw, 2rem);
    font-weight: 900;
    line-height: 1.1;
    margin-bottom: 0.35rem;
}
.metric-foot {
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 1.5;
}

.explain-box {
    background: linear-gradient(180deg, rgba(8,12,17,0.88), rgba(10,14,20,0.94));
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 22px;
    padding: 1.2rem 1.2rem;
    margin-top: 1rem;
    box-shadow: 0 18px 50px rgba(0,0,0,0.18);
    backdrop-filter: blur(12px);
}
.explain-title {
    color: #ffffff;
    font-size: 1.6rem;
    font-weight: 900;
    margin-bottom: 0.8rem;
    letter-spacing: -0.03em;
}
.explain-subtitle {
    color: #ffffff;
    font-size: 1.06rem;
    font-weight: 900;
    margin-top: 1rem;
    margin-bottom: 0.55rem;
}
.explain-list {
    margin: 0;
    padding-left: 1.2rem;
}
.explain-list li {
    color: rgba(255,255,255,0.96) !important;
    font-size: 0.98rem;
    line-height: 1.72;
    font-weight: 650;
    margin-bottom: 0.38rem;
}
.explain-box p,
.explain-box span,
.explain-box div,
.explain-box strong,
.explain-box b {
    color: #ffffff !important;
}

.sub-block {
    padding: 1rem;
    margin-top: 0.8rem;
}
.sub-heading {
    color: var(--text);
    font-size: 1rem;
    font-weight: 900;
    margin-bottom: 0.6rem;
}
.notice-box {
    border-radius: 18px;
    padding: 0.95rem 1rem;
    color: var(--text);
    font-size: 0.95rem;
    line-height: 1.6;
}
.warning-soft {
    background: linear-gradient(180deg, rgba(255,248,228,0.92), rgba(255,243,202,0.92));
    border: 1px solid rgba(245,158,11,0.20);
}

.risk-high {
    background: linear-gradient(90deg, #b42318, #ef5350);
}
.risk-moderate {
    background: linear-gradient(90deg, #b7791f, #f6ad55);
}
.risk-low {
    background: linear-gradient(90deg, #15803d, #34d399);
}
.risk-banner {
    border-radius: 18px;
    padding: 1rem;
    color: white;
    font-weight: 850;
    margin-top: 0.8rem;
    box-shadow: 0 14px 26px rgba(0,0,0,0.10);
}

.pred-row {
    margin-bottom: 0.62rem;
}
.pred-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.7rem;
    font-size: 0.92rem;
    color: var(--text);
    font-weight: 800;
    margin-bottom: 0.24rem;
}
.pred-bar {
    width: 100%;
    height: 9px;
    border-radius: 999px;
    overflow: hidden;
    background: rgba(15,23,42,0.08);
}
.pred-fill {
    height: 100%;
    border-radius: 999px;
}

.side-label {
    font-size: 0.8rem;
    color: var(--muted-2);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    font-weight: 900;
    margin-bottom: 0.32rem;
}
.side-value {
    font-size: 1.02rem;
    color: var(--text);
    font-weight: 800;
}

div.stButton > button {
    width: 100%;
    min-height: 54px;
    border-radius: 16px;
    border: 1px solid rgba(15,23,42,0.06) !important;
    background: linear-gradient(135deg, var(--gold), var(--gold-2)) !important;
    color: #1a1408 !important;
    -webkit-text-fill-color: #1a1408 !important;
    font-weight: 900 !important;
    font-size: 1rem !important;
    box-shadow: 0 14px 28px rgba(212,175,55,0.18);
    transition: all 0.22s ease !important;
}
div.stButton > button:hover {
    transform: translateY(-1px) scale(1.01);
    box-shadow: 0 18px 34px rgba(212,175,55,0.22);
    filter: brightness(1.02);
}
div.stButton > button *,
div.stButton > button p,
div.stButton > button span,
div.stButton > button div {
    color: #1a1408 !important;
    fill: #1a1408 !important;
    -webkit-text-fill-color: #1a1408 !important;
    font-weight: 900 !important;
}

label, .stTextInput label, .stTextArea label, .stSelectbox label,
.stFileUploader label, div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] p {
    color: #f8fafc !important;
    font-weight: 850 !important;
    opacity: 1 !important;
}

input, textarea {
    border-radius: 16px !important;
}
input {
    background: rgba(255,255,255,0.98) !important;
    color: #1f2937 !important;
}
input::placeholder, textarea::placeholder { color: #7b8493 !important; }

div[data-baseweb="textarea"] textarea {
    background: rgba(255,255,255,0.98) !important;
    color: #202734 !important;
    border-radius: 16px !important;
    min-height: 128px !important;
}

div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.98) !important;
    border: 1px solid rgba(15,23,42,0.08) !important;
    border-radius: 16px !important;
    min-height: 52px !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] input {
    color: #1f2937 !important;
    font-weight: 650 !important;
}
div[role="listbox"] {
    background: #f8fafc !important;
    color: #202734 !important;
    border-radius: 14px !important;
}
div[role="option"] {
    color: #202734 !important;
}
div[role="option"]:hover { background: #eef2f7 !important; }

section[data-testid="stFileUploadDropzone"] {
    background: rgba(255,255,255,0.98) !important;
    border: 1.5px dashed rgba(15,23,42,0.14) !important;
    border-radius: 20px !important;
    min-height: 130px !important;
}
section[data-testid="stFileUploadDropzone"] * {
    color: #1f2937 !important;
    fill: #1f2937 !important;
    stroke: #1f2937 !important;
}

/* Uploaded file row visibility fix (applies only after a file is selected) */
div[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"] {
    background: rgba(15,23,42,0.88) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 12px !important;
    padding: 8px 10px !important;
}
div[data-testid="stFileUploader"] div[data-testid="stFileUploaderFileName"],
div[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"] small,
div[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"] span,
div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] + div small {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    opacity: 1 !important;
}
div[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"] svg,
div[data-testid="stFileUploader"] section[data-testid="stFileUploadDropzone"] + div svg {
    color: #ffffff !important;
    fill: #ffffff !important;
    stroke: #ffffff !important;
    opacity: 1 !important;
}
div[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"] button {
    background: rgba(255,255,255,0.14) !important;
    border: 1px solid rgba(255,255,255,0.26) !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    opacity: 1 !important;
}
div[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"] button * {
    color: #ffffff !important;
    fill: #ffffff !important;
    stroke: #ffffff !important;
    opacity: 1 !important;
}
div[data-testid="stFileUploader"] div[data-testid="stFileUploaderFile"] button:hover {
    background: rgba(255,255,255,0.24) !important;
    border-color: rgba(255,255,255,0.34) !important;
}

div[data-testid="stFileUploader"] button {
    background: #0f172a !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid #0f172a !important;
    font-weight: 700 !important;
}
div[data-testid="stFileUploader"] button * {
    color: #ffffff !important;
    fill: #ffffff !important;
    stroke: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

div[data-testid="stTabPanel"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 1rem 0 0 0 !important;
    margin-top: 0.8rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0.65rem;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    min-height: 50px;
    border-radius: 16px;
    color: rgba(255,255,255,0.9) !important;
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    font-size: 0.98rem;
    font-weight: 850;
    padding: 0 1.15rem;
    backdrop-filter: blur(12px);
}
.stTabs [aria-selected="true"] {
    color: #111111 !important;
    background: linear-gradient(135deg, #d4af37, #f2d06b) !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    box-shadow: 0 12px 28px rgba(212,175,55,0.18) !important;
}
.stTabs [aria-selected="true"] * {
    color: #111111 !important;
    fill: #111111 !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background: transparent !important;
}

/* expander */
div[data-testid="stExpander"] {
    border: 1px solid rgba(15,23,42,0.08) !important;
    border-radius: 16px !important;
    overflow: hidden;
    background: rgba(255,255,255,0.92) !important;
    box-shadow: var(--shadow-soft);
}
div[data-testid="stExpander"] summary {
    background: rgba(248,250,252,0.98) !important;
    color: var(--text) !important;
    font-weight: 850 !important;
}
div[data-testid="stExpanderDetails"] {
    background: rgba(255,255,255,0.98) !important;
}
div[data-testid="stExpander"] *,
div[data-testid="stExpanderDetails"] *,
div[data-testid="stExpander"] p,
div[data-testid="stExpander"] li,
div[data-testid="stExpander"] span {
    color: var(--text) !important;
    text-shadow: none !important;
}

img {
    border-radius: 18px;
}
small, p, span, li {
    text-shadow: none !important;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(18px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes spinIn {
    from { opacity: 0; transform: scale(0.8) rotate(-90deg); }
    to { opacity: 1; transform: scale(1) rotate(0deg); }
}

@media (max-width: 1100px) {
    .hero-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 900px) {
    .block-container {
        padding-top: 0.55rem;
        padding-left: 0.55rem;
        padding-right: 0.55rem;
    }
    .main-shell {
        border-radius: 24px;
        padding: 0.65rem;
    }
    .hero-shell {
        padding: 1rem;
        border-radius: 24px;
    }
    .hero-title {
        font-size: 2.45rem;
    }
    .hero-subtitle {
        font-size: 0.98rem;
    }
    .metric-grid-3, .metric-grid-2 {
        grid-template-columns: 1fr;
    }
    .conf-ring {
        width: 120px;
        height: 120px;
    }
    .conf-ring::before {
        width: 88px;
        height: 88px;
    }
}


/* FINAL VISIBILITY PATCH */
.result-card,
.result-card *,
.result-card p,
.result-card span,
.result-card div,
.result-card li,
.result-card h1,
.result-card h2,
.result-card h3,
.result-card h4,
.result-card h5,
.result-card h6 {
    color: #f8fafc !important;
    text-shadow: none !important;
}

.result-name,
.result-header,
.result-header *,
.sub-heading,
.notice-box,
.notice-box *,
.metric-kicker,
.metric-big,
.metric-foot,
.conf-top,
.soft-chip,
.soft-chip *,
.risk-banner,
.risk-banner *,
label,
small {
    text-shadow: none !important;
}

.result-name { color: #ffffff !important; }
.conf-top span { color: #e5e7eb !important; }
.soft-chip { color: #f9fafb !important; background: rgba(255,255,255,0.10) !important; }
.soft-chip i { color: #f9fafb !important; }
.sub-heading { color: #ffffff !important; }
.sub-heading.other-match-title { color: #0f172a !important; }
.notice-box { color: #f8fafc !important; }
.notice-box * { color: #f8fafc !important; }

/* white labels above image panels */
.stMarkdown p strong,
.stMarkdown strong,
.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3,
.stMarkdown h4 { text-shadow: none !important; }

/* prediction rows on dark background */
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] li {
    text-shadow: none !important;
}

/* specific right-side section titles */
.section-sub,
.section-title { text-shadow: none !important; }

/* make image model prediction labels visible */
.prediction-label,
.prediction-label * { color: #ffffff !important; }

/* extra protection for any headings rendered outside cards */
body .stApp h1,
body .stApp h2,
body .stApp h3,
body .stApp h4,
body .stApp h5,
body .stApp h6 {
    color: #f8fafc;
}


/* FINAL SERIOUS VISIBILITY FIX */
.light-note,
.light-note *,
.warning-soft,
.warning-soft *,
.side-label,
.side-value {
    color: #0f172a !important;
    text-shadow: none !important;
}
.light-note b,
.light-note strong,
.warning-soft b,
.warning-soft strong {
    color: #0f172a !important;
}
.side-stack .result-card .sub-heading,
.side-stack .result-card .metric-kicker,
.side-stack .result-card .metric-big,
.side-stack .result-card .metric-foot,
.side-stack .result-card .pred-head,
.side-stack .result-card .pred-head * {
    color: #ffffff !important;
}

/* HIDE EMPTY MARKDOWN-CREATED WRAPPER DIVS */
.panel-card:empty,
.result-card:empty,
.preview-card:empty,
.notice-box:empty,
.sub-block:empty,
.metric-tile:empty,
.ring-wrap:empty,
.metric-grid-2:empty,
.metric-grid-3:empty,
.soft-chip-row:empty {
    display: none !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    box-shadow: none !important;
    min-height: 0 !important;
    height: 0 !important;
    background: transparent !important;
}

/* Streamlit wraps each st.markdown call; opening wrapper divs can become visible empty cards */
div[data-testid="stMarkdownContainer"] > div.panel-card:empty,
div[data-testid="stMarkdownContainer"] > div.result-card:empty,
div[data-testid="stMarkdownContainer"] > div.preview-card:empty,
div[data-testid="stMarkdownContainer"] > div.notice-box:empty,
div[data-testid="stMarkdownContainer"] > div.sub-block:empty,
div[data-testid="stMarkdownContainer"] > div.metric-tile:empty {
    display: none !important;
}

/* Remove accidental empty placeholder blocks without affecting populated cards */
div[data-testid="stMarkdownContainer"] > div:empty {
    display: none !important;
}

/* Extra guard against rounded white bars generated by standalone opening tags */
div.result-card.diagnosis-surface:empty,
div.preview-card:empty,
div.panel-card.input-panel:empty {
    display: none !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    with open("master_diseases_embedded.json", "r", encoding="utf-8") as f:
        return json.load(f)


def safe_str(value):
    return str(value).strip() if value is not None else ""


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def group_crop_options(records):
    grouped = {}
    for item in records:
        category = safe_str(item.get("category")) or "Other"
        crop = safe_str(item.get("crop"))
        if crop:
            grouped.setdefault(category, set()).add(crop)

    preferred_order = [
        "Cereals",
        "Pulses",
        "Oilseeds",
        "Vegetables",
        "Fruits",
        "Commercial Crops",
        "Spices",
        "Plantation Crops",
        "Flowers",
        "Other",
    ]

    ordered_categories = [c for c in preferred_order if c in grouped]
    ordered_categories += sorted([c for c in grouped if c not in preferred_order])

    options = ["All Crops"]
    for category in ordered_categories:
        options.append(f"──── {category} ────")
        for crop in sorted(grouped[category]):
            options.append(f"   {crop}")
    return options


def extract_crop_name(grouped_option: str) -> str:
    if not grouped_option or grouped_option == "All Crops":
        return "All"
    if grouped_option.strip().startswith("────"):
        return ""
    return grouped_option.strip()


def get_type_options(records, selected_crop_name):
    filtered = records
    if selected_crop_name != "All":
        filtered = [
            r for r in records if safe_str(r.get("crop")).lower() == selected_crop_name.lower()
        ]
    types = sorted({safe_str(r.get("diagnosis_type")) for r in filtered if safe_str(r.get("diagnosis_type"))})
    return ["All"] + types


def apply_filters(records, crop, diagnosis_type):
    filtered = records
    if crop != "All":
        filtered = [r for r in filtered if safe_str(r.get("crop")).lower() == crop.lower()]
    if diagnosis_type != "All":
        filtered = [
            r for r in filtered if safe_str(r.get("diagnosis_type")).lower() == diagnosis_type.lower()
        ]
    return filtered


def get_confidence_label(score):
    if score >= 0.60:
        return "High Confidence", "#2fb36d"
    elif score >= 0.40:
        return "Moderate Confidence", "#f59e0b"
    return "Low Confidence", "#ef4444"


def estimate_match_percent(score):
    return int(max(5, min(98, round(score * 100))))


def get_risk_ui(risk_level):
    level = str(risk_level).strip().lower()
    if level == "high":
        return "risk-high"
    elif level == "moderate":
        return "risk-moderate"
    return "risk-low"


def build_fusion_query(user_symptoms: str, image_result: dict, image_weight: float) -> str:
    symptom_text = user_symptoms.strip()
    best = image_result.get("best_prediction")
    if best and image_weight > 0.30:
        image_label = best.get("normalized_label", "")
        if image_label.lower() not in ("healthy", "background", ""):
            return f"{symptom_text} {image_label}".strip()
    return symptom_text


def rerank_results(results: list, image_result: dict, image_weight: float) -> list:
    best_pred = image_result.get("best_prediction") if image_result else None
    pred_label = best_pred.get("normalized_label", "").lower() if best_pred else ""

    reranked = []
    for r in results:
        retrieval_score = float(r.get("score", 0.0))
        name_lower = safe_str(r.get("name")).lower()
        sci_lower = safe_str(r.get("scientific_name")).lower()

        image_boost = 0.0
        if pred_label and pred_label not in ("healthy", "background"):
            pred_tokens = set(pred_label.split())
            name_tokens = set(name_lower.split())
            overlap = pred_tokens & name_tokens
            if overlap:
                image_boost = 1.0
            elif pred_label in name_lower or name_lower in pred_label:
                image_boost = 0.8
            elif pred_label in sci_lower:
                image_boost = 0.6

        combined_score = (1 - image_weight) * retrieval_score + image_weight * image_boost
        item = dict(r)
        item["combined_score"] = combined_score
        item["image_boost"] = image_boost
        reranked.append(item)

    reranked.sort(key=lambda x: x["combined_score"], reverse=True)
    return reranked


def determine_evidence_source(image_result: dict, image_weight: float, has_symptoms: bool) -> str:
    trust = image_result.get("trust_level", "NONE") if image_result else "NONE"

    if trust == "NONE" and not has_symptoms:
        return "Insufficient evidence"
    elif trust == "NONE":
        return "Based on symptom text only"
    elif trust == "STRONG" and has_symptoms:
        return "Image + symptom evidence"
    elif trust == "STRONG" and not has_symptoms:
        return "Based on image prediction"
    elif trust == "MODERATE" and has_symptoms:
        return "Primarily symptom-based, image as support"
    elif trust == "MODERATE":
        return "Image prediction with moderate confidence"
    elif trust == "WEAK" and has_symptoms:
        return "Based on symptom text; image confidence is low"
    return "Image prediction unreliable, limited evidence"


def calculate_roi(area_acres, expected_yield_per_acre, market_price_per_unit, disease_loss_percent, recovery_percent, treatment_cost):
    total_expected_yield = area_acres * expected_yield_per_acre
    expected_revenue_without_disease = total_expected_yield * market_price_per_unit
    estimated_loss_value = expected_revenue_without_disease * (disease_loss_percent / 100.0)
    recoverable_value = estimated_loss_value * (recovery_percent / 100.0)
    net_benefit = recoverable_value - treatment_cost
    roi_percent = (net_benefit / treatment_cost) * 100 if treatment_cost > 0 else 0.0

    return {
        "expected_revenue_without_disease": round(expected_revenue_without_disease, 2),
        "estimated_loss_value": round(estimated_loss_value, 2),
        "recoverable_value": round(recoverable_value, 2),
        "net_benefit": round(net_benefit, 2),
        "roi_percent": round(roi_percent, 2),
    }


def get_profit_loss_label(net_benefit):
    if net_benefit > 0:
        return "Profit", "✅"
    elif net_benefit < 0:
        return "Loss", "⚠️"
    return "Break-even", "➖"


def get_default_risk_profile(location: str):
    weather_data = get_weather_data(location)
    if weather_data.get("success"):
        humidity = float(weather_data.get("humidity", 0) or 0)
        rainfall = float(weather_data.get("rainfall_mm", 0) or 0)
        temperature = float(weather_data.get("temperature_c", 0) or 0)

        if humidity >= 85 or rainfall >= 10:
            return {
                "risk_level": "High",
                "risk_score": 80,
                "reason": "High humidity or rainfall suggests elevated disease pressure.",
                "weather_data": weather_data,
            }
        elif humidity >= 65 or rainfall >= 3 or 20 <= temperature <= 32:
            return {
                "risk_level": "Moderate",
                "risk_score": 55,
                "reason": "Moderate environmental conditions may support disease development.",
                "weather_data": weather_data,
            }

    return {
        "risk_level": "Low",
        "risk_score": 25,
        "reason": "Current conditions suggest relatively low disease pressure.",
        "weather_data": weather_data,
    }


def severity_from_risk_and_confidence(risk_level, match_percent):
    rl = str(risk_level).lower()
    if rl == "high" and match_percent >= 75:
        return "High"
    if rl in ("high", "moderate") and match_percent >= 55:
        return "Moderate"
    return "Mild"


def plain_weather_text(weather_data):
    if not weather_data.get("success"):
        return "Weather data unavailable"
    return (
        f"Humidity {weather_data.get('humidity', 'N/A')}%, "
        f"Temperature {weather_data.get('temperature_c', 'N/A')}°C, "
        f"Rain {weather_data.get('rainfall_mm', 'N/A')} mm"
    )


def make_conf_ring(percent: int) -> str:
    pct = max(0, min(100, int(percent)))
    return f"""
    <div class="ring-wrap">
        <div class="conf-ring" style="--pct:{pct};">
            <div class="conf-ring-inner">
                <div class="conf-ring-value">{pct}%</div>
                <div class="conf-ring-label">Confidence</div>
            </div>
        </div>
    </div>
    """


def prediction_bar_html(label: str, pct: int, crop_relevant: bool) -> str:
    pct = max(0, min(100, int(pct)))
    if pct >= 90:
        color = "linear-gradient(90deg,#2fb36d,#59d39a)"
    elif pct >= 75:
        color = "linear-gradient(90deg,#f59e0b,#f7c35a)"
    else:
        color = "linear-gradient(90deg,#ef4444,#f87171)"
    icon = "✅" if crop_relevant else "⚠️"
    return f"""
    <div class="pred-row">
        <div class="pred-head" style="color:#0f172a !important;"><span style="color:#0f172a !important;">{icon} {label}</span><span style="color:#0f172a !important;">{pct}%</span></div>
        <div class="pred-bar"><div class="pred-fill" style="width:{pct}%; background:{color};"></div></div>
    </div>
    """


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
data = load_data()
crop_options = group_crop_options(data)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "diag_result" not in st.session_state:
    st.session_state.diag_result = None
if "roi_only" not in st.session_state:
    st.session_state.roi_only = None

location_default = "Detroit"

st.markdown('<div class="main-shell">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="hero-shell">
    <div class="hero-grid">
        <div class="hero-left">
            <div class="hero-title">Smart Crop Health Advisor</div>
            <div class="hero-subtitle">
                Diagnose plant diseases with a cleaner Apple-style dashboard, combine symptom retrieval with image signals,
                review local weather risk, and estimate treatment impact in one premium workflow.
            </div>
        </div>
        <div class="hero-right">
            <p class="mini-note"><b>Best results:</b> upload one clear leaf image and add 1–2 symptom phrases such as powdery growth, yellow halo, leaf spot, curling, wilting, chewing damage, or blight.</p>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────
tab_diag, tab_weather, tab_profit = st.tabs(["Diagnosis", "Weather Risk", "Profit/Loss"])

with tab_diag:
    st.markdown('<div class="section-title">Diagnosis Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A real SaaS-style layout with a control panel on the left and insights on the right.</div>', unsafe_allow_html=True)

    left_input, right_input = st.columns([0.9, 1.1], gap="large")

    with left_input:
        st.markdown('<div class="panel-card input-panel">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload Plant Image",
            type=["jpg", "jpeg", "png"],
            help="Clear leaf images improve results.",
        )
        user_input = st.text_area(
            "Enter Symptoms (optional)",
            placeholder="Example: white powder on leaves, yellow halo, curling, leaf spots, wilting, chewing damage",
            height=128,
        )

        selected_crop_option = st.selectbox("Select Crop", crop_options)
        selected_crop = extract_crop_name(selected_crop_option)
        if selected_crop_option.strip().startswith("────"):
            st.caption("Choose a crop item below the category heading.")

        diagnosis_type_options = get_type_options(data, selected_crop if selected_crop else "All")
        selected_diagnosis_type = st.selectbox(
            "Select Disease / Pest Type",
            diagnosis_type_options,
        )
        run_check = st.button("Check Diagnosis", key="analyze_plant")
        st.markdown('</div>', unsafe_allow_html=True)

    with right_input:
        st.markdown('<div class="preview-card">', unsafe_allow_html=True)
        st.markdown('<div class="preview-title">Leaf Preview</div>', unsafe_allow_html=True)
        if uploaded_file is not None:
            uploaded_file.seek(0)
            st.image(uploaded_file, use_container_width=True)
        else:
            st.markdown(
                '<div class="preview-empty">No image uploaded yet.<br>Your preview will appear here once you upload a plant photo.</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

with tab_weather:
    st.markdown('<div class="section-title">Weather Risk</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Estimate environmental pressure from current local conditions.</div>', unsafe_allow_html=True)

    wcol1, wcol2 = st.columns([0.95, 1.05], gap="large")
    with wcol1:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        weather_location = st.text_input("Location", value=location_default, key="weather_location_only")
        weather_check = st.button("Check Weather Risk", key="check_weather_risk")
        st.markdown('</div>', unsafe_allow_html=True)
    with wcol2:
        st.markdown(
            '<div class="notice-box">This module uses humidity, rainfall, and temperature to estimate disease favorability. It supports decisions, but it does not replace field scouting.</div>',
            unsafe_allow_html=True,
        )

with tab_profit:
    st.markdown('<div class="section-title">Profit / Loss Estimator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Estimate treatment value against likely disease-related loss.</div>', unsafe_allow_html=True)

    pcol1, pcol2 = st.columns([1, 1], gap="large")
    with pcol1:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        location = st.text_input("Location", value=location_default, key="profit_location")
        area_acres = st.number_input("Field Area (acres)", min_value=0.0, value=1.0, step=0.5)
        expected_yield_per_acre = st.number_input("Yield per Acre", min_value=0.0, value=25.0, step=1.0)
        st.markdown('</div>', unsafe_allow_html=True)
    with pcol2:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        market_price_per_unit = st.number_input("Market Price per Unit ($)", min_value=0.0, value=20.0, step=1.0)
        treatment_cost = st.number_input("Estimated Treatment Cost ($)", min_value=0.0, value=50.0, step=5.0)
        auto_calc_profit_loss = st.button("Calculate Profit / Loss", key="auto_profit_loss")
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# WEATHER ONLY TAB ACTION
# ─────────────────────────────────────────────
if weather_check:
    weather_data_only = get_weather_data(weather_location)
    weather_profile = get_default_risk_profile(weather_location)
    risk_level = weather_profile["risk_level"]
    risk_class = get_risk_ui(risk_level)

    with tab_weather:
        st.markdown('<div class="metric-grid-3">', unsafe_allow_html=True)
        metrics = [
            ("Humidity", f"{weather_data_only.get('humidity', 'N/A')}%", "Relative humidity"),
            ("Temperature", f"{weather_data_only.get('temperature_c', 'N/A')}°C", "Ambient temperature"),
            ("Rainfall", f"{weather_data_only.get('rainfall_mm', 'N/A')} mm", "Recent precipitation"),
        ]
        cols = st.columns(3)
        for col, (title, value, foot) in zip(cols, metrics):
            with col:
                st.markdown(
                    f'<div class="metric-tile"><div class="metric-kicker">{title}</div><div class="metric-big">{value}</div><div class="metric-foot">{foot}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="risk-banner {risk_class}">Risk Level: {risk_level.upper()}<br><span style="font-size:0.95rem; font-weight:700;">{_escape(weather_profile["reason"])}</span></div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# PROFIT / LOSS ONLY MODE
# ─────────────────────────────────────────────
if auto_calc_profit_loss:
    default_risk = get_default_risk_profile(location)
    risk_level = default_risk["risk_level"]
    risk_score = default_risk["risk_score"]
    risk_reason = default_risk["reason"]
    weather_data = default_risk["weather_data"]

    if str(risk_level).lower() == "high":
        disease_loss_percent = 30
        recovery_percent = 60
    elif str(risk_level).lower() == "moderate":
        disease_loss_percent = 18
        recovery_percent = 45
    else:
        disease_loss_percent = 8
        recovery_percent = 25

    roi_result = calculate_roi(
        area_acres=area_acres,
        expected_yield_per_acre=expected_yield_per_acre,
        market_price_per_unit=market_price_per_unit,
        disease_loss_percent=disease_loss_percent,
        recovery_percent=recovery_percent,
        treatment_cost=treatment_cost,
    )
    profit_loss_label, profit_loss_icon = get_profit_loss_label(roi_result["net_benefit"])
    st.session_state.roi_only = {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_reason": risk_reason,
        "weather_data": weather_data,
        "roi_result": roi_result,
        "profit_loss_label": profit_loss_label,
        "profit_loss_icon": profit_loss_icon,
        "location": location,
    }

if st.session_state.roi_only:
    info = st.session_state.roi_only
    risk_class = get_risk_ui(info["risk_level"])
    with tab_profit:
        cols = st.columns(3)
        metrics = [
            ("Expected Revenue", f"${info['roi_result']['expected_revenue_without_disease']}", "Without disease loss"),
            ("Net Benefit", f"{info['profit_loss_icon']} ${info['roi_result']['net_benefit']}", info['profit_loss_label']),
            ("ROI", f"{info['roi_result']['roi_percent']}%", "Treatment return estimate"),
        ]
        for col, (title, value, foot) in zip(cols, metrics):
            with col:
                st.markdown(
                    f'<div class="metric-tile"><div class="metric-kicker">{title}</div><div class="metric-big">{value}</div><div class="metric-foot">{foot}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown(
            f'<div class="risk-banner {risk_class}">Risk Level: {info["risk_level"].upper()}<br><span style="font-size:0.95rem; font-weight:700;">{_escape(info["risk_reason"])}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sub-block"><div class="sub-heading">Calculation Details</div>'
            f'<p><b>Location:</b> {_escape(info["location"])}<br>'
            f'<b>Weather:</b> {_escape(plain_weather_text(info["weather_data"]))}<br>'
            f'<b>Estimated Disease Loss Value:</b> ${info["roi_result"]["estimated_loss_value"]}<br>'
            f'<b>Recoverable Value After Treatment:</b> ${info["roi_result"]["recoverable_value"]}</p></div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# MAIN DIAGNOSIS LOGIC
# ─────────────────────────────────────────────
if run_check:
    if selected_crop_option.strip().startswith("────"):
        st.warning("Please select a crop item, not the category heading.")
        st.stop()

    if not user_input.strip() and uploaded_file is None:
        st.warning("Please enter symptoms or upload an image.")
        st.stop()

    filtered_data = apply_filters(data, selected_crop if selected_crop else "All", selected_diagnosis_type)

    if not filtered_data:
        st.error("No records match the selected crop or type. Try broadening your selection.")
        st.stop()

    crop_for_search = "All" if selected_crop == "All" else selected_crop
    has_symptoms = bool(user_input.strip())

    image_result = {
        "predictions": [],
        "best_prediction": None,
        "trust_image": False,
        "trust_level": "NONE",
        "warning": "",
    }

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            with st.spinner("Analyzing uploaded image..."):
                image_result = predict_disease_from_image(
                    uploaded_file,
                    selected_crop=crop_for_search,
                    dataset_records=filtered_data,
                )
        except Exception as e:
            gradcam_image = None
            print("Grad-CAM DEBUG:", e)

    gradcam_image = None

    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            pil_img = Image.open(uploaded_file).convert("RGB")
            model, processor = get_loaded_model()
            if model is not None and processor is not None:
                gradcam_image = generate_gradcam(model, processor, pil_img)
            else:
                st.warning("Grad-CAM skipped: model or processor not loaded.")
        except Exception as e:
            st.warning(f"Grad-CAM failed: {e}")

    image_weight = compute_image_weight(
        trust_level=image_result.get("trust_level", "NONE"),
        has_symptoms=has_symptoms,
    )

    combined_query = build_fusion_query(user_input, image_result, image_weight)

    if not combined_query.strip():
        if image_result.get("best_prediction"):
            bp = image_result["best_prediction"]
            combined_query = bp.get("normalized_label", "")
        if not combined_query.strip():
            st.warning("Could not build a search query. Please enter symptoms or upload a clearer image.")
            st.stop()

    with st.spinner("Searching knowledge base..."):
        results = find_best_matches(combined_query, filtered_data, crop_for_search, top_k=5)

    if not results:
        st.error("No matching disease or pest found. Try different symptoms or broader selection.")
        st.stop()

    results = rerank_results(results, image_result, image_weight)
    top_result = results[0]
    other_results = results[1:3]
    evidence_source = determine_evidence_source(image_result, image_weight, has_symptoms)

    weather_data = get_weather_data(location)
    risk_result = calculate_risk(
        diagnosis_name=top_result.get("name", ""),
        diagnosis_type=top_result.get("diagnosis_type", ""),
        weather=weather_data,
    )

    llm_output = generate_llm_explanation(
        disease=top_result.get("name"),
        symptoms=top_result.get("symptoms"),
        weather=weather_data,
    )

    score = float(top_result.get("combined_score", top_result.get("score", 0.0)))

    state = {
        "results": results,
        "llm_output": llm_output,
        "confidence": score,
        "weather": weather_data,
    }

    final = graph.invoke(state)
    if final.get("warning"):
        st.warning(final["warning"])
    if final.get("weather_risk"):
        st.info(final["weather_risk"])

    results = final["results"]
    llm_output = final["llm_output"]
    best_prediction = image_result.get("best_prediction") or {}
    user_selected_symptoms = [part.strip() for part in user_input.split(",") if part.strip()]
    humidity = weather_data.get("humidity")
    temperature = weather_data.get("temperature_c")
    explanation = generate_dynamic_explanation(
        disease_name=best_prediction.get("normalized_label") or top_result.get("name", ""),
        symptoms=user_selected_symptoms,
        weather={
            "humidity": humidity,
            "temperature": temperature,
        },
        confidence=best_prediction.get("score", score),
    )

    confidence_label, conf_color = get_confidence_label(score)
    match_percent = estimate_match_percent(score)

    top_symptoms = top_result.get("symptoms", [])
    top_management = top_result.get("management", [])
    top_cause = safe_str(top_result.get("cause_description"))
    top_name = _escape(safe_str(top_result.get("name")) or "Unknown")
    top_crop = _escape(safe_str(top_result.get("crop")) or "N/A")
    top_scientific = _escape(safe_str(top_result.get("scientific_name")) or "N/A")
    top_type = _escape(safe_str(top_result.get("diagnosis_type")) or "N/A")

    risk_level = risk_result.get("risk_level", "Low")
    risk_score = risk_result.get("risk_score", 0)
    risk_reason = risk_result.get("reason", "")
    risk_class = get_risk_ui(risk_level)
    severity = severity_from_risk_and_confidence(risk_level, match_percent)

    if str(risk_level).lower() == "high":
        disease_loss_percent = 30
        recovery_percent = 60
    elif str(risk_level).lower() == "moderate":
        disease_loss_percent = 18
        recovery_percent = 45
    else:
        disease_loss_percent = 8
        recovery_percent = 25

    roi_result = calculate_roi(
        area_acres=area_acres,
        expected_yield_per_acre=expected_yield_per_acre,
        market_price_per_unit=market_price_per_unit,
        disease_loss_percent=disease_loss_percent,
        recovery_percent=recovery_percent,
        treatment_cost=treatment_cost,
    )
    profit_loss_label, profit_loss_icon = get_profit_loss_label(roi_result["net_benefit"])

    st.session_state.diag_result = {
        "top_result": top_result,
        "other_results": other_results,
        "evidence_source": evidence_source,
        "weather_data": weather_data,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_reason": risk_reason,
        "risk_class": risk_class,
        "severity": severity,
        "explanation": explanation,
        "gradcam_image": gradcam_image,
        "score": score,
        "match_percent": match_percent,
        "confidence_label": confidence_label,
        "conf_color": conf_color,
        "top_symptoms": top_symptoms,
        "top_management": top_management,
        "top_cause": top_cause,
        "top_name": top_name,
        "top_crop": top_crop,
        "top_scientific": top_scientific,
        "top_type": top_type,
        "image_result": image_result,
        "image_weight": image_weight,
        "roi_result": roi_result,
        "profit_loss_label": profit_loss_label,
        "profit_loss_icon": profit_loss_icon,
        "uploaded_file": uploaded_file,
    }

# ─────────────────────────────────────────────
# RENDER DIAGNOSIS RESULTS
# ─────────────────────────────────────────────
if st.session_state.diag_result:
    r = st.session_state.diag_result

    with tab_diag:
        main_col, side_col = st.columns([1.28, 0.72], gap="large")

        with main_col:
            st.markdown('<div class="result-card diagnosis-surface">', unsafe_allow_html=True)
            title_col, ring_col = st.columns([1.3, 0.7], gap="medium")

            with title_col:
                st.markdown(
                    f'''
                    <div class="result-header" style="margin-bottom:0.4rem;">
                        <div>
                            <div class="result-name">{r['top_name']}</div>
                            <div class="result-subline">Primary diagnosis surfaced from symptom retrieval, image ranking, and weather-aware reasoning.</div>
                        </div>
                    </div>
                    <div><span class="result-chip">{r['top_type']}</span></div>
                    <div class="conf-wrap">
                        <div class="conf-top">
                            <span>{r['confidence_label']}</span>
                            <span>Severity: {r['severity']}</span>
                        </div>
                        <div class="conf-bar"><div class="conf-fill" style="width:{r['match_percent']}%;"></div></div>
                    </div>
                    <div class="soft-chip-row">
                        <span class="soft-chip">Crop: {r['top_crop']}</span>
                        <span class="soft-chip">Scientific: <i>{r['top_scientific']}</i></span>
                        <span class="soft-chip">Evidence: {_escape(r['evidence_source'])}</span>
                        <span class="soft-chip">Image Weight: {int(r['image_weight'] * 100)}%</span>
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )
                if r["top_cause"]:
                    st.markdown(
                        f'<div class="sub-block"><div class="sub-heading">Likely Cause</div><p>{_escape(r["top_cause"])}</p></div>',
                        unsafe_allow_html=True,
                    )

            with ring_col:
                st.markdown(make_conf_ring(r["match_percent"]), unsafe_allow_html=True)
                st.markdown(
                    f'<div class="notice-box light-note" style="margin-top:0.85rem;"><div class="side-label">Decision Summary</div><div class="side-value">{r["confidence_label"]} · {r["severity"]} severity</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="explain-box">', unsafe_allow_html=True)
            st.markdown('<div class="explain-title">Explanation</div>', unsafe_allow_html=True)
            explanation = r.get("explanation") or {}
            if isinstance(explanation, dict):
                why_match = explanation.get("why_match", [])
                difference = explanation.get("difference", [])

                st.markdown('<div class="explain-subtitle">1. Why This Disease Matches</div>', unsafe_allow_html=True)
                st.markdown(
                    "<ul class='explain-list'>" + "".join([f"<li>{item}</li>" for item in why_match]) + "</ul>",
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="explain-subtitle">2. What Makes It Different from Similar Diseases</div>', unsafe_allow_html=True)
                st.markdown(
                    "<ul class='explain-list'>" + "".join([f"<li>{item}</li>" for item in difference]) + "</ul>",
                    unsafe_allow_html=True,
                )
            else:
                st.write(explanation)
            st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("Symptoms of likely diagnosis", expanded=True):
                if r["top_symptoms"]:
                    for item in r["top_symptoms"]:
                        st.write(f"- {item}")
                else:
                    st.write("No symptom data available.")

            with st.expander("Treatment advice", expanded=True):
                if r["top_management"]:
                    for item in r["top_management"]:
                        st.write(f"- {item}")
                else:
                    st.write("No management data available.")

            if r["other_results"]:
                st.markdown('<div class="section-title" style="margin-top:1.2rem; color:#f8fafc;">Other Possible Matches</div>', unsafe_allow_html=True)
                for idx, result in enumerate(r["other_results"], start=2):
                    symptoms = result.get("symptoms", [])
                    management = result.get("management", [])
                    cause_description = _escape(safe_str(result.get("cause_description")))
                    result_score = float(result.get("combined_score", result.get("score", 0.0)))
                    result_percent = estimate_match_percent(result_score)
                    img_boost = result.get("image_boost", 0.0)
                    boost_tag = ""
                    if img_boost > 0.5:
                        boost_tag = "✅ Image match"
                    elif img_boost > 0:
                        boost_tag = "~ Partial image match"

                    r_name = _escape(safe_str(result.get("name")) or "Unknown")
                    r_crop = _escape(safe_str(result.get("crop")) or "N/A")
                    r_sci = _escape(safe_str(result.get("scientific_name")) or "N/A")
                    r_type = _escape(safe_str(result.get("diagnosis_type")) or "N/A")

                    st.markdown(
                        f'<div class="sub-block"><div class="sub-heading other-match-title">{idx}. {r_name}</div>'
                        f'<p><b>Crop:</b> {r_crop}<br><b>Scientific Name:</b> <i>{r_sci}</i><br><b>Type:</b> {r_type}<br><b>Score:</b> {result_percent}%<br><b>Image Alignment:</b> {boost_tag if boost_tag else "Low"}</p>'
                        + (f'<p><b>Cause:</b> {cause_description}</p>' if cause_description else '') + '</div>',
                        unsafe_allow_html=True,
                    )
                    with st.expander(f"Symptoms — {r_name}"):
                        if symptoms:
                            for item in symptoms:
                                st.write(f"- {item}")
                        else:
                            st.write("No symptom data available.")
                    with st.expander(f"Management — {r_name}"):
                        if management:
                            for item in management:
                                st.write(f"- {item}")
                        else:
                            st.write("No management data available.")

        with side_col:
            st.markdown('<div class="side-stack">', unsafe_allow_html=True)

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="sub-heading" style="color:#0f172a !important;">Uploaded Image</div>', unsafe_allow_html=True)
            if r["uploaded_file"] is not None:
                r["uploaded_file"].seek(0)
                st.image(r["uploaded_file"], use_container_width=True)
            else:
                st.markdown(
                    '<div class="notice-box light-note">No image was uploaded. Results are based on symptom text and retrieval.</div>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

            if r.get("gradcam_image") is not None:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<div class="sub-heading" style="color:#0f172a !important;">🔍 AI Focus (Grad-CAM)</div>', unsafe_allow_html=True)
                st.image(r["gradcam_image"], use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="sub-heading" style="color:#0f172a !important;">Weather & Risk</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-grid-2">'
                f'<div class="metric-tile"><div class="metric-kicker">Location</div><div class="metric-big" style="font-size:1.2rem;">{_escape(str(r["weather_data"].get("location", "N/A")))}</div><div class="metric-foot">Current selected location</div></div>'
                f'<div class="metric-tile"><div class="metric-kicker">Condition</div><div class="metric-big" style="font-size:1.2rem;">{_escape(str(r["weather_data"].get("weather_desc", "N/A")))}</div><div class="metric-foot">Observed weather</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="notice-box light-note" style="margin-top:0.8rem;"><b>Weather:</b> {_escape(plain_weather_text(r["weather_data"]))}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="risk-banner {r["risk_class"]}">Risk Level: {r["risk_level"].upper()}<br><span style="font-size:0.95rem; font-weight:700;">{_escape(str(r["risk_reason"]))}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

            if r["image_result"].get("predictions"):
                filtered_preds = [p for p in r["image_result"]["predictions"] if p["crop_relevant"]]
                if not filtered_preds:
                    st.markdown(
                        '<div class="notice-box warning-soft light-note">No crop-specific match found. Showing the closest visual matches.</div>',
                        unsafe_allow_html=True,
                    )
                    filtered_preds = r["image_result"]["predictions"]
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<div class="sub-heading" style="color:#0f172a !important;">Image Model Predictions</div>', unsafe_allow_html=True)
                for p in filtered_preds[:5]:
                    pct = int(round(p["score"] * 100))
                    label_display = _escape(p["normalized_label"])
                    st.markdown(prediction_bar_html(label_display, pct, p["crop_relevant"]), unsafe_allow_html=True)
                warning = r["image_result"].get("warning", "")
                if warning:
                    st.markdown(
                        f'<div class="notice-box warning-soft light-note" style="margin-top:0.8rem;">⚠ {_escape(warning)}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="sub-heading" style="color:#0f172a !important;">Treatment ROI Snapshot</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-grid-2">'
                f'<div class="metric-tile"><div class="metric-kicker">Net Benefit</div><div class="metric-big">{r["profit_loss_icon"]} ${r["roi_result"]["net_benefit"]}</div><div class="metric-foot">{r["profit_loss_label"]}</div></div>'
                f'<div class="metric-tile"><div class="metric-kicker">ROI</div><div class="metric-big">{r["roi_result"]["roi_percent"]}%</div><div class="metric-foot">Treatment outcome estimate</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    with tab_weather:
        cols = st.columns(3)
        metrics = [
            ("Humidity", f"{r['weather_data'].get('humidity', 'N/A')}%", "Disease favorability"),
            ("Temperature", f"{r['weather_data'].get('temperature_c', 'N/A')}°C", "Ambient condition"),
            ("Risk Index", f"{r['risk_score']}", "Out of 100"),
        ]
        for col, (title, value, foot) in zip(cols, metrics):
            with col:
                st.markdown(
                    f'<div class="metric-tile"><div class="metric-kicker">{title}</div><div class="metric-big">{value}</div><div class="metric-foot">{foot}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown(
            f'<div class="risk-banner {r["risk_class"]}">Disease Spread Alert: {r["risk_level"].title()}<br><span style="font-size:0.95rem; font-weight:700;">{_escape(str(r["risk_reason"]))}</span></div>',
            unsafe_allow_html=True,
        )

    with tab_profit:
        cols = st.columns(3)
        metrics = [
            ("Diagnosis Confidence", f"{r['match_percent']}%", r['top_name']),
            ("Net Benefit", f"{r['profit_loss_icon']} ${r['roi_result']['net_benefit']}", r['profit_loss_label']),
            ("ROI", f"{r['roi_result']['roi_percent']}%", "Treatment outcome estimate"),
        ]
        for col, (title, value, foot) in zip(cols, metrics):
            with col:
                st.markdown(
                    f'<div class="metric-tile"><div class="metric-kicker">{title}</div><div class="metric-big">{value}</div><div class="metric-foot">{foot}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown(
            f'<div class="sub-block"><div class="sub-heading">ROI Breakdown</div>'
            f'<p><b>Expected Revenue:</b> ${r["roi_result"]["expected_revenue_without_disease"]}<br>'
            f'<b>Estimated Disease Loss Value:</b> ${r["roi_result"]["estimated_loss_value"]}<br>'
            f'<b>Recoverable Value After Treatment:</b> ${r["roi_result"]["recoverable_value"]}<br>'
            f'<b>Net Benefit:</b> ${r["roi_result"]["net_benefit"]}</p></div>',
            unsafe_allow_html=True,
        )

st.markdown('</div>', unsafe_allow_html=True)
