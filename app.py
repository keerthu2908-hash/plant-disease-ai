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
# CUSTOM CSS — FULL PREMIUM STYLE
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
:root {
    --glass: rgba(20, 23, 20, 0.58);
    --glass-2: rgba(28, 31, 28, 0.72);
    --stroke: rgba(255,255,255,0.12);
    --muted: #dfdfd6;
    --text: #fbfbf7;
    --gold-1: #d5ab46;
    --gold-2: #f0cf6b;
    --green-1: #7ecb67;
    --green-2: #223426;
    --danger-1: #c84a3b;
    --warn-1: #d6992d;
}

html, body, [class*="css"]  {
    font-family: "Inter", "Segoe UI", sans-serif;
}

.stApp {
    background:
        linear-gradient(rgba(20, 14, 7, 0.22), rgba(10, 10, 10, 0.62)),
        url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1800&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: var(--text);
}

.block-container {
    max-width: 1320px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

#MainMenu, footer, header {visibility: hidden;}

.hero-shell {
    position: relative;
    padding: 1.1rem 0 1.3rem 0;
    margin-bottom: 1rem;
}
.hero-grid {
    display: grid;
    grid-template-columns: 1.4fr 0.9fr;
    gap: 1.2rem;
    align-items: start;
}
.hero-left {
    padding: 0.6rem 0.2rem;
}
.hero-title {
    font-size: clamp(2.3rem, 4vw, 4.1rem);
    line-height: 1.02;
    font-weight: 900;
    color: #f6f5ef;
    margin-bottom: 0.9rem;
    letter-spacing: -0.04em;
    text-shadow: 0 4px 18px rgba(0,0,0,0.30);
}
.hero-subtitle {
    font-size: 1.05rem;
    line-height: 1.65;
    color: #ece8dc;
    max-width: 680px;
    margin-bottom: 1rem;
}
.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.8rem;
}
.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.8rem 1.2rem;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.16);
    background: rgba(37, 38, 31, 0.48);
    color: #f8f8f2;
    font-weight: 700;
    box-shadow: 0 10px 24px rgba(0,0,0,0.16);
    backdrop-filter: blur(10px);
}
.hero-right {
    background: linear-gradient(180deg, rgba(38,31,21,0.56), rgba(24,22,18,0.68));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 24px;
    padding: 1rem;
    box-shadow: 0 20px 50px rgba(0,0,0,0.22);
    backdrop-filter: blur(12px);
}
.mini-note {
    color: #f4efdf;
    font-size: 0.95rem;
    line-height: 1.55;
    margin: 0;
}

.glass-card {
    background: linear-gradient(180deg, rgba(20,20,20,0.52), rgba(14,14,14,0.70));
    border: 1px solid rgba(255,255,255,0.11);
    border-radius: 24px;
    box-shadow: 0 20px 46px rgba(0,0,0,0.25);
    backdrop-filter: blur(12px);
}
.section-card {
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.section-title {
    font-size: 1.6rem;
    font-weight: 800;
    color: #fbfbf6;
    margin-bottom: 0.4rem;
    letter-spacing: -0.02em;
}
.section-sub {
    color: #dbd7ca;
    font-size: 0.98rem;
    margin-bottom: 0.9rem;
}

.upload-shell {
    border: 1.4px dashed rgba(255,255,255,0.16);
    background: rgba(255,255,255,0.03);
    border-radius: 22px;
    padding: 0.45rem;
}

.result-card {
    background: linear-gradient(180deg, rgba(30,31,27,0.66), rgba(18,18,18,0.86));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 22px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 16px 30px rgba(0,0,0,0.18);
}
.explain-box {
    background: rgba(18, 18, 18, 0.82);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 22px;
    padding: 1.2rem 1.3rem;
    margin-top: 0.8rem;
    box-shadow: 0 14px 32px rgba(0,0,0,0.22);
    backdrop-filter: blur(10px);
}

.explain-title {
    color: #fffaf2;
    font-size: 1.9rem;
    font-weight: 900;
    margin-bottom: 1rem;
}

.explain-subtitle {
    color: #fff7ea;
    font-size: 1.15rem;
    font-weight: 800;
    margin-top: 1rem;
    margin-bottom: 0.7rem;
}

.explain-list {
    margin: 0;
    padding-left: 1.2rem;
}

.explain-list li {
    color: #fffdf7 !important;
    font-size: 1.02rem;
    line-height: 1.8;
    font-weight: 600;
    margin-bottom: 0.45rem;
    text-shadow: none !important;
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 0.8rem;
}
.result-name {
    font-size: 1.75rem;
    font-weight: 900;
    color: #fcfcf7;
}
.result-chip {
    display: inline-flex;
    align-items: center;
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(205,166,63,0.92), rgba(238,206,102,0.92));
    color: #2a2210;
    font-size: 0.86rem;
    font-weight: 900;
    margin-left: 0.45rem;
}
.conf-wrap {
    margin-top: 0.6rem;
}
.conf-top {
    display: flex;
    justify-content: space-between;
    color: #f2efdf;
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}
.conf-bar {
    width: 100%;
    height: 14px;
    border-radius: 999px;
    overflow: hidden;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.08);
}
.conf-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #7ed26f 0%, #c9ce4e 54%, #edc55b 100%);
}

.soft-chip {
    display: inline-flex;
    align-items: center;
    padding: 0.48rem 0.8rem;
    border-radius: 999px;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    color: #f7f3e8;
    font-size: 0.88rem;
    font-weight: 700;
}

.metric-tile {
    background: linear-gradient(180deg, rgba(36,38,33,0.68), rgba(20,20,19,0.86));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 1rem;
    min-height: 145px;
}
.metric-kicker {
    color: #e2dcc8;
    font-size: 0.92rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
}
.metric-big {
    color: #fffdf7;
    font-size: 2rem;
    font-weight: 900;
    margin-bottom: 0.35rem;
}
.metric-foot {
    color: #d7d4c8;
    font-size: 0.92rem;
}

.notice-box {
    border-radius: 18px;
    padding: 0.95rem 1rem;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    color: #f0ede3;
    font-size: 0.95rem;
    line-height: 1.55;
}

.risk-high {
    background: linear-gradient(90deg, rgba(166,42,42,0.95), rgba(216,94,54,0.95));
}
.risk-moderate {
    background: linear-gradient(90deg, rgba(145,93,13,0.95), rgba(210,165,37,0.95));
}
.risk-low {
    background: linear-gradient(90deg, rgba(28,102,63,0.95), rgba(61,161,101,0.95));
}
.risk-banner {
    border-radius: 18px;
    padding: 1rem 1rem;
    color: white;
    font-weight: 800;
    margin-top: 0.85rem;
    box-shadow: 0 14px 24px rgba(0,0,0,0.18);
}

.sub-block {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 1rem;
    margin-top: 0.8rem;
}
.sub-heading {
    color: #fffaf2;
    font-size: 1.08rem;
    font-weight: 800;
    margin-bottom: 0.6rem;
}

div.stButton > button {
    width: 100%;
    min-height: 56px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.16) !important;
    background: linear-gradient(90deg, var(--gold-1), var(--gold-2)) !important;
    color: #1a1408 !important;
    -webkit-text-fill-color: #1a1408 !important;
    font-weight: 900 !important;
    font-size: 1.05rem !important;
    box-shadow: 0 10px 28px rgba(0,0,0,0.22);
}
div.stButton > button * {
    color: #1a1408 !important;
    fill: #1a1408 !important;
    -webkit-text-fill-color: #1a1408 !important;
}
div.stButton > button p,
div.stButton > button span,
div.stButton > button div {
    color: #1a1408 !important;
    fill: #1a1408 !important;
    -webkit-text-fill-color: #1a1408 !important;
    font-weight: 900 !important;
}
div.stButton > button:hover {
    filter: brightness(1.03);
    transform: translateY(-1px);
}

label, .stTextInput label, .stTextArea label, .stSelectbox label,
.stFileUploader label, div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] p {
    color: #fffef8 !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}

input, textarea {
    border-radius: 16px !important;
}

input {
    background: rgba(255,255,255,0.95) !important;
    color: #1f2937 !important;
}
input::placeholder, textarea::placeholder { color: #6b7280 !important; }

div[data-baseweb="textarea"] textarea {
    background: rgba(255,255,255,0.96) !important;
    color: #202734 !important;
    border-radius: 16px !important;
    min-height: 135px !important;
}

div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.94) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 16px !important;
    min-height: 52px !important;
}
div[data-baseweb="select"] span,
div[data-baseweb="select"] input {
    color: #1f2937 !important;
    font-weight: 600 !important;
}
div[role="listbox"] {
    background: #f8f7f2 !important;
    color: #202734 !important;
    border-radius: 14px !important;
}
div[role="option"] {
    color: #202734 !important;
}
div[role="option"]:hover { background: #ebe8dd !important; }

/* ---------- FILE UPLOADER: full visibility fix ---------- */

section[data-testid="stFileUploadDropzone"] {
    background: rgba(255,255,255,0.96) !important;
    border: 1.5px dashed rgba(0,0,0,0.18) !important;
    border-radius: 22px !important;
    min-height: 130px !important;
}

section[data-testid="stFileUploadDropzone"] * {
    color: #1f2937 !important;
    fill: #1f2937 !important;
    stroke: #1f2937 !important;
}

/* drag-drop title + helper text */
section[data-testid="stFileUploadDropzone"] small,
section[data-testid="stFileUploadDropzone"] span,
section[data-testid="stFileUploadDropzone"] p,
section[data-testid="stFileUploadDropzone"] div {
    color: #1f2937 !important;
}

/* browse files button */
div[data-testid="stFileUploader"] button {
    background: #111111 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #111111 !important;
    font-weight: 700 !important;
}

div[data-testid="stFileUploader"] button * {
    color: #ffffff !important;
    fill: #ffffff !important;
    stroke: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

div[data-testid="stFileUploader"] button:hover {
    background: #222222 !important;
    color: #ffffff !important;
    border-color: #222222 !important;
}

/* uploaded file row */
div[data-testid="stFileUploaderFile"] {
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    border-radius: 14px !important;
    padding: 0.35rem 0.55rem !important;
}

div[data-testid="stFileUploaderFile"] * {
    color: #f8fafc !important;
    fill: #f8fafc !important;
    stroke: #f8fafc !important;
}

/* remove / clear / toolbar icon buttons near uploaded file */
div[data-testid="stFileUploader"] [role="button"],
div[data-testid="stFileUploader"] button[kind="icon"],
div[data-testid="stFileUploader"] .st-emotion-cache-1erivf3,
div[data-testid="stFileUploader"] .st-emotion-cache-1pbsqtx {
    background: #111111 !important;
    color: #ffffff !important;
    fill: #ffffff !important;
    stroke: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
}

/* specifically force SVG icons visible */
div[data-testid="stFileUploader"] svg {
    fill: currentColor !important;
    stroke: currentColor !important;
    color: inherit !important;
}

/* file uploader labels */
div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] label p,
div[data-testid="stFileUploader"] label span {
    color: #fffef8 !important;
    font-weight: 800 !important;
}

/* ensure aria/title icon buttons are fully visible */
div[data-testid="stFileUploader"] [aria-label],
div[data-testid="stFileUploader"] [title] {
    opacity: 1 !important;
}

/* ---------- TOOLTIP FIX (IMPORTANT) ---------- */

div[data-testid="stTooltipContent"],
div[data-testid="stTooltipContent"] * {
    background: rgba(20,20,20,0.9) !important;
    backdrop-filter: blur(6px);
    color: #ffffff !important;
    border-radius: 8px !important;
    padding: 6px 10px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}

/* arrow of tooltip */
div[data-testid="stTooltipContent"]::before {
    background: rgba(20,20,20,0.9) !important;
}

/* ensure tooltip text is never white-on-white */
div[data-testid="stTooltipContent"] span,
div[data-testid="stTooltipContent"] p {
    color: #ffffff !important;
}

/* make the info icon visible */
div[data-testid="stFileUploader"] svg {
    color: #111111 !important;
    fill: #111111 !important;
}


div[data-testid="stTabPanel"] {
    background: linear-gradient(180deg, rgba(20,20,20,0.46), rgba(14,14,14,0.60));
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 24px;
    box-shadow: 0 20px 46px rgba(0,0,0,0.22);
    backdrop-filter: blur(12px);
    padding: 1.2rem 1.2rem 1rem 1.2rem;
    margin-top: 0.8rem;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.8rem;
    border-bottom: none !important;
}
.stTabs [data-baseweb="tab"] {
    min-height: 52px;
    border-radius: 16px 16px 0 0;
    color: #f3eee2 !important;
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    font-size: 1rem;
    font-weight: 800;
    padding: 0 1.15rem;
}
.stTabs [data-baseweb="tab"] * {
    color: #f3eee2 !important;
    fill: #f3eee2 !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255,255,255,0.10) !important;
}
.stTabs [aria-selected="true"] {
    color: #111111 !important;
    background: linear-gradient(135deg, #d4af37, #f2d06b) !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    box-shadow: 0 12px 30px rgba(0,0,0,0.18) !important;
}
.stTabs [aria-selected="true"] * {
    color: #111111 !important;
    fill: #111111 !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    background: transparent !important;
}

div[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 16px !important;
    overflow: hidden;
    background: rgba(255,255,255,0.035) !important;
}
div[data-testid="stExpander"] summary {
    background: rgba(255,255,255,0.04) !important;
    color: #fffaf0 !important;
    font-weight: 800 !important;
}
div[data-testid="stExpanderDetails"] {
    background: rgba(0,0,0,0.14) !important;
}

img {
    border-radius: 20px;
}

small, p, span {
    color: #f2eee1;
}

li {
    color: #fffdf7;
}

@media (max-width: 900px) {
    .hero-grid {
        grid-template-columns: 1fr;
    }
    .hero-right {
        padding: 0.9rem;
    }
    .hero-title {
        font-size: 2.5rem;
    }
    .block-container {
        padding-top: 0.7rem;
        padding-left: 0.7rem;
        padding-right: 0.7rem;
    }
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
                Upload a photo to diagnose plant diseases, review weather-driven risk,
                and estimate field-level impact with a cleaner, decision-friendly dashboard.
            </div>
            <div class="badge-row">
                <div class="hero-pill">🌿 AI Diagnosis</div>
                <div class="hero-pill">☁️ Weather Risk</div>
                <div class="hero-pill">💰 Profit / Loss</div>
            </div>
        </div>
        <div class="hero-right">
            <p class="mini-note"><b>Best results:</b> add a clear leaf image and 1–2 symptom details such as spots, curling, yellowing, wilting, chewing damage, or leaf blight.</p>
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
    st.markdown('<div class="section-title">Diagnosis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload a plant image, add symptoms if you have them, and select the crop once.</div>', unsafe_allow_html=True)

    left_input, right_input = st.columns([1.1, 0.9], gap="large")

    with left_input:
        st.markdown('<div class="upload-shell">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload Plant Image",
            type=["jpg", "jpeg", "png"],
            help="Clear leaf images improve results.",
        )
        user_input = st.text_area(
            "Enter Symptoms (optional)",
            placeholder="Example: white powder on leaves, yellow halo, curling, leaf spots, wilting, chewing damage",
            height=140,
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

    with right_input:
        preview_card = st.container()
        with preview_card:
            if uploaded_file is not None:
                uploaded_file.seek(0)
                st.image(uploaded_file, use_container_width=True)
            else:
                st.markdown(
                    '<div class="notice-box">No image uploaded yet. Your leaf preview will appear here once you upload a photo.</div>',
                    unsafe_allow_html=True,
                )
        

with tab_weather:
    st.markdown('<div class="section-title">Weather Risk</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Use local conditions to estimate environmental disease pressure.</div>', unsafe_allow_html=True)

    wcol1, wcol2 = st.columns([1, 1], gap="large")
    with wcol1:
        weather_location = st.text_input("Location", value="Detroit", key="weather_location_only")
        weather_check = st.button("Check Weather Risk", key="check_weather_risk")
    with wcol2:
        st.markdown(
            '<div class="notice-box">This section estimates disease-favoring conditions from humidity, temperature, and rainfall. It supports decision-making but does not replace field inspection.</div>',
            unsafe_allow_html=True,
        )

with tab_profit:
    st.markdown('<div class="section-title">Profit / Loss Estimator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Estimate treatment value against likely disease-driven loss.</div>', unsafe_allow_html=True)

    pcol1, pcol2 = st.columns([1, 1], gap="large")
    with pcol1:
        location = st.text_input("Location", value="Detroit", key="profit_location")
        area_acres = st.number_input("Field Area (acres)", min_value=0.0, value=1.0, step=0.5)
        expected_yield_per_acre = st.number_input("Yield per Acre", min_value=0.0, value=25.0, step=1.0)
    with pcol2:
        market_price_per_unit = st.number_input("Market Price per Unit ($)", min_value=0.0, value=20.0, step=1.0)
        treatment_cost = st.number_input("Estimated Treatment Cost ($)", min_value=0.0, value=50.0, step=5.0)
        auto_calc_profit_loss = st.button("Calculate Profit / Loss", key="auto_profit_loss")

# ─────────────────────────────────────────────
# WEATHER ONLY TAB ACTION
# ─────────────────────────────────────────────
if weather_check:
    weather_data_only = get_weather_data(weather_location)
    weather_profile = get_default_risk_profile(weather_location)
    risk_level = weather_profile["risk_level"]
    risk_class = get_risk_ui(risk_level)

    with tab_weather:
        st.markdown('<div class="section-title">Weather Overview</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="metric-tile"><div class="metric-kicker">Humidity</div><div class="metric-big">{weather_data_only.get("humidity", "N/A")}%</div><div class="metric-foot">Relative humidity</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-tile"><div class="metric-kicker">Temperature</div><div class="metric-big">{weather_data_only.get("temperature_c", "N/A")}°C</div><div class="metric-foot">Ambient temperature</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="metric-tile"><div class="metric-kicker">Rain</div><div class="metric-big">{weather_data_only.get("rainfall_mm", "N/A")} mm</div><div class="metric-foot">Recent precipitation</div></div>',
                unsafe_allow_html=True,
            )
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
            print("Grad-CAM DEBUG:", e)  # console only

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
    # 🔥 LANGGRAPH STARTS HERE
    
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
        dleft, dright = st.columns([1.08, 0.92], gap="large")

        with dleft:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(
                f'''
                <div class="result-header">
                    <div>
                        <div class="result-name">{r['top_name']} <span class="result-chip">{r['top_type']}</span></div>
                    </div>
                    <div style="font-size:2rem; font-weight:900; color:#fff6df;">{r['match_percent']}%</div>
                </div>
                <div class="conf-wrap">
                    <div class="conf-top">
                        <span>{r['confidence_label']}</span>
                        <span>Severity: {r['severity']}</span>
                    </div>
                    <div class="conf-bar"><div class="conf-fill" style="width:{r['match_percent']}%;"></div></div>
                </div>
                <div style="margin-top:0.9rem;">
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
        
            st.markdown('<div class="explain-box">', unsafe_allow_html=True)
            st.markdown('<div class="explain-title">Explanation</div>', unsafe_allow_html=True)

            explanation = r.get("explanation") or {}
            if isinstance(explanation, dict):
                why_match = explanation.get("why_match", [])
                difference = explanation.get("difference", [])

                st.markdown('<div class="explain-subtitle">1. Why This Disease Matches</div>', unsafe_allow_html=True)
                st.markdown(
                    "<ul class='explain-list'>" +
                    "".join([f"<li>{item}</li>" for item in why_match]) +
                    "</ul>",
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="explain-subtitle">2. What Makes It Different from Similar Diseases</div>', unsafe_allow_html=True)
                st.markdown(
                    "<ul class='explain-list'>" +
                    "".join([f"<li>{item}</li>" for item in difference]) +
                    "</ul>",
                    unsafe_allow_html=True,
                )
            else:
                st.write(explanation)

            st.markdown("</div>", unsafe_allow_html=True)
        
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

        with dright:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            if r["uploaded_file"] is not None:
                r["uploaded_file"].seek(0)
                st.image(r["uploaded_file"], use_container_width=True)
                if r.get("gradcam_image") is not None:
                    st.markdown("### 🔍 AI Focus (Grad-CAM)")
                    st.image(r["gradcam_image"], use_container_width=True)
                else:
                    st.markdown(
                        '<div class="notice-box">Grad-CAM is not available for this prediction yet, but the uploaded image was used for diagnosis.</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div class="notice-box">No image was uploaded for this diagnosis. Results are based on symptom text and retrieval.</div>',
                    unsafe_allow_html=True,
                )

            if r["image_result"].get("predictions"):
                filtered_preds = [
                    p for p in r["image_result"]["predictions"]
                    if p["crop_relevant"]
                ]
                
                if not filtered_preds:
                    st.warning("No crop-specific match found. Showing closest visual matches.")
                    filtered_preds = r["image_result"]["predictions"]
                
                st.markdown('<div class="sub-heading" style="margin-top:0.8rem;">Image Model Predictions</div>', unsafe_allow_html=True)
                for p in filtered_preds[:5]:
                    pct = int(round(p["score"] * 100))
                    color = "#2fb36d" if pct >= 90 else "#f59e0b" if pct >= 75 else "#ef4444"
                    label_display = _escape(p["normalized_label"])
                    crop_icon = "✅" if p["crop_relevant"] else "⚠️"
                    st.markdown(
                        f"<p style='margin-bottom:0.2rem;'><b>{crop_icon} {label_display}</b> — {pct}%</p>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div class='conf-bar' style='height:10px; margin-bottom:0.55rem;'><div class='conf-fill' style='width:{pct}%; background:{color};'></div></div>",
                        unsafe_allow_html=True,
                    )

                warning = r["image_result"].get("warning", "")
                if warning:
                    st.markdown(
                        f'<div class="notice-box" style="margin-top:0.8rem;">⚠ {_escape(warning)}</div>',
                        unsafe_allow_html=True,
                    )
        
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="sub-heading">Weather & Risk</div>', unsafe_allow_html=True)
            st.markdown(
                f'<p><b>Location:</b> {_escape(str(r["weather_data"].get("location", "N/A")))}<br>'
                f'<b>Condition:</b> {_escape(str(r["weather_data"].get("weather_desc", "N/A")))}<br>'
                f'<b>Weather:</b> {_escape(plain_weather_text(r["weather_data"]))}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="risk-banner {r["risk_class"]}">Risk Level: {r["risk_level"].upper()}<br><span style="font-size:0.95rem; font-weight:700;">{_escape(str(r["risk_reason"]))}</span></div>',
                unsafe_allow_html=True,
            )
        
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
    
    if r["other_results"]:
        with tab_diag:
            st.markdown('<div class="section-title">Other Possible Matches</div>', unsafe_allow_html=True)
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
                    f'<div class="sub-block"><div class="sub-heading">{idx}. {r_name}</div>'
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
        