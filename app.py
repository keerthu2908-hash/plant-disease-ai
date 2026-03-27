import json
import streamlit as st
from retriever import find_best_matches
from explainer import generate_explanation
from image_predictor import predict_disease_from_image
from weather_utils import get_weather_data, calculate_risk

st.set_page_config(
    page_title="Smart Crop Health Advisor",
    page_icon="🌿",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
/* ===== GLOBAL ===== */
.stApp {
    background:
        linear-gradient(rgba(7, 18, 10, 0.45), rgba(7, 18, 10, 0.68)),
        url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1600&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #f4f7f1;
}

.block-container {
    max-width: 1450px;
    padding-top: 0.8rem;
    padding-bottom: 2rem;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ===== HEADER ===== */
.hero-wrap {
    width: 100%;
    text-align: center;
    padding: 18px 20px 14px 20px;
    margin-bottom: 22px;
    border-radius: 0;
    background:
        linear-gradient(180deg, rgba(45,70,55,0.70), rgba(25,42,28,0.76)),
        linear-gradient(90deg, rgba(59,89,63,0.80), rgba(113,84,40,0.50));
    border-bottom: 3px solid rgba(113, 189, 52, 0.70);
    box-shadow: 0 10px 30px rgba(0,0,0,0.34);
}

.hero-title {
    font-size: 3rem;
    font-weight: 900;
    color: #ffffff;
    text-align: center;
    line-height: 1.1;
    margin: 0;
    text-shadow: 0 2px 6px rgba(0,0,0,0.45);
}

.hero-subtitle {
    font-size: 1rem;
    color: #eef4e8;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 0;
}

/* ===== INPUT PANEL ===== */
.top-panel {
    background: rgba(7, 24, 11, 0.52);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 10px 24px rgba(0,0,0,0.25);
    margin-bottom: 22px;
    backdrop-filter: blur(6px);
}

.section-title {
    font-size: 1.3rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 10px;
}

/* ===== CARDS ===== */
.dashboard-card {
    background:
        linear-gradient(180deg, rgba(50,66,58,0.90), rgba(27,39,32,0.92));
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    padding: 14px;
    box-shadow: 0 10px 22px rgba(0,0,0,0.30);
    margin-bottom: 16px;
    backdrop-filter: blur(6px);
}

.card-title {
    font-size: 1.08rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.18);
}

.stat-line {
    color: #f3f7ee;
    font-size: 0.98rem;
    margin-bottom: 8px;
    line-height: 1.5;
}

.note-box {
    padding: 12px 14px;
    border-radius: 12px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    color: #e3ebdb;
    font-size: 0.94rem;
    margin-top: 8px;
}

/* ===== RISK BAR ===== */
.risk-banner-high {
    background: linear-gradient(90deg, #8b1914, #d94a20);
    color: white;
    font-weight: 900;
    font-size: 1.05rem;
    border-radius: 0 0 8px 8px;
    padding: 12px 14px;
    margin: 10px -14px -14px -14px;
}

.risk-banner-moderate {
    background: linear-gradient(90deg, #8f5409, #db9d18);
    color: white;
    font-weight: 900;
    font-size: 1.05rem;
    border-radius: 0 0 8px 8px;
    padding: 12px 14px;
    margin: 10px -14px -14px -14px;
}

.risk-banner-low {
    background: linear-gradient(90deg, #17663a, #239b56);
    color: white;
    font-weight: 900;
    font-size: 1.05rem;
    border-radius: 0 0 8px 8px;
    padding: 12px 14px;
    margin: 10px -14px -14px -14px;
}

/* ===== STATUS / ALERTS ===== */
.status-badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 0.88rem;
    color: white;
    margin-top: 8px;
    margin-bottom: 8px;
}

.badge-green { background: linear-gradient(90deg, #1f9d55, #2fb36d); }
.badge-orange { background: linear-gradient(90deg, #d97706, #f59e0b); }
.badge-red { background: linear-gradient(90deg, #b91c1c, #ef4444); }

.alert-strip {
    margin-top: 16px;
    margin-bottom: 14px;
    border-radius: 10px;
    padding: 14px 16px;
    color: white;
    font-weight: 900;
    font-size: 1rem;
    box-shadow: 0 8px 18px rgba(0,0,0,0.25);
}

.alert-green { background: linear-gradient(90deg, #17663a, #239b56); }
.alert-orange { background: linear-gradient(90deg, #9a5a05, #dd8b14); }
.alert-red { background: linear-gradient(90deg, #8b1212, #d13232); }

/* ===== DATA DASHBOARD ===== */
.metric-card {
    background:
        linear-gradient(180deg, rgba(49,65,57,0.93), rgba(23,32,26,0.95));
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    padding: 14px;
    min-height: 170px;
    box-shadow: 0 8px 18px rgba(0,0,0,0.20);
}

.metric-title {
    color: #edf3e8;
    font-size: 0.98rem;
    font-weight: 800;
    margin-bottom: 14px;
}

.metric-value {
    font-size: 2rem;
    font-weight: 900;
    color: #ffffff;
    margin-bottom: 8px;
}

.metric-sub {
    font-size: 0.92rem;
    color: #d8e3cf;
}

/* ===== RESULT HIGHLIGHT ===== */
.result-highlight {
    background:
        linear-gradient(180deg, rgba(76,99,68,0.75), rgba(28,41,31,0.85));
    border: 1px solid rgba(211,230,184,0.15);
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 10px 20px rgba(0,0,0,0.22);
}

.result-title {
    font-size: 1.45rem;
    font-weight: 900;
    color: #ffffff;
    margin-bottom: 12px;
}

.info-chip {
    display: inline-block;
    margin-right: 8px;
    margin-bottom: 8px;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.10);
    color: #f4f7ef;
    font-size: 0.88rem;
    font-weight: 700;
}

/* ===== FORM LABELS ===== */
label,
.stTextInput label,
.stTextArea label,
.stSelectbox label,
.stFileUploader label,
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] p {
    color: #ffffff !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}

.stMarkdown p {
    color: #eef4e8;
}

p, span, small {
    color: #eef4e8;
}

h1, h2, h3, h4 {
    color: #ffffff;
}

/* ===== SELECT BOX ===== */
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    min-height: 48px !important;
}

div[data-baseweb="select"] span {
    color: #ffffff !important;
}

div[role="listbox"] {
    background-color: #1f2d22 !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 10px !important;
}

div[role="option"] {
    background-color: #1f2d22 !important;
    color: #ffffff !important;
}

div[role="option"]:hover {
    background-color: #2d4733 !important;
    color: #ffffff !important;
}

div[aria-selected="true"] {
    background-color: #35593d !important;
    color: #ffffff !important;
}

/* ===== TEXT AREA / INPUT ===== */
div[data-baseweb="textarea"] textarea {
    background: rgba(255,255,255,0.92) !important;
    color: #1f2937 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    min-height: 130px !important;
}

div[data-baseweb="textarea"] textarea::placeholder {
    color: #6b7280 !important;
}

input {
    background: rgba(255,255,255,0.92) !important;
    color: #1f2937 !important;
    border-radius: 12px !important;
}

input::placeholder {
    color: #6b7280 !important;
}

/* ===== FILE UPLOADER ===== */
div[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    border: 1px dashed rgba(255,255,255,0.22) !important;
    color: #ffffff !important;
}

div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] p {
    color: #eef3ea !important;
}

/* ===== BUTTON ===== */
div.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.16);
    background: linear-gradient(90deg, #0f5ad6, #1b7bff) !important;
    color: #ffffff !important;
    font-size: 1rem;
    font-weight: 900;
    box-shadow: 0 10px 18px rgba(0,0,0,0.25);
    opacity: 1 !important;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #0a48ae, #1667d9) !important;
    color: #ffffff !important;
}

/* ===== EXPANDER ===== */
.stExpander {
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 14px !important;
    background: rgba(255,255,255,0.03) !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HELPERS
# -----------------------------
@st.cache_data
def load_data():
    with open("master_diseases_embedded.json", "r", encoding="utf-8") as f:
        return json.load(f)


def safe_str(value):
    return str(value).strip() if value else ""


def apply_filters(records, category, crop, diagnosis_type):
    filtered = records

    if category != "All":
        filtered = [r for r in filtered if safe_str(r.get("category")) == category]

    if crop != "All":
        filtered = [r for r in filtered if safe_str(r.get("crop")).lower() == crop.lower()]

    if diagnosis_type != "All":
        filtered = [
            r for r in filtered
            if safe_str(r.get("diagnosis_type")).lower() == diagnosis_type.lower()
        ]

    return filtered


def get_confidence_label(score):
    if score >= 0.60:
        return "HIGH CONFIDENCE", "badge-green", "alert-green"
    elif score >= 0.40:
        return "MODERATE CONFIDENCE", "badge-orange", "alert-orange"
    return "LOW CONFIDENCE", "badge-red", "alert-red"


def estimate_match_percent(score):
    return int(max(5, min(98, round(score * 100))))


def get_risk_ui(risk_level):
    level = str(risk_level).strip().lower()
    if level == "high":
        return "risk-banner-high", "alert-red", "badge-red"
    elif level == "moderate":
        return "risk-banner-moderate", "alert-orange", "badge-orange"
    return "risk-banner-low", "alert-green", "badge-green"


# -----------------------------
# LOAD DATA
# -----------------------------
data = load_data()

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">Smart Crop Health Advisor</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# INPUT PANEL
# -----------------------------
st.markdown('<div class="top-panel">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Input Details</div>', unsafe_allow_html=True)

fcol1, fcol2, fcol3 = st.columns([1, 1, 1])

category_list = sorted({safe_str(item.get("category")) for item in data if safe_str(item.get("category"))})
category_options = ["All"] + category_list

with fcol1:
    selected_category = st.selectbox("Select category", category_options)

filtered_for_crop = data
if selected_category != "All":
    filtered_for_crop = [item for item in data if safe_str(item.get("category")) == selected_category]

crop_list = sorted({safe_str(item.get("crop")) for item in filtered_for_crop if safe_str(item.get("crop"))})
crop_options = ["All"] + crop_list

with fcol2:
    selected_crop = st.selectbox("Select crop", crop_options)

filtered_for_type = filtered_for_crop
if selected_crop != "All":
    filtered_for_type = [
        item for item in filtered_for_type
        if safe_str(item.get("crop")).lower() == selected_crop.lower()
    ]

diagnosis_type_list = sorted({
    safe_str(item.get("diagnosis_type"))
    for item in filtered_for_type
    if safe_str(item.get("diagnosis_type"))
})
diagnosis_type_options = ["All"] + diagnosis_type_list

with fcol3:
    selected_diagnosis_type = st.selectbox("Select disease / pest type", diagnosis_type_options)

left_input, right_input = st.columns([1.2, 1])

with left_input:
    user_input = st.text_area(
        "Enter symptoms",
        placeholder="Example: brown leaf spots, yellow halo, wilting, blight, chewing damage, curling, lesions, drying",
        height=140
    )

with right_input:
    uploaded_file = st.file_uploader(
        "Upload leaf image (optional)",
        type=["jpg", "jpeg", "png"]
    )
    location = st.text_input(
        "Enter location for weather-based risk assessment",
        value="Detroit",
        placeholder="Example: Detroit, Michigan"
    )
    st.markdown(
        '<div class="note-box">Tip: best results come from a clear leaf image plus symptom text.</div>',
        unsafe_allow_html=True
    )

run_check = st.button("Check Diagnosis")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# MAIN LOGIC
# -----------------------------
if run_check:
    if not user_input.strip() and uploaded_file is None:
        st.warning("Please enter symptoms or upload an image.")
    else:
        filtered_data = apply_filters(
            data,
            selected_category,
            selected_crop,
            selected_diagnosis_type
        )

        crop_for_search = "All" if selected_crop == "All" else selected_crop
        image_prediction_text = ""
        image_predicted_label = "Not available"
        image_predicted_score = 0.0

        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
                with st.spinner("Analyzing uploaded image..."):
                    image_results = predict_disease_from_image(uploaded_file)

                if image_results:
                    top_image_result = image_results[0]
                    image_predicted_label = safe_str(top_image_result.get("label", "Not available"))
                    image_predicted_score = float(top_image_result.get("score", 0.0))
                    image_prediction_text = image_predicted_label
                else:
                    st.warning("Image model returned no prediction.")
            except Exception as e:
                st.error(f"Image prediction failed: {e}")

        combined_query = user_input.strip()
        if image_prediction_text:
            combined_query = f"{combined_query} {image_prediction_text}".strip()

        results = find_best_matches(combined_query, filtered_data, crop_for_search, top_k=5)

        if not results:
            st.error("No matching disease or pest found.")
        else:
            top_result = results[0]
            other_results = results[1:]

            weather_data = get_weather_data(location)
            risk_result = calculate_risk(
                diagnosis_name=top_result.get("name", ""),
                diagnosis_type=top_result.get("diagnosis_type", ""),
                weather=weather_data
            )

            explanation = generate_explanation(
                top_result.get("name"),
                top_result.get("symptoms")
            )

            score = float(top_result.get("score", 0.0))
            confidence_label, confidence_badge_class, confidence_alert_class = get_confidence_label(score)
            match_percent = estimate_match_percent(score)

            top_symptoms = top_result.get("symptoms", [])
            top_management = top_result.get("management", [])
            top_cause = safe_str(top_result.get("cause_description"))
            top_name = safe_str(top_result.get("name")) or "Unknown"
            top_crop = safe_str(top_result.get("crop")) or "N/A"
            top_scientific = safe_str(top_result.get("scientific_name")) or "N/A"
            top_type = safe_str(top_result.get("diagnosis_type")) or "N/A"

            alternative_count = len(other_results)
            image_conf_pct = int(round(image_predicted_score * 100)) if uploaded_file is not None else 0
            recommendation_count = len(top_management) if isinstance(top_management, list) else 0

            risk_level = risk_result.get("risk_level", "Low")
            risk_score = risk_result.get("risk_score", 0)
            risk_reason = risk_result.get("reason", "")
            risk_banner_class, final_alert_class, risk_badge_class = get_risk_ui(risk_level)

            # -----------------------------
            # TOP DASHBOARD
            # -----------------------------
            left_col, right_col = st.columns([1.03, 1.08])

            with left_col:
                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Uploaded Image Analysis</div>', unsafe_allow_html=True)

                if uploaded_file is not None:
                    uploaded_file.seek(0)
                    st.image(uploaded_file, use_container_width=True)
                else:
                    st.markdown(
                        '<div class="note-box">No image uploaded. Diagnosis is based on symptom text only.</div>',
                        unsafe_allow_html=True
                    )

                st.markdown(
                    f"""
                    <div class="stat-line">✅ <b>Diagnosis:</b> {top_name} <i>({top_scientific})</i></div>
                    <div class="stat-line">✅ <b>Severity / Match:</b> {match_percent}% retrieval match</div>
                    {"<div class='stat-line'>✅ <b>Image Model Label:</b> " + image_predicted_label + "</div>" if uploaded_file is not None else ""}
                    {"<div class='stat-line'>✅ <b>Image Confidence:</b> " + str(image_conf_pct) + "%</div>" if uploaded_file is not None else ""}
                    <div class="status-badge {confidence_badge_class}">{confidence_label}</div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Management Recommendations</div>', unsafe_allow_html=True)

                if top_management:
                    for item in top_management[:5]:
                        st.markdown(f"<div class='stat-line'>✅ {item}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='stat-line'>No management data available.</div>", unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    st.button("👤 Submit for Expert Review", key="expert_review")
                with action_col2:
                    st.button("💾 Save Report", key="save_report")

            with right_col:
                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Weather & Risk Assessment</div>', unsafe_allow_html=True)

                if weather_data.get("success"):
                    st.markdown(
                        f"""
                        <div class="stat-line"><b>Location:</b> {weather_data.get("location", "N/A")}</div>
                        <div class="stat-line"><b>Humidity:</b> {weather_data.get("humidity", "N/A")}%</div>
                        <div class="stat-line"><b>Temperature:</b> {weather_data.get("temperature_c", "N/A")}°C</div>
                        <div class="stat-line"><b>Rain Forecast / Current Rain:</b> {weather_data.get("rainfall_mm", "N/A")} mm</div>
                        <div class="stat-line"><b>Condition:</b> {weather_data.get("weather_desc", "N/A")}</div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown("<div class='stat-line'>Weather data unavailable.</div>", unsafe_allow_html=True)

                st.markdown(
                    f"""
                    <div class="{risk_banner_class}">
                        ⚠ Risk Level: {str(risk_level).upper()}<br>
                        <span style="font-size:0.95rem; font-weight:700;">{risk_reason}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">Data Dashboard</div>', unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)

                with m1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">Disease Incidence</div>
                            <div class="metric-value">{match_percent}%</div>
                            <div class="metric-sub">{top_name}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with m2:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">Yield Loss Comparison</div>
                            <div class="metric-value">${max(25, recommendation_count * 7)}</div>
                            <div class="metric-sub">Estimated treatment action cost</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with m3:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-title">Weather Risk Index</div>
                            <div class="metric-value">{risk_score}</div>
                            <div class="metric-sub">Risk score out of 100</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown('</div>', unsafe_allow_html=True)

            # -----------------------------
            # ALERT STRIP
            # -----------------------------
            st.markdown(
                f"""
                <div class="alert-strip {final_alert_class}">
                    ⚠ Alert: {str(risk_level).title()} risk of disease spread. Recommend immediate review and action.
                </div>
                """,
                unsafe_allow_html=True
            )

            # -----------------------------
            # AI EXPLANATION
            # -----------------------------
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">AI Explanation</div>', unsafe_allow_html=True)
            st.write(explanation)
            st.markdown('</div>', unsafe_allow_html=True)

            # -----------------------------
            # MAIN RESULT CARD
            # -----------------------------
            st.markdown(
                f"""
                <div class="result-highlight">
                    <div class="result-title">{top_name}</div>
                    <span class="info-chip">Crop: {top_crop}</span>
                    <span class="info-chip">Scientific Name: {top_scientific}</span>
                    <span class="info-chip">Type: {top_type}</span>
                    <span class="info-chip">Score: {score:.3f}</span>
                    {"<div style='margin-top:12px; color:#e8f0e2;'><b>Cause Description:</b> " + top_cause + "</div>" if top_cause else ""}
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander("Symptoms of likely diagnosis", expanded=True):
                if top_symptoms:
                    for item in top_symptoms:
                        st.write(f"- {item}")
                else:
                    st.write("No symptom data available.")

            with st.expander("Management of likely diagnosis", expanded=True):
                if top_management:
                    for item in top_management:
                        st.write(f"- {item}")
                else:
                    st.write("No management data available.")

            # -----------------------------
            # OTHER POSSIBLE MATCHES
            # -----------------------------
            if other_results:
                st.markdown('<div class="section-title" style="margin-top:18px;">Other Possible Matches</div>', unsafe_allow_html=True)

                for idx, result in enumerate(other_results, start=2):
                    symptoms = result.get("symptoms", [])
                    management = result.get("management", [])
                    cause_description = safe_str(result.get("cause_description"))
                    result_score = float(result.get("score", 0.0))
                    result_percent = estimate_match_percent(result_score)

                    st.markdown(
                        f"""
                        <div class="dashboard-card">
                            <div class="card-title">{idx}. {safe_str(result.get("name")) or "Unknown Disease"}</div>
                            <div class="stat-line"><b>Crop:</b> {safe_str(result.get("crop")) or "N/A"}</div>
                            <div class="stat-line"><b>Scientific Name:</b> {safe_str(result.get("scientific_name")) or "N/A"}</div>
                            <div class="stat-line"><b>Diagnosis Type:</b> {safe_str(result.get("diagnosis_type")) or "N/A"}</div>
                            {"<div class='stat-line'><b>Cause Description:</b> " + cause_description + "</div>" if cause_description else ""}
                            <div class="stat-line"><b>Match Score:</b> {result_score:.3f} ({result_percent}%)</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    with st.expander(f"Symptoms - {safe_str(result.get('name')) or 'Disease'}"):
                        if symptoms:
                            for item in symptoms:
                                st.write(f"- {item}")
                        else:
                            st.write("No symptom data available.")

                    with st.expander(f"Management - {safe_str(result.get('name')) or 'Disease'}"):
                        if management:
                            for item in management:
                                st.write(f"- {item}")
                        else:
                            st.write("No management data available.")