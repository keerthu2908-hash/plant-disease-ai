import streamlit as st
from matcher import load_data, get_all_categories
from retriever import search_disease
from reranker import rerank_results
from explainer import generate_explanation

st.set_page_config(page_title="Plant Disease Chatbot", page_icon="🌿", layout="centered")


# -----------------------------
# Helper functions
# -----------------------------
def get_confidence_label(score):
    if score >= 0.80:
        return "High"
    elif score >= 0.60:
        return "Medium"
    return "Low"


def score_to_percent(score):
    return round(score * 100, 1)


def extract_matched_symptoms(user_input, stored_symptoms):
    user_words = set(user_input.lower().replace(",", " ").split())
    matched = []

    for symptom in stored_symptoms:
        symptom_words = set(str(symptom).lower().replace(",", " ").split())
        if user_words.intersection(symptom_words):
            matched.append(symptom)

    return list(dict.fromkeys(matched))


def build_query(category, crop, disease_type, symptoms):
    parts = [
        f"category {category}",
        f"crop {crop}",
        f"symptoms {symptoms}"
    ]

    if disease_type:
        parts.append(f"disease type {disease_type}")
    else:
        parts.append("fungal bacterial viral plant disease")

    return " ".join(parts)


def display_management(management):
    if isinstance(management, list):
        for tip in management:
            st.write(f"- {tip}")
    elif isinstance(management, str) and management.strip():
        st.write(f"- {management}")
    else:
        st.write("- Management information not available.")


# -----------------------------
# UI
# -----------------------------
st.title("🌿 Plant Disease AI Assistant")
st.write("Select the crop and enter symptoms to find the most likely disease.")

data = load_data()

if not data:
    st.error("No disease data found. Please check your dataset.")
    st.stop()

category_list = get_all_categories(data)

if not category_list:
    st.error("No categories found in the dataset.")
    st.stop()

category = st.selectbox("Select category", category_list, key="category")

filtered_crops = sorted(
    {
        item["crop"].capitalize()
        for item in data
        if item.get("category", "").capitalize() == category
    }
)

if not filtered_crops:
    st.warning(f"No crops available for category: {category}")
    st.stop()

crop = st.selectbox("Select crop", filtered_crops, key="crop")

disease_type = st.selectbox(
    "Select disease type (optional)",
    ["All", "fungal", "bacterial", "viral"],
    key="disease_type"
)

selected_disease_type = None if disease_type == "All" else disease_type

symptoms = st.text_area(
    "Enter symptoms",
    placeholder="Example: brown patches on sheath, oval lesions, lodging",
    key="symptoms"
)

if st.button("Check Diagnosis", use_container_width=True):
    if not symptoms.strip():
        st.warning("Please enter symptoms.")
    else:
        query = build_query(category, crop, selected_disease_type, symptoms)
        results = search_disease(query)

        filtered_results = []

        for match in results:
            meta = match.get("metadata", {})

            if meta.get("category", "").lower() != category.lower():
                continue

            if meta.get("crop", "").lower() != crop.lower():
                continue

            if selected_disease_type is not None and meta.get("type", "").lower() != selected_disease_type.lower():
                continue

            filtered_results.append(match)

        filtered_results = rerank_results(query, filtered_results)

        if filtered_results:
            top_match = filtered_results[0]
            top_data = top_match["metadata"]

            vector_score = top_match["score"]
            confidence_percent = score_to_percent(vector_score)
            confidence_label = get_confidence_label(vector_score)
            matched = extract_matched_symptoms(symptoms, top_data.get("symptoms", []))

            disease_name = top_data.get("disease", top_data.get("disease_name", "Unknown"))

            symptom_data = top_data.get("symptoms")

            if not symptom_data:
                symptom_data = symptoms  # fallback to user input

            explanation = generate_explanation(disease_name, symptom_data)
            st.divider()

            st.subheader("🩺 Assistant Answer")
            st.success(explanation)

            st.success("Diagnosis completed")

            st.markdown(f"## 🌱 Most likely disease: **{top_data.get('disease', 'Unknown')}**")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Disease type:** {top_data.get('type', 'N/A').capitalize()}")
                st.write(f"**Crop:** {top_data.get('crop', 'N/A').capitalize()}")
                st.write(f"**Category:** {top_data.get('category', 'N/A').capitalize()}")

            with col2:
                st.metric("Confidence", f"{confidence_percent}%")
                st.write(f"**Confidence level:** {confidence_label}")
                st.write(f"**AI ranking score:** {round(top_match.get('rerank_score', 0), 4)}")

            st.subheader("🔍 Symptom Match Analysis")
            if matched:
                st.write("Your input matches these key symptoms:")
                for m in matched:
                    st.write(f"✔ {m}")
            else:
                st.write(
                    f"This result was retrieved because your symptom description is semantically similar "
                    f"to the stored profile for **{top_data.get('disease', 'Unknown')}**."
                )

            st.subheader("🧬 Causal Organism & Disease Cause")
            st.write(f"**Causal Organism:** {top_data.get('causal_organism', 'Not available')}")
            st.write(f"**Cause description:** {top_data.get('cause', 'Not available')}")

            st.subheader("💊 Recommended Management")
            display_management(top_data.get("management", []))

            with st.expander("See stored symptom profile"):
                for symptom_item in top_data.get("symptoms", []):
                    st.write(f"- {symptom_item}")

            if len(filtered_results) > 1:
                st.subheader("Other possible matches")

                for match in filtered_results[1:4]:
                    meta = match["metadata"]
                    alt_vector_score = match["score"]
                    alt_percent = score_to_percent(alt_vector_score)
                    alt_label = get_confidence_label(alt_vector_score)

                    st.markdown(f"### {meta.get('disease', 'Unknown disease')}")
                    st.write(f"**Type:** {meta.get('type', 'N/A').capitalize()}")
                    st.write(f"**Confidence:** {alt_percent}% ({alt_label})")
                    st.write(f"**AI ranking score:** {round(match.get('rerank_score', 0), 4)}")
                    st.write(f"**Causal Organism:** {meta.get('causal_organism', 'Not available')}")

                    st.write("**Management:**")
                    display_management(meta.get("management", []))

                    with st.expander(f"View symptoms for {meta.get('disease', 'Unknown disease')}"):
                        for symptom_item in meta.get("symptoms", []):
                            st.write(f"- {symptom_item}")
        else:
            st.error("No match found for the selected filters and symptoms.")