import streamlit as st
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    HAS_LANGCHAIN = True
except ModuleNotFoundError:
    HAS_LANGCHAIN = False


chain = None


def _build_chain():
    if not HAS_LANGCHAIN:
        return None

    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        return None

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=api_key,
    )

    prompt = ChatPromptTemplate.from_template("""
You are an agricultural expert assistant.

The system has already predicted the disease. Do NOT change it.

Prediction: {disease}
Symptoms: {symptoms}
Weather: {weather}

IMPORTANT RULES:
- If the prediction is "Unknown", empty, or unclear, DO NOT assume a disease.
- If symptoms are not sufficient or confidence is low, DO NOT give a confirmed diagnosis.
- In such cases, say clearly:
  "Unable to identify a reliable disease or pest from the current input."
- Then provide only general guidance (monitoring, basic care, next steps).
- Do NOT hallucinate or guess a disease.

If the prediction is valid and clear, then explain:

1. What the disease is  
2. Why it happens  
3. Immediate treatment  
4. Prevention  

Keep it simple, practical, and farmer-friendly.
""")

    parser = StrOutputParser()
    return prompt | llm | parser


def _fallback_explanation(disease, symptoms, weather):
    return (
        f"Likely issue: {disease}.\n\n"
        f"Possible reasons: {symptoms or 'Symptoms not provided'}, often linked to humid conditions, poor airflow, or crop stress.\n\n"
        "Immediate treatment: remove heavily affected plant parts, keep foliage dry, and apply a crop-appropriate control (fungicide/insecticide/biocontrol) based on local recommendations.\n\n"
        "Prevention: use clean planting material, maintain spacing for airflow, avoid overhead irrigation late in the day, rotate crops, and monitor field conditions regularly.\n\n"
        f"Weather context: {weather or 'Not available'}."
    )


def generate_explanation(disease, symptoms=None, weather=None):
    global chain

    disease_name = disease if disease and str(disease).strip() else "Unknown"
    symptom_text = ", ".join(symptoms) if isinstance(symptoms, list) else symptoms
    weather_text = weather if weather else "Not available"

    if chain is None:
        chain = _build_chain()

    try:
        if chain is None:
            return _fallback_explanation(disease_name, symptom_text, weather_text)

        response = chain.invoke({
            "disease": disease_name,
            "symptoms": symptom_text,
            "weather": weather_text
        })

        return response

    except Exception as e:
        return f"Error generating explanation: {str(e)}"