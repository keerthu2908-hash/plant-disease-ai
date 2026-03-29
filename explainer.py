import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API key
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=st.secrets["OPENAI_API_KEY"]
)

# Prompt template (same logic, but structured)
prompt = ChatPromptTemplate.from_template("""
You are an agricultural expert assistant.

The system has already predicted the disease. Do NOT change it.

Prediction: {disease}
Symptoms: {symptoms}
Weather: {weather}

Explain clearly:
1. What the disease is
2. Why it happens
3. Immediate treatment
4. Prevention

Keep it simple and practical.
""")

# Output parser
parser = StrOutputParser()

# Chain
chain = prompt | llm | parser


def generate_explanation(disease, symptoms=None, weather=None):
    symptom_text = ", ".join(symptoms) if isinstance(symptoms, list) else symptoms
    weather_text = weather if weather else "Not available"

    try:
        response = chain.invoke({
            "disease": disease,
            "symptoms": symptom_text,
            "weather": weather_text
        })

        return response

    except Exception as e:
        return f"Error generating explanation: {str(e)}"