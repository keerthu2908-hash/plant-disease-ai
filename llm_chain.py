import os
from langchain_openai import ChatOpenAI

def get_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=api_key,
    )

def generate_llm_explanation(disease, symptoms, weather):
    llm = get_llm()

    if llm is None:
        return {
            "reasoning": "LLM explanation is unavailable because the OpenAI API key is not configured."
        }

    prompt = f"""
    Disease: {disease}
    Symptoms: {symptoms}
    Weather: {weather}

    Explain briefly why this disease matches.
    """

    response = llm.invoke(prompt)

    return {
        "reasoning": response.content
    }