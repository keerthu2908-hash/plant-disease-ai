from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

prompt = PromptTemplate(
    template="""
You are an expert plant pathologist.

Disease: {disease}
Symptoms: {symptoms}
Weather: {weather}

Explain:
1. Why this disease matches
2. What makes it different from similar diseases
3. Immediate farmer-friendly action

Give a clear, practical explanation.
""",
    input_variables=["disease", "symptoms", "weather"],
)

def generate_llm_explanation(disease, symptoms, weather):
    chain = prompt | llm
    response = chain.invoke({
        "disease": disease,
        "symptoms": ", ".join(symptoms),
        "weather": str(weather)
    })
    return {"reasoning": response.content}