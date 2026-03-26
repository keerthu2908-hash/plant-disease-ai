from transformers import pipeline

# Load once
generator = pipeline("text-generation", model="distilgpt2")

def generate_explanation(disease, symptoms):
    if isinstance(symptoms, list):
        symptom_text = ", ".join(symptoms)
    else:
        symptom_text = symptoms

    return (
        f"The symptoms you entered strongly indicate **{disease}**. "
        f"Typical signs of this disease include {symptom_text}. "
        f"This disease commonly affects plants under favorable environmental conditions "
        f"and requires timely management to prevent spread."
    )