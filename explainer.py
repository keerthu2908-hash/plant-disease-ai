from transformers import pipeline

# Load once
generator = pipeline("text-generation", model="distilgpt2")

def generate_explanation(disease, symptoms, matched):
    if matched:
        return (
            f"Based on the symptoms you provided such as {', '.join(matched)}, "
            f"the condition closely matches {disease}. "
            f"These symptoms are commonly associated with this disease pattern."
        )
    else:
        return (
            f"The system identified {disease} because your overall symptom description "
            f"is similar to known cases of this disease, even though exact symptom matches were limited."
        )