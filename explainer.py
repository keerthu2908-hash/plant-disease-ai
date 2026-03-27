def generate_explanation(disease, symptoms):
    if isinstance(symptoms, list):
        symptom_text = ", ".join(symptoms[:3])
    else:
        symptom_text = str(symptoms)

    return (
        f"Based on the symptoms you provided, the most likely condition is {disease}. "
        f"This disease typically presents with {symptom_text}. "
        f"Early identification is important to prevent spread and reduce crop loss. "
        f"Proper management practices should be followed immediately."
    )