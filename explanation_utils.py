import random

def generate_dynamic_explanation(
    disease_name: str,
    symptoms: list = None,
    weather: dict = None,
    confidence: float = None
):
    """
    Generate short, dynamic AI-style explanations (5 + 3 lines)
    """

    symptoms = symptoms or []
    weather = weather or {}

    humidity = weather.get("humidity")
    temp = weather.get("temperature")

    # --- WHY THIS MATCHES ---
    why_match = []

    # 1. Disease-specific line
    why_match.append(f"Symptoms are consistent with {disease_name.lower()}")

    # 2. Symptoms-based reasoning
    if symptoms:
        why_match.append(f"Observed signs like {random.choice(symptoms)} match this condition")
    else:
        why_match.append("Visible leaf damage matches known patterns")

    # 3. Weather reasoning
    if humidity:
        if humidity > 80:
            why_match.append("High humidity strongly supports disease spread")
        elif humidity > 60:
            why_match.append("Moderate humidity supports infection")
        else:
            why_match.append("Low humidity slightly limits spread")

    if temp:
        if temp < 20:
            why_match.append("Cool temperature favors disease development")
        elif temp < 30:
            why_match.append("Temperature is suitable for disease growth")
        else:
            why_match.append("High temperature may accelerate symptoms")

    # 4. Moisture / environment
    why_match.append("Moist conditions help pathogen survival")

    # 5. Confidence-aware reasoning
    if confidence:
        if confidence > 0.85:
            why_match.append("Model prediction confidence is high")
        elif confidence > 0.65:
            why_match.append("Prediction is moderately reliable")
        else:
            why_match.append("Prediction confidence is low")

    # Ensure max 5 lines
    why_match = why_match[:5]

    # --- DIFFERENCE FROM OTHERS ---
    difference = []

    difference.append(f"{disease_name} shows distinct lesion patterns")

    if "rust" in disease_name.lower():
        difference.append("Produces raised rust-like spots unlike flat lesions")
    elif "blight" in disease_name.lower():
        difference.append("Causes larger dead patches than spot diseases")
    elif "spot" in disease_name.lower():
        difference.append("Spots are smaller and more defined than blight")

    difference.append("Pattern and spread differ from similar diseases")

    # Ensure max 3 lines
    difference = difference[:3]

    return {
        "why_match": why_match,
        "difference": difference
    }