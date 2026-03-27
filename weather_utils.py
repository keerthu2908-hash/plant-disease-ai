import requests


def get_weather_data(location):
    """
    Fetch current weather data from wttr.in
    No API key required.
    """
    try:
        url = f"https://wttr.in/{location}?format=j1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        current = data["current_condition"][0]

        temperature_c = float(current.get("temp_C", 0))
        humidity = float(current.get("humidity", 0))
        weather_desc = current.get("weatherDesc", [{}])[0].get("value", "Unknown")
        rainfall_mm = 0.0

        # wttr.in sometimes gives precipMM as current precipitation
        try:
            rainfall_mm = float(current.get("precipMM", 0))
        except:
            rainfall_mm = 0.0

        return {
            "location": location,
            "temperature_c": temperature_c,
            "humidity": humidity,
            "weather_desc": weather_desc,
            "rainfall_mm": rainfall_mm,
            "success": True
        }

    except Exception as e:
        return {
            "location": location,
            "temperature_c": None,
            "humidity": None,
            "weather_desc": "Unavailable",
            "rainfall_mm": None,
            "success": False,
            "error": str(e)
        }


def calculate_risk(diagnosis_name, diagnosis_type, weather):
    """
    Simple risk scoring logic.
    You can improve this later disease-by-disease.
    """
    if not weather.get("success"):
        return {
            "risk_score": 0,
            "risk_level": "Unknown",
            "reason": "Weather data unavailable."
        }

    humidity = weather.get("humidity", 0)
    temp = weather.get("temperature_c", 0)
    rain = weather.get("rainfall_mm", 0)

    risk_score = 0
    reasons = []

    diagnosis_type = str(diagnosis_type).lower()
    diagnosis_name = str(diagnosis_name).lower()

    # General fungal disease-friendly conditions
    fungal_keywords = [
        "blight", "mildew", "rust", "leaf spot", "anthracnose",
        "rot", "wilt", "blast", "smut", "fungal"
    ]

    is_fungal_like = any(word in diagnosis_name for word in fungal_keywords)

    # Disease logic
    if diagnosis_type == "disease" or is_fungal_like:
        if humidity >= 85:
            risk_score += 35
            reasons.append("High humidity favors disease development.")
        elif humidity >= 70:
            risk_score += 20
            reasons.append("Moderately high humidity may support disease spread.")

        if 20 <= temp <= 30:
            risk_score += 25
            reasons.append("Temperature is favorable for many foliar diseases.")
        elif 15 <= temp < 20 or 30 < temp <= 34:
            risk_score += 10
            reasons.append("Temperature is somewhat supportive.")

        if rain > 0:
            risk_score += 25
            reasons.append("Rain or leaf wetness can increase infection spread.")

    # Pest logic
    elif diagnosis_type == "pest":
        if 24 <= temp <= 34:
            risk_score += 30
            reasons.append("Warm temperature can support pest multiplication.")
        elif 20 <= temp < 24:
            risk_score += 15
            reasons.append("Temperature is somewhat favorable for pests.")

        if humidity >= 60:
            risk_score += 15
            reasons.append("Humidity may support pest survival.")
        if rain == 0:
            risk_score += 15
            reasons.append("Dry conditions may support some pest activity.")

    # General bonuses
    if humidity >= 90:
        risk_score += 10
    if rain >= 2:
        risk_score += 10

    risk_score = min(risk_score, 100)

    if risk_score >= 70:
        risk_level = "High"
    elif risk_score >= 40:
        risk_level = "Moderate"
    else:
        risk_level = "Low"

    if not reasons:
        reasons.append("Current weather does not strongly favor rapid spread.")

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "reason": " ".join(reasons)
    }