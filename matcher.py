import json


def load_data():
    with open("disease_data.json", "r", encoding="utf-8") as file:
        return json.load(file)


def get_all_crops(data):
    crops = set()
    for item in data:
        crops.add(item["crop"].capitalize())
    return sorted(list(crops))
def get_all_categories(data):
    categories = set()
    for item in data:
        categories.add(item["category"].capitalize())
    return sorted(list(categories))

def retrieve_diseases(crop, symptoms, data, disease_type=None):
    crop = crop.lower().strip()
    symptoms = symptoms.lower().strip()

    results = []

    for item in data:
        item_crop = item["crop"].lower()
        item_type = item["type"].lower()

        if item_crop != crop:
            continue

        if disease_type is not None and item_type != disease_type.lower():
            continue

        user_words = symptoms.split()
        matched_symptoms = []

        for symptom in item["symptoms"]:
            symptom_lower = symptom.lower()

            for word in user_words:
                if word in symptom_lower:
                    matched_symptoms.append(symptom)
                    break

        if matched_symptoms:
            results.append({
                "category": item["category"],
                "crop": item["crop"],
                "disease": item["disease"],
                "type": item["type"],
                "cause": item["cause"],
                "management": item["management"],
                "matched_symptoms": matched_symptoms,
                "match_count": len(matched_symptoms)
            })

    results.sort(key=lambda x: x["match_count"], reverse=True)
    return results