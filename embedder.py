import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embedding(text):
    return model.encode(text).tolist()


def prepare_data():
    with open("disease_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    vectors = []

    for i, item in enumerate(data):
        symptoms = item.get("symptoms", [])
        management = item.get("management", [])

        if isinstance(symptoms, list):
            symptoms_text = " ".join(symptoms)
        else:
            symptoms_text = str(symptoms)

        if isinstance(management, list):
            management_text = " ".join(management)
        else:
            management_text = str(management)

        text = f"""
Category: {item.get('category', '')}
Crop: {item.get('crop', '')}
Disease: {item.get('disease', '')}
Type: {item.get('type', '')}
Causal Organism: {item.get('causal_organism', '')}
Symptoms: {symptoms_text}
Management: {management_text}
"""

        embedding = create_embedding(text)
        print(item.get("disease"), "->", item.get("causal_organism"))

        vectors.append({
            "id": str(i),
            "values": embedding,
            "metadata": {
                "category": item.get("category", ""),
                "crop": item.get("crop", ""),
                "disease": item.get("disease", ""),
                "type": item.get("type", ""),
                "symptoms": item.get("symptoms", []),
                "cause": item.get("cause", ""),
                "causal_organism": item.get("causal_organism", ""),
                "management": item.get("management", [])
            }
        })

    return vectors