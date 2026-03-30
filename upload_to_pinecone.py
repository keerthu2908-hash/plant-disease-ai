import json
from pinecone_db import upload_vectors

input_file = "master_diseases_embedded.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

vectors = []

for i, entry in enumerate(data, start=1):
    vectors.append({
        "id": str(i),
        "values": entry.get("embedding", []),
        "metadata": {
            "crop": entry.get("crop", ""),
            "name": entry.get("name", ""),
            "scientific_name": entry.get("scientific_name", ""),
            "diagnosis_type": entry.get("diagnosis_type", ""),
            "cause_description": entry.get("cause_description", ""),
            "symptoms": " | ".join(entry.get("symptoms", [])),
            "management": " | ".join(entry.get("management", []))
        }
    })

print(f"Prepared {len(vectors)} vectors for upload.")

# optional debug
print(vectors[0])

upload_vectors(vectors)

print("Upload complete.")