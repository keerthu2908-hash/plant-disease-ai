import json
from embedder import create_embedding

def build_search_text(entry):
    return f"""
    Crop: {entry.get('crop', '')}
    Category: {entry.get('category', '')}
    Type: {entry.get('diagnosis_type', '')}
    Name: {entry.get('name', '')}
    Scientific Name: {entry.get('scientific_name', '')}
    Cause: {entry.get('cause_description', '')}
    Symptoms: {'; '.join(entry.get('symptoms', []))}
    Identification: {'; '.join(entry.get('identification', []))}
    Management: {'; '.join(entry.get('management', []))}
    """.lower().strip()

input_file = "master_diseases.json"
output_file = "master_diseases_embedded.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

for i, entry in enumerate(data, start=1):
    text = build_search_text(entry)
    entry["embedding"] = create_embedding(text)
    print(f"Processed {i}/{len(data)}: {entry.get('name', 'Unknown')}")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"\nDone! Saved to {output_file}")