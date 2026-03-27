import json
from embedder import create_embedding

def build_search_text(entry):
    return " ".join([
        (entry.get("name", "") + " ") * 3,
        (entry.get("crop", "") + " ") * 2,
        entry.get("scientific_name", ""),
        entry.get("diagnosis_type", ""),
        entry.get("cause_description", ""),
        " ".join(entry.get("symptoms", [])),
        " ".join(entry.get("management", []))
    ]).lower().strip()

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