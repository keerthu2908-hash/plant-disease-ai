import json
import re

def clean_cause_and_organism(data):
    for item in data:
        cause = item.get("cause", "")

        if cause:
            # Extract scientific name (first 2 words usually)
            words = cause.split()
            if len(words) >= 2:
                organism = " ".join(words[:2])
            else:
                organism = cause

            # Detect type
            disease_type = item.get("type", "").lower()

            if disease_type == "fungal":
                item["cause"] = "Fungal infection"
            elif disease_type == "bacterial":
                item["cause"] = "Bacterial infection"
            elif disease_type == "viral":
                item["cause"] = "Viral infection"
            else:
                item["cause"] = "Pathogenic infection"

            # Move organism
            item["causal_organism"] = organism

    return data


with open("disease_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

updated_data = clean_cause_and_organism(data)

with open("disease_data_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(updated_data, f, indent=2)

print("Fixed successfully!")