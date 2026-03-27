from embedder import create_embedding
import numpy as np

def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

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

def find_best_matches(query, data, selected_crop="All", top_k=5):
    query_embedding = create_embedding(query.strip().lower())
    results = []

    for entry in data:
        if selected_crop != "All" and entry.get("crop", "").lower() != selected_crop.lower():
            continue

        entry_embedding = entry.get("embedding")
        if not entry_embedding:
            continue

        score = cosine_similarity(query_embedding, entry_embedding)

        result = dict(entry)
        result["score"] = score
        results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]