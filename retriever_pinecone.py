import streamlit as st
from pinecone import Pinecone
from embedder import create_embedding

pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index("plant-disease-384")  # replace with real name

def find_best_matches(query, selected_crop="All", top_k=5):
    query_embedding = create_embedding(query.strip().lower())

    response = index.query(
        vector=query_embedding,
        top_k=top_k * 3,
        include_metadata=True
    )

    results = []
    for match in response["matches"]:
        item = match["metadata"]
        item["score"] = match["score"]
        results.append(item)

    if selected_crop != "All":
        crop_filtered = [
            r for r in results
            if r.get("crop", "").strip().lower() == selected_crop.strip().lower()
        ]
        if crop_filtered:
            return crop_filtered[:top_k]

    return results[:top_k]

    return results