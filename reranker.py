import streamlit as st
from sentence_transformers import CrossEncoder


@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def build_text(meta):
    return (
        f"{meta.get('disease', '')} "
        f"{meta.get('crop', '')} "
        f"{meta.get('type', '')} "
        f"{meta.get('cause', '')} "
        f"{' '.join(meta.get('symptoms', []))}"
    )


def rerank_results(query, results):
    if not results:
        return results

    model = load_reranker()

    pairs = []
    for r in results:
        text = build_text(r.get("metadata", {}))
        pairs.append((query, text))

    scores = model.predict(pairs)

    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)

    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)