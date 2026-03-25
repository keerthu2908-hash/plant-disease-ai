from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

pc = Pinecone(api_key="pcsk_4kYN75_Da1mpsx4eKSUb6G59kiENnsb2mn4nbAJmwT8dfE2NZoKGM8EmGXUTe2NiFU462s")
index = pc.Index("plant-disease-384")


def search_disease(query):
    query_vector = model.encode(query).tolist()

    results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )

    return results["matches"]