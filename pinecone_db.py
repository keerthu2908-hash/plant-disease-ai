import streamlit as st
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])

index_name = "plant-disease-384"

existing_indexes = pc.list_indexes().names()

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)


def upload_vectors(vectors):
    to_upsert = [
        (v["id"], v["values"], v["metadata"])
        for v in vectors
    ]
    index.upsert(vectors=to_upsert)