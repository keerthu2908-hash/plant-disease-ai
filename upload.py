from embedder import prepare_data
from pinecone_db import upload_vectors

vectors = prepare_data()
upload_vectors(vectors)

print("Uploaded successfully!")