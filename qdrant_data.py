from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
import numpy as np
import os

# Initialize Qdrant Cloud client (replace 'your-cluster-url' and 'your-api-key')
client = QdrantClient(
    url="https://cc595f43-0fbd-4cb0-9dce-b53a64920c4d.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="5d-hyrAIstlGzaVnXNeRJH4LXct9i26SFnrvDPZfu1VMb2pssYqHSQ",
)

# Initialize the Sentence Transformer model for generating text embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to load all descriptions from text files in a folder structure
def load_descriptions_from_folder(base_folder):
    descriptions = []
    
    # Traverse through the subfolders and txt files
    for subdir, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".txt"):  # Only process .txt files
                file_path = os.path.join(subdir, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()  # Read the content and remove leading/trailing whitespace
                    descriptions.append(content)  # Add the content to descriptions list
    return descriptions

# Define the base folder where your descriptions are stored
base_folder = "./vector_dataset/description/"  # Replace with your actual folder path
descriptions = load_descriptions_from_folder(base_folder)

# Convert the descriptions to embeddings
description_embeddings = model.encode(descriptions)

# Normalize embeddings (optional, depends on the model)
description_embeddings = np.array([embedding / np.linalg.norm(embedding) for embedding in description_embeddings])

# Create a Qdrant collection for storing the outfit descriptions
collection_name = "outfit_descriptions"
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=description_embeddings.shape[1], distance="Cosine")
)

# Upload embeddings to Qdrant Cloud
client.upload_collection(
    collection_name=collection_name,
    vectors=description_embeddings,
    payload=[{"description": desc} for desc in descriptions],
    ids=[i for i in range(len(descriptions))]
)

print(f"Uploaded {len(descriptions)} descriptions to Qdrant.")
