from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

client = QdrantClient(url="http://localhost:6333")

CONCEPT_DATA_PATH = "/Users/quocviet.nguyen/paper_idea/LLM_KG_LitRev/LLMKGLitRev/data/full_concepts.txt"

# Set device to MPS if available, otherwise CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Metal Performance Shaders)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
else:
    device = torch.device("cpu")
    print("Using CPU")

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
model = model.to(device)
model.eval()

full_concepts = pd.read_csv(CONCEPT_DATA_PATH)

text = full_concepts.iloc[:, 0].tolist()  # Get first column as list of strings

# Process embeddings in batches to avoid RAM explosion
batch_size_encoding = 32  # Adjust based on your GPU/RAM capacity
embeddings_list = []

print(f"Processing {len(text)} concepts in batches of {batch_size_encoding}...")
for i in tqdm(range(0, len(text), batch_size_encoding)):
    batch_texts = text[i:i + batch_size_encoding]
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Move back to CPU and convert to numpy
    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    embeddings_list.append(batch_embeddings)
    
    # Clear cache to free memory
    if device.type in ['mps', 'cuda']:
        torch.mps.empty_cache() if device.type == 'mps' else torch.cuda.empty_cache()

embeddings = np.vstack(embeddings_list)

# Create collection with correct embedding size (SciBERT has 768 dimensions)
embedding_size = embeddings.shape[1]

# Delete if exists and recreate
if client.collection_exists("concepts_collection"):
    client.delete_collection("concepts_collection")

client.create_collection(
    collection_name="concepts_collection",
    vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
)

# Prepare points for insertion
points = []
for idx, (text_item, embedding) in enumerate(zip(text, embeddings)):
    points.append(
        PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload={
                "text": text_item,
                "original_index": idx
            }
        )
    )

# Upload to Qdrant (batch upload for efficiency)
batch_size = 1000
for i in tqdm(range(0, len(points), batch_size)):
    batch = points[i:i + batch_size]
    client.upsert(
        collection_name="concepts_collection",
        points=batch
    )

print(f"Successfully uploaded {len(points)} points to Qdrant")


