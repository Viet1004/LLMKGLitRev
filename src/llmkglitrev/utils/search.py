from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
from qdrant_client.models import Filter, FieldCondition, MatchText
    
client = QdrantClient(url="http://localhost:6333")

# Example: Semantic search
def semantic_search(tokenizer, model, query_text, top_k=5):
    # Encode query
    query_inputs = tokenizer(query_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        query_outputs = model(**query_inputs)
    query_embedding = query_outputs.last_hidden_state[:, 0, :].numpy()[0]
    
    # Search
    results = client.query_points(
        collection_name="concepts_collection",
        query=query_embedding.tolist(),
        limit=top_k
    ).points
    
    return [(hit.payload["text"], hit.score) for hit in results]

# Example: Keyword search (text-based filtering)
def keyword_search(keyword, top_k=5):
    results, _ = client.scroll(
        collection_name="concepts_collection",
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="text",
                    match=MatchText(text=keyword)
                )
            ]
        ),
        limit=top_k
    )
    # return results
    return [(point.id, point.payload["text"]) for point in results]
    # return [(point.payload for) point in results]

if __name__ == "__main__":
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     print("Using MPS (Metal Performance Shaders)")
    # elif torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print("Using CUDA")
    # else:
    #     device = torch.device("cpu")
    #     print("Using CPU")
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    model = model.to(device)
    model.eval()

    example = ["korteweg devries equation"]
    # inputs = tokenizer(example, return_tensors='pt', padding=True, truncation=True, max_length=512)
    # inputs = {k: v.to(device) for k, v in inputs.items()}
    # with torch.no_grad():
    #     emb = model(example)
    print("Semantic Search Results:")
    for text, score in semantic_search(tokenizer, model, example):
        print(f"Text: {text}, Score: {score}")
    # vector_params=VectorParams(size=768, distance=Distance.COSINE)
    print("Keyword Search Results:")
    # print(keyword_search("korteweg"))
    for id, text in keyword_search("korteweg"):
        print(f"Text: {text}, ID: {id}")
    # for result in keyword_search("korteweg"):
    #     print(result)

