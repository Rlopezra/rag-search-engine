from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import numpy as np
import re

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def embed_text(text):
    sem_model = SemanticSearch()

    embedding = sem_model.generate_embedding(text)
    
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    sem_model = SemanticSearch()
    with open("data/movies.json", "r", encoding="utf-8") as file:
        movie_data = json.load(file)
    
    movie_list = movie_data.get('movies',[])

    embeddings = sem_model.load_or_create_embeddings(movie_list)

    print(f"Number of docs:   {len(movie_list)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )

def embed_query_text(query):
    sem_model = SemanticSearch()
    embedding = sem_model.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def search_movies(query, limit=5):
    sem_model = SemanticSearch()
    with open("data/movies.json", "r", encoding="utf-8") as file:
        movie_data = json.load(file)

    movie_list = movie_data.get('movies', [])
    sem_model.load_or_create_embeddings(movie_list)

    results = sem_model.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:\n")
    for i, r in enumerate(results, start=1):
        print(f"{i}. {r['title']} (score: {r['score']:.4f})")
        print(f"   {r['description']}\n")

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, n, overlap):
    text_split = text.split()
    step = n-overlap
    grouped = [' '.join(text_split[i : i + n]) for i in range(0, len(text_split), step)]

    return grouped

def sem_chunk_text(text, n, overlap):
    step = n - overlap
    chunks = []
    start = 0
    while start < len(text):
        end = start + n
        chunks.append(' '.join(text[start:end]))
        if end >= len(text):
            break
        start += step
    return chunks

def chunk_text_print_format(text, result):
    print(f"Chunking {len(text)} characters")
    for i, r in enumerate(result, start=1):
        print(f"{i}. {r}")

def sem_chunk_text_print_format(text, result):
    print(f"Semantically chunking {len(text)} characters")
    for i, r in enumerate(result, start=1):
        print(f"{i}. {r}")


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if len(text.strip()) == 0:
            raise ValueError("You entereted an empty string")
        
        result = self.model.encode([text])
        
        return result[0]

    def build_embeddings(self, documents):
        self.documents = documents

        doc_list = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_list.append(f"{doc['title']}: {doc['description']}")
        
        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)
        np.save('cache/movie_embeddings.npy', self.embeddings)

        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc

        em_path = 'cache/movie_embeddings.npy'

        if os.path.exists(em_path):
            self.embeddings = np.load(em_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)
    
    def search(self, query, limit):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        embeddings = self.generate_embedding(query)

        sim_list = []
        for num, emb in enumerate(self.embeddings):
            sim_scpre = cosine_similarity(embeddings, emb)
            sim_list.append((sim_scpre, self.documents[num]))
        
        result = sorted(sim_list, reverse=True, key=lambda x: x[0])
        
        results = []
        for score, doc in result[:limit]:
            results.append({
            "score": float(score),
            "title": doc["title"],
            "description": doc["description"],
        })
        return results



def verify_model():
    model = SemanticSearch()

    print(f'Model loaded: {model}')
    print(f'Max sequence length: {model.model.max_seq_length}')
