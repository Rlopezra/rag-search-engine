import numpy as np
import json
import os
import re
from lib.semantic_search import (
    SemanticSearch,
    sem_chunk_text,
    cosine_similarity,
)
from lib.search_utils import (
    format_search_result,
)
 
class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        chunk_list = []
        chunk_meta = []

        for index, doc in enumerate(documents):
            self.document_map[doc["id"]] = doc

            if not doc.get("description") or not doc["description"].strip():
                continue
            
            strip_text = doc.get("description").strip()
            if len(strip_text) == 0:
                return []
            sem_split = re.split(r"(?<=[.!?])\s+", strip_text)
            if len(sem_split) == 1 and not sentences[0].endswith(('.', '!', '?')):
                chunk_text = sem_split
            else:
                chunk_text = sem_chunk_text(sem_split, 4, 1)
            total_chunks = len(chunk_text)

            movie_idx = index

            for chu_index, chunk in enumerate(chunk_text):
                strip_chunk = chunk.strip()
                if strip_chunk:
                    chunk_list.append(strip_chunk)
                    chunk_meta.append({
                        "movie_idx": movie_idx,
                        "chunk_idx": chu_index,
                        "total_chunks": total_chunks
                    })
        
        self.chunk_embeddings = self.model.encode(chunk_list, show_progress_bar=True)
        self.chunk_metadata = chunk_meta
        np.save('cache/chunk_embeddings.npy', self.chunk_embeddings)

        cache = "cache/chunk_metadata.json"
        with open(cache, "w") as f:
            json.dump({"chunks": chunk_meta, "total_chunks": len(chunk_list)}, f, indent=2)

        return self.chunk_embeddings
        
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.array:
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        em_path = "cache/chunk_embeddings.npy"
        md_path = "cache/chunk_metadata.json"
        
        if os.path.exists(em_path) and os.path.exists(md_path):
            self.chunk_embeddings = np.load(em_path)
            with open(md_path, 'r') as file:
                data = json.load(file)
                self.chunk_metadata = data["chunks"]

            return self.chunk_embeddings
        
        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = 10):
        query_emb = self.generate_embedding(query)
        scores = []

        for index, em in enumerate(self.chunk_embeddings):
            movie_idx = self.chunk_metadata[index]["movie_idx"]
            sim_score = cosine_similarity(query_emb, em)
            scores.append({
                "chunk_idx": index,
                "movie_idx": movie_idx,
                "score" : sim_score,
            })

        score_mapping = { }

        for score in scores:
            m_idx = score["movie_idx"]
            s_val = score["score"]
            if m_idx not in score_mapping or s_val> score_mapping[m_idx]:
                score_mapping[m_idx] = s_val

        sorted_movies = sorted(score_mapping.items(), key= lambda x: x[1], reverse=True )

        limited_results = sorted_movies[:limit]

        formated_list = []
        for res in limited_results:
            formated_list.append(format_search_result(
                self.documents[res[0]]["id"], 
                self.documents[res[0]]["title"] , 
                self.documents[res[0]]["description"][:100] ,
                res[1]))

        return formated_list
