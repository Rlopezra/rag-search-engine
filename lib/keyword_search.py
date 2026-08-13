from collections import defaultdict, Counter
import pickle
import sys
import os
import math

from lib.search_utils import (
    BM25_K1,
    BM25_B,
    load_movies,
    tokenize_text,
    tok_single_term,
)


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)
        for tok in tokens:
            self.index[tok].add(doc_id)

    def get_documents(self, term):
        terms_doc_ids = self.index.get(term, set())
        return sorted(terms_doc_ids)

    def get_tf(self, doc_id, term):
        return self.term_frequencies[doc_id][term]

    def get_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.get_documents(term))
        return math.log((N + 1) / (df + 1))

    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.get_documents(term))
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_bm25_tf(self, doc_id, term, k1, b):
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        bm25_tf = self.get_bm25_tf(doc_id, term, k1, b)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit=5):
        tokens = tokenize_text(query)
        score_dict = {}
        for doc_id in self.docmap:
            total_score = 0.0
            for token in tokens:
                total_score += self.bm25(doc_id, token)
            score_dict[doc_id] = total_score
        ranked = sorted(score_dict.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:limit]

    def build(self):
        movie_data = load_movies()
        for movie in movie_data["movies"]:
            join_title_desc = f"{movie['title']} {movie['description']}"
            self.docmap[movie["id"]] = movie
            self.__add_document(movie["id"], join_title_desc)

    def save(self):
        os.makedirs("cache", exist_ok=True)
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open("cache/doc_lengths.pkl", "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        for name, attr in [
            ("index", "index"),
            ("docmap", "docmap"),
            ("term_frequencies", "term_frequencies"),
            ("doc_lengths", "doc_lengths"),
        ]:
            path = f"cache/{name}.pkl"
            if not os.path.exists(path):
                raise FileNotFoundError(f"Error: {name} cache file does not exist.")
            with open(path, "rb") as f:
                setattr(self, attr, pickle.load(f))


def _load_index() -> InvertedIndex:
    index = InvertedIndex()
    try:
        index.load()
    except Exception as e:
        print(e)
        sys.exit()
    return index


def build_command():
    index = InvertedIndex()
    index.build()
    index.save()


def search_command(query, limit=5):
    index = _load_index()
    tokens = tokenize_text(query)

    matched_ids = set()
    for token in tokens:
        matched_ids.update(index.get_documents(token))

    results = []
    for doc_id in sorted(matched_ids)[:limit]:
        movie = index.docmap[doc_id]
        results.append({"id": doc_id, "title": movie["title"]})
    return results


def tf_command(doc_id, term):
    index = _load_index()
    clean_term = tok_single_term(term)
    return index.get_tf(doc_id, clean_term)


def idf_command(term):
    index = _load_index()
    clean_term = tok_single_term(term)
    return index.get_idf(clean_term)


def tfidf_command(doc_id, term):
    index = _load_index()
    clean_term = tok_single_term(term)
    tf = index.get_tf(doc_id, clean_term)
    idf = index.get_idf(clean_term)
    return tf * idf


def bm25_idf_command(term):
    index = _load_index()
    clean_term = tok_single_term(term)
    return index.get_bm25_idf(clean_term)


def bm25_tf_command(doc_id, term, k1, b):
    index = _load_index()
    clean_term = tok_single_term(term)
    return index.get_bm25_tf(doc_id, clean_term, k1, b)


def bm25search_command(query, limit=5):
    index = _load_index()
    ranked = index.bm25_search(query, limit)

    results = []
    for doc_id, score in ranked:
        movie = index.docmap[doc_id]
        results.append({"id": doc_id, "title": movie["title"], "score": score})
    return results
