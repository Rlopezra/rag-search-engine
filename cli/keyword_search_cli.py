import argparse
from collections import defaultdict, Counter
import json
import string
from nltk.stem import PorterStemmer
import pickle
import sys
import os
import math

with open("data/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read().lower().splitlines()

def load_movies():
    with open("data/movies.json", "r", encoding="utf-8") as file:
        movie_data = json.load(file)
    return movie_data

def tokenize_text(text: str) -> list[str]:
    stemmer = PorterStemmer()
    translator = str.maketrans('', '', string.punctuation)
    strip_text =text.translate(translator).lower().split()
    non_empt_tokens = [stemmer.stem(w) for w in strip_text if w and w not in stopwords]
    return non_empt_tokens

def tok_single_term(term):
    token = tokenize_text(term)
    if len(token) > 1:
        raise ValueError("Not a single token")
     
    return token[0]

def bm25_idf_comman(term):
    index = InvertedIndex()
    try:
        index.load()
    except Exception as e:
        print(e)
        sys.exit()

    clean_term = tok_single_term(term)

    return index.get_bm25_idf(clean_term)

def bm25_tf_command(doc_id, term, k1):
    index = InvertedIndex()
    try:
        index.load()
    except Exception as e:
        print(e)
        sys.exit()    
    
    clean_term = tok_single_term(term)

    return index.get_bm25_tf(doc_id, clean_term, k1)


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)

        self.term_frequencies[doc_id].update(tokens)

        for tok in tokens:
            self.index[tok].add(doc_id)

    
    def get_documents(self, term):
        terms_doc_ids = self.index.get(term, set())
        sorted_results = sorted(terms_doc_ids)

        return sorted_results
    
    def get_tf(self, doc_id, term):
        return self.term_frequencies[doc_id][term]
    
    def get_bm25_idf(self, term:str) -> float:
        N = len(self.docmap)
        df = len(self.get_documents(term))

        return math.log((N - df + 0.5) / (df + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id, term, k1):
        tf = self.get_tf(doc_id, term)
        bm25 = (tf * (k1 + 1)) / (tf + k1)

        return bm25
    
    def build(self):
        movie_data = load_movies()

        for movie in movie_data["movies"]:
            join_title_desc = f"{movie['title']} {movie['description']}"
            self.docmap[movie["id"]] = movie
            self.__add_document(movie["id"], join_title_desc)

    def save(self):
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as index_file:
            pickle.dump(self.index, index_file)

        with open("cache/docmap.pkl", "wb") as docmap_file:
            pickle.dump(self.docmap, docmap_file)

        with open("cache/term_frequencies.pkl", "wb") as counter_file:
            pickle.dump(self.term_frequencies, counter_file)

    def load(self):
        #Index loading
        if not os.path.exists("cache/index.pkl"):
            raise FileNotFoundError(f"Error: Index file does not exist.")
        
        with open("cache/index.pkl", "rb") as file:
            self.index = pickle.load(file)

        #docmap loading
        if not os.path.exists("cache/docmap.pkl"):
            raise FileNotFoundError(f"Error: DocMap file doesn't exist.")

        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = pickle.load(file)

        #counter loading
        if not os.path.exists("cache/term_frequencies.pkl"):
            raise FileNotFoundError(f"Error: Term Frequency file doesn't exist.")

        with open("cache/term_frequencies.pkl", "rb") as file:
            self.term_frequencies = pickle.load(file)


def build_command():
    index = InvertedIndex()
    index.build()
    index.save()


def main() -> None:
    BM25_K1 = 1.5
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    #build parser
    subparsers.add_parser("build", help="Build the inverted index and save it to disk")

    #term frequency parser
    tf_parser = subparsers.add_parser("tf", help="Search the term frequency for in a doc")
    tf_parser.add_argument("doc_id", type=int, help="doc id to search in")
    tf_parser.add_argument("term", type=str, help="Term to get its term frequency")

    #idf
    idf_parser = subparsers.add_parser("idf", help="get the term idf")
    idf_parser.add_argument("idf_term", type=str, help = "calculate term's idf" )

    #tf-idf
    tf_idf_parser = subparsers.add_parser("tfidf", help="Calculate the tf-idf")
    tf_idf_parser.add_argument("tfidf_docid", type=int, help= "Docid for the td-idf calculation")
    tf_idf_parser.add_argument("tfidf_term", type=str, help= "Term for the td-idf calculation")

    #bm-25-idf parser
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    #bm25-tf parser
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
        
    args = parser.parse_args()
    index = InvertedIndex()
    match args.command:
        case "search":
            try:
                index.load()
            except Exception as e:
                print(e)
                sys.exit()
            search_term_strip = tokenize_text(args.query)
            print(f"Searching for: {args.query}")

            for search_token in search_term_strip:
                results = index.get_documents(search_token)
                for i, item in enumerate(results[:5], start=1):
                    print(f"{i}. {index.docmap[item]}")
            pass
        case "build":
            build_command()
        case "tf":
            try:
                index.load()
            except Exception as e:
                print(e)
                sys.exit()
            token = tok_single_term(args.term)
            print(index.get_tf(args.doc_id, token))
        case "idf":
            try:
                index.load()
            except Exception as e:
                print(e)
                sys.exit()
            term = tok_single_term(args.idf_term)
            term_match_doc_count = len(index.get_documents(term))
            doc_count = len(index.docmap)
            idf = math.log((doc_count+1)/(term_match_doc_count+1))
            print(f"Inverse document frequency of '{args.idf_term}': {idf:.2f}")
        case "tfidf":
            try:
                index.load()
            except Exception as e:
                print(e)
                sys.exit()
            
            term = tok_single_term(args.tfidf_term)
            
            #tf
            doc_id = args.tfidf_docid
            tf = index.get_tf(doc_id, term)


            #idf
            term_match_doc_count = len(index.get_documents(term))
            doc_count = len(index.docmap)
            idf = math.log((doc_count+1)/(term_match_doc_count+1))

            #tf-idf
            tf_idf = tf * idf

            print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            bm25idf = bm25_idf_comman(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1)

            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()