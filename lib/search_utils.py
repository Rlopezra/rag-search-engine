import string
import json
from nltk.stem import PorterStemmer

BM25_K1 = 1.5
BM25_B = 0.75


def load_stopwords() -> list[str]:
    with open("data/stopwords.txt", "r", encoding="utf-8") as file:
        return file.read().lower().splitlines()


STOPWORDS = load_stopwords()


def load_movies():
    with open("data/movies.json", "r", encoding="utf-8") as file:
        movie_data = json.load(file)
    return movie_data


def tokenize_text(text: str) -> list[str]:
    stemmer = PorterStemmer()
    translator = str.maketrans("", "", string.punctuation)
    strip_text = text.translate(translator).lower().split()
    tokens = [stemmer.stem(w) for w in strip_text if w and w not in STOPWORDS]
    return tokens


def tok_single_term(term: str) -> str:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError("Not a single token")
    return tokens[0]
