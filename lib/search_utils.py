import string
import json
from nltk.stem import PorterStemmer
from typing import Any, TypedDict

class Movie(TypedDict):
    id: int
    title: str
    description: str


class SearchResult(TypedDict):
    id: int
    title: str
    document: str
    score: float
    metadata: dict[str, Any]


class GoldenTestCase(TypedDict):
    query: str
    relevant_docs: list[str]


class GoldenDataset(TypedDict):
    test_cases: list[GoldenTestCase]


BM25_K1 = 1.5
BM25_B = 0.75
SCORE_PRECISION = 3


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

def format_search_result(
    doc_id: int, title: str, document: str, score: float, **metadata: Any
) -> SearchResult:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }
