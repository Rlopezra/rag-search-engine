import argparse
import json
import string
from nltk.stem import PorterStemmer

def main() -> None:
    stemmer = PorterStemmer()
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    translator = str.maketrans('', '', string.punctuation)

    with open("data/movies.json", "r", encoding="utf-8") as file:
        movie_data = json.load(file)
    result_list = []

    with open("data/stopwords.txt", "r", encoding="utf-8") as file:
        stopwords = file.read()
    stopwords = stopwords.translate(translator).lower()
    stopwords = stopwords.splitlines()


    search_term_strip = args.query.translate(translator).lower().split()
    non_empt_tokens = [stemmer.stem(w) for w in search_term_strip if w and w not in stopwords]

    def match_substring(movie_title, list1, list2):
        for item in list1:
            for item2 in list2:
                if item in item2:
                    return movie_title
        return None
    
    for movie in movie_data["movies"]:
        stripped_movie = movie["title"].translate(translator).lower().split()
        stripped_movie_clean = [stemmer.stem(word) for word in stripped_movie if word and word not in stopwords]
        result = match_substring(movie["title"],non_empt_tokens,  stripped_movie_clean)
        if result:
            result_list.append(result)

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            for i, item in enumerate(result_list[:5], start=1):
                print(f"{i}. {item}")
            pass
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()