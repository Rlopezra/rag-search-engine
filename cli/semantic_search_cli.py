import sys
from pathlib import Path

# Force Python to look in the project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from lib.semantic_search import (
    verify_model,
    verify_embeddings,
    embed_text,
    embed_query_text,
    search_movies
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. 'verify' command (takes no additional arguments)
    verify_parser = subparsers.add_parser("verify", help="Run verification")

    # 2. 'embed_text' command (accepts a single string argument)
    embed_parser = subparsers.add_parser("embed_text", help="Embed the provided text")
    embed_parser.add_argument("text", type=str, help="The string text to embed")

    #3. 'verify_embeddings'
    ver_em_parset = subparsers.add_parser("verify_embeddings", help="Verify the embedddings")

    #4. embedding query
    em_qu_parset = subparsers.add_parser("embed_query", help="Embed the query")
    em_qu_parset.add_argument("query", type=str, help="Embed the query")

   #5. 'search' command
    search_parser = subparsers.add_parser("search", help="Search movies by semantic similarity")
    search_parser.add_argument("query", type=str, help="The search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 5)")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            text = args.text
            embed_text(text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search_movies(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()