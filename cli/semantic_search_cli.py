import sys
from pathlib import Path

# Force Python to look in the project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import re
import json
from lib.semantic_search import (
    verify_model,
    verify_embeddings,
    embed_text,
    embed_query_text,
    search_movies,
    chunk_text,
    chunk_text_print_format,
    sem_chunk_text,
    sem_chunk_text_print_format
)
from lib.chunked_semantic_search import (
    ChunkedSemanticSearch
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

    #6. 'chunk' command
    chunk_parser = subparsers.add_parser("chunk", help="The size to split documents to easily digest them.")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="Size to break the documents into")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Size of the overlap chunk")

    #semantic chunk
    semantic_parser = subparsers.add_parser("semantic_chunk", help="Chunk them by semantics.")
    semantic_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_parser.add_argument("--max-chunk-size", type=int, default=4, help="Size to break the documents into")
    semantic_parser.add_argument("--overlap", type=int, default=0, help="Size of the overlap chunk")

    #embed chunk
    embed_parser = subparsers.add_parser("embed_chunks", help="Embed the chunks")
    #embed_parser.add_argument("text", type=str, help="Text to chunk")  

    #embed chunk
    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search the embed the chunks")
    search_chunked_parser.add_argument("text", type=str, help="Text to search")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number to limit results")



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
        case "chunk":
            text= args.text 
            n = args.chunk_size
            overlap = args.overlap

            result = chunk_text(text, n, overlap)
            chunk_text_print_format(text, result)
        case "semantic_chunk":
            text= args.text
            n = args.max_chunk_size
            overlap = args.overlap 

            sem_split = re.split("(?<=[.!?])\s+", text)
            sem_split = [s.strip() for s in sem_split if s.strip()]
            result = sem_chunk_text(sem_split, n, overlap)
            sem_chunk_text_print_format(text, result)
        case "embed_chunks":
            with open("data/movies.json", "r", encoding="utf-8") as file:
                movie_data = json.load(file)

            movie_list = movie_data.get('movies', [])
            chunk_inst = ChunkedSemanticSearch()

            chunk_inst.load_or_create_chunk_embeddings(movie_list)

            print(f"Generated {len(chunk_inst.chunk_embeddings)} chunked embeddings")
        case "search_chunked":
            query= args.text
            limit_n = args.limit
            with open("data/movies.json", "r", encoding="utf-8") as file:
                movie_data = json.load(file)

            movie_list = movie_data.get('movies', [])
            chunk_inst = ChunkedSemanticSearch()
            chunk_inst.load_or_create_chunk_embeddings(movie_list)

            results = chunk_inst.search_chunks(query, limit_n)
            print(results)

        case _:
            parser.print_help()



if __name__ == "__main__":
    main()