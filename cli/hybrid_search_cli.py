import argparse
import sys
from lib.hybrid_search import (
    normalize_scores,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    #normalize command
    normalize_parser = subparser.add_parser("normalize", help="Normalize the scores")
    normalize_parser.add_argument("nums", type=float, nargs="*", help="Enter the scores to be normalized")
    
    args = parser.parse_args()

    match args.command:
        case "normalize":
            nums = args.nums
            normalized = normalize_scores(nums)
            for score in normalized:
                print(f"* {score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()