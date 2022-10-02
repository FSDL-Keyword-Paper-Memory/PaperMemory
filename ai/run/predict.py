import argparse

from src.predict.predict import Predictor
from src.preprocessing.cleaner import AbstractCleaner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--abstract", type=str, required=True)
    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()

    cleaner = AbstractCleaner()
    clean_abstract = cleaner.perform_cleaning(args.abstract)

    predictor = Predictor(args.model)
    keywords, _ = predictor.predict_keywords(clean_abstract, 10, args.threshold)

    return keywords


if __name__ == "__main__":
    result = main()
    print(result)
