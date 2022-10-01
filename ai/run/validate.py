import argparse
import logging
import os
from datetime import datetime

import pandas as pd
from src.preprocessing.cleaner import AbstractCleaner
from src.validate.validate import Validator

NOW = datetime.now()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(f"logs/validate_{NOW}.log"),
        logging.StreamHandler(),
    ],
)


MODELS = [
    "allenai-specter",
    "average_word_embeddings_glove.6B.300d",
    "average_word_embeddings_komninos",
    "paraphrase-MiniLM-L3-v2",
    "paraphrase-MiniLM-L6-v2",
    "all-MiniLM-L6-v1",
    "all-MiniLM-L6-v2",
    "paraphrase-MiniLM-L12-v2",
    "paraphrase-albert-small-v2",
    "paraphrase-TinyBERT-L6-v2",
    "paraphrase-distilroberta-base-v2",
    "all-roberta-large-v1",
    "all-mpnet-base-v1",
    "all-mpnet-base-v2",
    "all-MiniLM-L12-v1",
    "all-distilroberta-v1",
    "all-MiniLM-L12-v2",
]


def clean_data_if_needed(path: str):
    if not "cleaned.json" in path:
        clean_abstracts_path = path.split(".")[0] + "_cleaned.txt"
        clean_devset_path = path.split(".")[0] + "_cleaned.json"

        abstract_cleaner = AbstractCleaner(path)
        abstract_cleaner.clean()

        devset = pd.read_json(path, orient="records")
        clean_abstract = pd.read_table(clean_abstracts_path, header=None)
        os.remove(clean_abstracts_path)
        devset["abstract"] = clean_abstract
        devset.to_json(clean_devset_path, orient="records")
    else:
        clean_devset_path = path

    return clean_devset_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()
    logging.info("Loading dataset. Cleaning if needed.")
    clean_devset_path = clean_data_if_needed(args.path)

    for model in MODELS:
        validator = Validator(model, clean_devset_path)
        score = validator.validate()
        logging.info(f"Model: {model}, score: {score}")


if __name__ == "__main__":
    main()
