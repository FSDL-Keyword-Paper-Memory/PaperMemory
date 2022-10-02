import argparse
import os

import numpy as np
import pandas as pd
from src.preprocessing.cleaner import AbstractCleaner
from src.validate.validate import Validator

MODELS = [
    "allenai-specter",
    "all-MiniLM-L6-v1",
    "all-MiniLM-L6-v2",
    "paraphrase-distilroberta-base-v2",
    "all-roberta-large-v1",
    "all-mpnet-base-v1",
    "all-mpnet-base-v2",
    "all-MiniLM-L12-v1",
    "all-distilroberta-v1",
]

THRESHOLDS = np.arange(1, 0.3, -0.01)


def clean_data_if_needed(path: str):
    if not "cleaned.json" in path:
        clean_abstracts_path = path.split(".")[0] + "_cleaned.txt"
        clean_devset_path = path.split(".")[0] + "_cleaned.json"

        abstract_cleaner = AbstractCleaner()
        abstract_cleaner.clean(path)

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
    clean_devset_path = clean_data_if_needed(args.path)
    save_output_to = (
        "/".join(clean_devset_path.split("/")[:-1]) + "/validation_results.csv"
    )

    output = pd.DataFrame()
    for model in MODELS:
        validator = Validator(model, clean_devset_path)
        results = validator.validate(THRESHOLDS)
        output = pd.concat([output, results], axis=0, ignore_index=True)

    output.to_csv(save_output_to, index=False)


if __name__ == "__main__":
    main()
