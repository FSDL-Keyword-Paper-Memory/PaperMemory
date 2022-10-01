import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm


def setup_paths(path: str) -> Tuple[Path]:
    data_folder = Path(path)
    keys_folder = data_folder / "keys"
    papers_folder = data_folder / "docsutf8"

    papers_paths = sorted(papers_folder.iterdir())
    keys_paths = sorted(keys_folder.iterdir())

    save_to = data_folder / ".." / "devset.json"

    return papers_paths, keys_paths, save_to


def read_keywords(keys_path: Path) -> List[str]:
    keys = pd.read_csv(keys_path, header=None).squeeze("columns")
    keys_list = keys.to_list()

    return keys_list


def read_abstract(paper_path: Path) -> str:
    with open(paper_path) as f:
        lines = f.readlines()

    indices = []
    for i, line in enumerate(lines):
        if "--A".casefold() in line.casefold():
            indices.append(i)
        if "--B".casefold() in line.casefold():
            indices.append(i)
            break

    abstract = " ".join(lines[indices[0] + 1 : indices[-1]]).strip("\n")

    return abstract


def filter_keywords_in_abstract(abstract: str, keywords: List[str]) -> List[str]:
    filtered_keywords = [keyword for keyword in keywords if keyword in abstract]

    return filtered_keywords


def append_doc(
    df: pd.DataFrame, abstract: str, keywords: List[str], index: str
) -> pd.DataFrame:
    row = pd.DataFrame(
        [
            {
                "abstract": abstract,
                "keywords": keywords,
            }
        ],
        index=[index],
    )
    df = pd.concat([df, row], axis=0)

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()
    papers_paths, keys_paths, save_to = setup_paths(args.path)

    df = pd.DataFrame()
    for paper_path, keys_path in tqdm(zip(papers_paths, keys_paths)):
        keys_list = read_keywords(keys_path)
        abstract = read_abstract(paper_path)
        keys_list_filtered = filter_keywords_in_abstract(abstract, keys_list)
        if not keys_list_filtered:
            continue

        df = append_doc(df, abstract, keys_list, paper_path.stem)

    df.to_json(save_to, orient="records")


if __name__ == "__main__":
    main()
