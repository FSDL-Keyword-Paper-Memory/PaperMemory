import re
from pathlib import Path

import pandas as pd


class AbstractCleaner:
    def __init__(self, filepath: str) -> None:
        self.filepath = Path(filepath)

    def clean(self) -> None:
        save_to = self.filepath.parent / (self.filepath.stem + "_cleaned.json")

        df = self._load_dataset()
        df = self._drop_duplicates(df)
        df = self._remove_abstracts_for_withdrawn_papers(df)
        df["abstract_clean"] = df.abstract.apply(self._perform_cleaning)
        df.to_json(save_to)

    def _load_dataset(self) -> pd.DataFrame:
        return pd.read_json(self.filepath)

    def _perform_cleaning(self, text: str) -> str:
        text = self._replace_newline_character_with_whitespace(text)
        text = self._remove_redundant_escapes(text)
        text = self._remove_latex_suffixes_prefixes(text)
        text = self._remove_multiple_whitespaces(text)
        text = self._remove_urls(text)

        return text

    @staticmethod
    def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(subset=["abstract"], keep="first")

    @staticmethod
    def _remove_abstracts_for_withdrawn_papers(df: pd.DataFrame) -> pd.DataFrame:
        pattern = r"(paper has been withdrawn)|(withdrawn due to)"
        return df[~df.abstract.str.contains(pattern, case=False, regex=True)]

    @staticmethod
    def _remove_redundant_escapes(text: str) -> str:
        return re.sub(r"\\+", r"\\", text)

    @staticmethod
    def _remove_latex_suffixes_prefixes(text: str) -> str:
        text = re.sub(
            r"\$([a-z])\$", r"\1", text, flags=re.IGNORECASE | re.DOTALL
        )  # clean single letters
        text = re.sub(
            r"\$\\(\w+)\s*\$", r"\1", text, flags=re.IGNORECASE | re.DOTALL
        )  # clean symbols
        return re.sub(
            r"\$.*?\$", "equation", text, flags=re.IGNORECASE | re.DOTALL
        )  # clean equations

    @staticmethod
    def _replace_newline_character_with_whitespace(text: str) -> str:
        return re.sub(r"\\n", " ", text)

    @staticmethod
    def _remove_multiple_whitespaces(text: str) -> str:
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def _remove_urls(text: str) -> str:
        return re.sub(
            r"((https?)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*",
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
