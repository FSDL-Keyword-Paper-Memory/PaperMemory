import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from src.predict.predict import Predictor

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class Validator:
    def __init__(self, model: str, filepath: str) -> None:
        self.filepath = filepath
        self.predictor = Predictor(model)

    def validate(self) -> float:
        df = self._read_devset()
        max_keywords_num = self._get_max_keywords_num(df)
        keywords_predicted, scores = self.predictor.predict_keywords(
            df["abstract"].to_list(), top_n=max_keywords_num
        )
        df["keywords_predicted"], df["scores"] = keywords_predicted, scores
        df[["keywords_predicted", "scores"]] = df.apply(
            self._adjust_predicted_keywords_num, axis=1
        ).to_list()
        score = self._calculate_score(df)

        return score

    @staticmethod
    def _get_max_keywords_num(df: pd.DataFrame) -> int:
        return df.keywords.str.len().max()

    def _read_devset(self) -> pd.DataFrame:
        df = pd.read_json(self.filepath, orient="records")

        return df

    @staticmethod
    def _calculate_score(df: pd.DataFrame) -> float:
        scores = []
        for _, row in df.iterrows():
            print(row["keywords"])
            print(row["keywords_predicted"])
            counter = 0
            for keyword in row["keywords"]:
                for keyword_predicted in row["keywords_predicted"]:
                    if keyword.lower() in keyword_predicted.lower():
                        counter += 1
                        break
            score = counter / len(row["keywords"])
            scores.append(score)
        print(scores)
        return np.mean(scores)

    @staticmethod
    def _adjust_predicted_keywords_num(row: pd.Series) -> Tuple[List[str], List[float]]:
        num_keywords = len(row["keywords"])
        predicted_keywords = row["keywords_predicted"][:num_keywords]
        scores = row["scores"][:num_keywords]

        return predicted_keywords, scores
