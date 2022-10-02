import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from src.predict.predict import Predictor

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

NOW = datetime.now()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(f"logs/validator_{NOW}.log"),
        logging.StreamHandler(),
    ],
    force=True,
)


class Validator:
    def __init__(self, model: str, filepath: str) -> None:
        self.filepath = filepath
        self.predictor = Predictor(model)

    def validate(
        self, thresholds: Union[float, List[float]], save_predicted: bool = True
    ) -> pd.DataFrame:
        logging.info("Reading dataset to validate")
        df = self._read_devset()
        max_keywords_num = self._get_max_keywords_num(df)
        start_time = time.perf_counter()
        keywords_predicted, scores = self.predictor.predict_keywords(
            df["abstract"].to_list(), top_n=max_keywords_num, threshold=0
        )
        logging.info(
            f"Predict time per abstract: {int((time.perf_counter() - start_time)*1000) / df.shape[0]} ms"
        )
        df["keywords_predicted"], df["scores"] = keywords_predicted, scores
        logging.info("Calculating scores")
        results = self._get_metrics_for_thresholds(df, thresholds, save_predicted)

        return results

    def _get_metrics_for_thresholds(
        self,
        df: pd.DataFrame,
        thresholds: Union[float, List[float]],
        save_predicted: bool = True,
    ) -> pd.DataFrame:
        results = pd.DataFrame()

        if isinstance(thresholds, float):
            thresholds = [thresholds]

        for threshold in thresholds:
            dff = df.copy()
            dff[["keywords_predicted", "scores"]] = dff.apply(
                self._adjust_predicted_keywords_num, args=(threshold,), axis=1
            ).to_list()
            if save_predicted:
                self._save_to_json(dff, self.predictor.model_name, threshold)

            precision, recall = self._calculate_score(dff)
            result = pd.DataFrame(
                {
                    "model": [self.predictor.model_name],
                    "threshold": [threshold],
                    "precision": [precision],
                    "recall": [recall],
                }
            )
            results = pd.concat([results, result], axis=0, ignore_index=True)

        return results

    @staticmethod
    def _get_max_keywords_num(df: pd.DataFrame) -> int:
        return df.keywords.str.len().max()

    def _read_devset(self) -> pd.DataFrame:
        df = pd.read_json(self.filepath, orient="records")

        return df

    @staticmethod
    def _save_to_json(df: pd.DataFrame, model_name: str, threshold: float) -> None:
        filepath_str = f"data/validated/validated_devset_{model_name}_{threshold}.json"
        filepath = Path(filepath_str)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(filepath, orient="records")

    @staticmethod
    def _calculate_score(df: pd.DataFrame) -> float:
        scores = []
        score = 0
        found_keywords = 0
        for _, row in df.iterrows():
            counter = 0
            if row["keywords_predicted"].size == 0:
                continue
            found_keywords += 1
            for keyword_predicted in row["keywords_predicted"]:
                for keyword in row["keywords"]:
                    if (
                        keyword.lower() in keyword_predicted.lower()
                        or keyword_predicted.lower() in keyword.lower()
                    ):
                        counter += 1
                    break
            score = counter / len(row["keywords_predicted"])
            if score:
                scores.append(score)

        precision = np.mean(scores) if scores else 0
        recall = found_keywords / df.shape[0]

        return precision, recall

    @staticmethod
    def _adjust_predicted_keywords_num(
        row: pd.Series, threshold: float
    ) -> Tuple[List[str], List[float]]:
        mask = np.array(row["scores"]) > threshold
        predicted_keywords = np.array(row["keywords_predicted"])[mask]
        scores = np.array(row["scores"])[mask]

        return predicted_keywords, scores
