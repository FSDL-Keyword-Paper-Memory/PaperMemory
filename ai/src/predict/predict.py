import logging
from datetime import datetime
from typing import List, Tuple, Union

from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer

NOW = datetime.now()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(f"logs/predictor_{NOW}.log"),
        logging.StreamHandler(),
    ],
)


class Predictor:
    def __init__(self, model: str) -> None:
        self.model_name = model

    def predict_keywords(
        self, abstracts: Union[str, List[str]], top_n: int
    ) -> Tuple[List[List[str]], List[List[float]]]:
        output_keywords = []
        output_scores = []
        logging.info(f"Loading model: {self.model_name}")
        model = KeyBERT(self.model_name)
        logging.info(f"Predicting keywords with model: {self.model_name}")
        keywords_scores = model.extract_keywords(
            docs=abstracts,
            stop_words="english",
            top_n=top_n,
            vectorizer=KeyphraseCountVectorizer(),
        )

        if isinstance(abstracts, str):
            keywords_scores = [keywords_scores]

        logging.info("Aggregating results")
        for result in keywords_scores:
            keywords = []
            scores = []
            for keyword, score in result:
                keywords.append(keyword)
                scores.append(score)
            output_keywords.append(keywords)
            output_scores.append(scores)

        return output_keywords, output_scores
