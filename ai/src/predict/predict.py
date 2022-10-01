from typing import List, Tuple, Union

from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer


class Predictor:
    def __init__(self, model: str) -> None:
        self.model_name = model

    def predict_keywords(
        self, abstracts: Union[str, List[str]], top_n: int
    ) -> Tuple[List[List[str]], List[List[float]]]:
        output_keywords = []
        output_scores = []
        model = KeyBERT(self.model_name)
        keywords_scores = model.extract_keywords(
            docs=abstracts,
            stop_words="english",
            top_n=top_n,
            vectorizer=KeyphraseCountVectorizer(),
        )

        if isinstance(abstracts, str):
            keywords_scores = [keywords_scores]

        for result in keywords_scores:
            keywords = []
            scores = []
            for keyword, score in result:
                keywords.append(keyword)
                scores.append(score)
            output_keywords.append(keywords)
            output_scores.append(scores)

        return output_keywords, output_scores
