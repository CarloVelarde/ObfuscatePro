from abc import ABC, abstractmethod
from typing import Union

import pandas as pd

from table import Table
from utility.CommonUtility import CommonUtility


class BaseModel:

    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name

    @abstractmethod
    def obfuscate_code(self, code: str, language: str, obfuscation_method: str) -> str:
        pass

    @abstractmethod
    def embed_code(self, code: Union[str, list], dimensionality: int):
        pass

    def similarity(self, emb1: list, emb2: list, algo_type=None, round_to=None):
        similarity = 0

        if algo_type is None:
            algo_type = "COSINE_SIMILARITY"

        if algo_type == "COSINE_SIMILARITY":
            similarity = CommonUtility.calculate_cosine_similarity(emb1, emb2)

        elif algo_type == "EUCLIDEAN_DISTANCE":
            similarity = CommonUtility.calculate_euclidean_distance(emb1, emb2)

        if round_to is not None:
            similarity = round(similarity, round_to)

        return similarity

    def create_table(self):
        return Table(pd.DataFrame())
