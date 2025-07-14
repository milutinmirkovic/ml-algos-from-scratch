from pandas.api.types import is_numeric_dtype
import pandas as pd
from typing import List



def _is_continuous(column:pd.Series, threshold:int = 3) -> bool:
    if is_numeric_dtype(column) and column.nunique()> threshold:
        return True
    return False 

class Feature:
    def __init__(self, column:pd.Series, threshold:int = 3):
        self.name = column.name
        self.is_continuous = _is_continuous(column,threshold)
        self.data = column


def get_features(data:pd.DataFrame)->List[Feature]:

    features = []
    for feature in data.columns:
        features.append(Feature(data[feature]))

    return features


