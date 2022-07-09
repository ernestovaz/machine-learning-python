import pandas as pd
import numpy as np

from typing import Tuple


def normalize(dataframe: pd.dataframe) -> None:
    for column in dataframe.columns[1:-1]:
        dataframe[column] = np.divide(
            dataframe[column].values,
            np.max(dataframe[column].values)
        )


def split_train_test_instances(instanceList):

    group_indexes = [
        group.index.values.tolist() for group in [
            dataframe[dataframe['target'] == 0],
            dataframe[dataframe['target'] == 1]
        ]
    ]
    for instance in instance_list:
