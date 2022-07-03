import pandas as pd
import numpy as np

from typing import List, Tuple
from random import sample

from knn import KNNInstance, KNNModel
from distance_algorithms import euclidean_distance


def normalize_dataframe(dataframe):
    for column in dataframe.columns[1:-1]:
        dataframe[column] = np.divide(
            dataframe[column].values,
            np.max(dataframe[column].values)
        )


def KNNInstances_from_file(
    file_path: str,
    validation_ratio: float,
    normalize: bool
) -> Tuple[List[KNNInstance], List[KNNInstance]]:

    dataframe = pd.read_csv(file_path)
    if(normalize):
        normalize_dataframe(dataframe)

    group_indexes = [
        group.index.values.tolist() for group in [
            dataframe[dataframe['target'] == 0],
            dataframe[dataframe['target'] == 1]
        ]
    ]

    sample_indexes = []
    for group in group_indexes:
        sample_indexes += sample(group, k=int(len(group) * validation_ratio))

    training_instances: List[KNNInstance] = []
    validation_instances: List[KNNInstance] = []

    for index, row in dataframe.iterrows():
        new_instance = KNNInstance(
            row.values[1:31].tolist(),  # attributes
            row.values[31].tolist()     # label
        )
        if (index in sample_indexes):
            validation_instances.append(new_instance)
        else:
            training_instances.append(new_instance)

    return training_instances, validation_instances


def main():
    # usage example, compare with and without normalization

    training, validation = KNNInstances_from_file(
        'breast_cancer_data.csv', 0.2, False
    )
    model = KNNModel(training)
    print(model.calculate_accuracy(validation, 11, euclidean_distance))

    training, validation = KNNInstances_from_file(
        'breast_cancer_data.csv', 0.2, True
    )
    model = KNNModel(training)
    print(model.calculate_accuracy(validation, 11, euclidean_distance))


if __name__ == "__main__":
    main()
