from typing import List, Callable
from statistics import mode

from algorithms.distance import euclidean_distance
from instance import Instance


class Model():
    def __init__(self, training_instances: List[Instance]):
        self.training_instances = training_instances

    # calculates predicted label
    def predict(
        self,
        instance: Instance,
        k: int,  # number of closest neighbours
        distance_function: Callable[[List[float], List[float]], float]
    ) -> float:

        sorted_instances: List[Instance] = sorted(
            self.training_instances, key=lambda training_instance:
            distance_function(instance.attributes,
                              training_instance.attributes))

        closest_labels: List[float] = [i.label for i in sorted_instances[0:k]]
        predicted_label: float = mode(closest_labels)
        return predicted_label

    def calculate_accuracy(
        self,
        validation_instances: List[Instance],
        k: int,
        distance_function: Callable[[List[float], List[float]], float]
    ) -> float:

        correct_predictions: List[int] = [
            self.predict(instance, k, distance_function) == instance.label
            for instance in validation_instances
        ]

        return sum(correct_predictions) / len(validation_instances)


def main():
    # usage example
    training_instances: List[Instance] = [
        Instance([0.20, 0.60], 0.0),
        Instance([0.90, 0.70], 1.0),
        Instance([0.61, 0.40], 1.0),
        Instance([0.61, 0.20], 0.0),
        Instance([0.40, 0.50], 1.0),
        Instance([0.60, 1.00], 1.0),
        Instance([0.50, 0.10], 0.0),
        Instance([0.70, 0.40], 1.0),
        Instance([0.80, 0.30], 1.0),
        Instance([0.40, 0.30], 0.0),
        Instance([0.60, 0.43], 0.0),
        Instance([1.00, 0.20], 0.0)
    ]
    model: Model = Model(training_instances)
    new_instance = Instance([0.3, 0.7], None)
    print(model.predict(new_instance, 3, euclidean_distance))
    print(model.predict(new_instance, 7, euclidean_distance))


if __name__ == "__main__":
    main()
