from typing import List, Callable
from statistics import mode
from distance_algorithms import euclidean_distance


class KNNInstance():
    def __init__(self, attributes: List[float], label: float):
        self.attributes = attributes
        self.label = label


class KNNModel():
    def __init__(self, training_instances: List[KNNInstance]):
        self.training_instances = training_instances

    # calculates predicted label
    def predict(
        self,
        instance: KNNInstance,
        k: int,  # number of closest neighbours
        distance_function: Callable[[List[float], List[float]], float]
    ) -> float:

        sorted_instances: List[KNNInstance] = sorted(
            self.training_instances, key=lambda training_instance:
            distance_function(instance.attributes,
                              training_instance.attributes))

        closest_labels: List[float] = [i.label for i in sorted_instances[0:k]]
        predicted_label: float = mode(closest_labels)
        return predicted_label


def main():
    # usage example
    training_instances: List[KNNInstance] = [
        KNNInstance([0.20, 0.60], 0.0),
        KNNInstance([0.90, 0.70], 1.0),
        KNNInstance([0.61, 0.40], 1.0),
        KNNInstance([0.61, 0.20], 0.0),
        KNNInstance([0.40, 0.50], 1.0),
        KNNInstance([0.60, 1.00], 1.0),
        KNNInstance([0.50, 0.10], 0.0),
        KNNInstance([0.70, 0.40], 1.0),
        KNNInstance([0.80, 0.30], 1.0),
        KNNInstance([0.40, 0.30], 0.0),
        KNNInstance([0.60, 0.43], 0.0),
        KNNInstance([1.00, 0.20], 0.0)
    ]
    model: KNNModel = KNNModel(training_instances)
    new_instance = KNNInstance([0.3, 0.7], None)
    print(model.predict(new_instance, 3, euclidean_distance))
    print(model.predict(new_instance, 7, euclidean_distance))


if __name__ == "__main__":
    main()
