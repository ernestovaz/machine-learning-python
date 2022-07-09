from typing import List
from math import sqrt


def euclidean_distance(i: List[float], j: List[float]) -> float:
    individual_distances: List[float] = [(x - y)**2
                                         for x, y in zip(i, j)]
    return sqrt(sum(individual_distances))


def manhattan_distance(i: List[float], j: List[float]) -> float:
    individual_distances: List[float] = [abs(x - y)
                                         for x, y in zip(i, j)]
    return sum(individual_distances)


def hamming_distance(i: List[float], j: List[float]) -> float:
    individual_distances: List[float] = [(x != y)
                                         for x, y in zip(i, j)]
    return sum(individual_distances)


def matching_distance(i: List[float], j: List[float]) -> float:
    return hamming_distance(i, j) / len(i)


def main():
    # usage example
    print(euclidean_distance([0.5, 0.43], [1.00, 0.57]))
    print(manhattan_distance([0.5, 0.43], [1.00, 0.57]))
    print(hamming_distance([0, 1], [1, 1]))
    print(matching_distance([0, 1], [1, 1]))


if __name__ == "__main__":
    main()
