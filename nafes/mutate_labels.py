"""implementation of Label Mutation Algorithm"""

from collections import Counter
from enum import Enum
from dataclasses import dataclass
import random

import numpy as np
from typing import List

random.seed(4)
np.random.seed(4)


class MutationType(str, Enum):
    SWAPNEXTLABEL = "next"
    SWAPLABEL = "unique"


class PerturbationDistribution(str, Enum):
    GLOBAL = "global"
    UNIFORMCLASSAWARE = "uniform"
    BALANCEDCLASSAWARE = "balanced"


@dataclass
class TargetsInfo:
    unique_labels: np.ndarray
    mutation_value: int
    label_indices: np.ndarray


@dataclass
class MutatedValidationSet:
    dataset: np.ndarray
    mutated_labels: List[int]
    original_labels: List[int]


def get_target_info(labels: np.ndarray, perturbation_ratio: float) -> TargetsInfo:
    return TargetsInfo(
        unique_labels=np.unique(labels),
        mutation_value=int(np.ceil(perturbation_ratio * len(labels))),
        label_indices=np.arange(len(labels)),
    )


def mutate_targets_randomly(
        labels: np.ndarray,
        random_label_indices: List[int],
        swapped_labels: List[int]
):
    for index_mutated, mutated_labels in zip(random_label_indices, swapped_labels):
        for index_original, _original_labels in enumerate(labels):
            if index_original == index_mutated:
                labels[index_original] = mutated_labels


def mutate_targets_next_in_list(labels: np.ndarray, random_label_indices: List[int]):
    last_label_index = np.arange(len(labels))[-1]
    list_indices_for_mutation = [
        index + 1 if index != last_label_index else 0 for index in random_label_indices
    ]

    for index in list_indices_for_mutation:
        if index != 0:
            labels[index - 1] = labels[index]
        else:
            labels[last_label_index] = labels[index]


def mutate_targets(
        perturbation_type: str,
        labels: np.ndarray,
        random_label_indices: List[int],
        swapped_labels: List[int],
) -> None:
    if perturbation_type == MutationType.SWAPLABEL:
        mutate_targets_randomly(labels, random_label_indices, swapped_labels)
    if perturbation_type == MutationType.SWAPNEXTLABEL:
        mutate_targets_next_in_list(labels, random_label_indices)
    # raise RuntimeError("mutate_targets: none of the checks did match")


def get_random_label_indices_list(
        cardinality_of_target_space: List[int],
        sorted_indices_targets: List[int],
        class_aware_mutation_ratio: List[int],
) -> List[int]:
    count, random_label_indices_list = 0, []
    for index, size in enumerate(cardinality_of_target_space):
        sorted_label_indices = sorted_indices_targets[count: count + size]
        random_label_indices = random.sample(
            sorted(sorted_label_indices), class_aware_mutation_ratio[index]
        )
        count += size
        random_label_indices_list.append(random_label_indices)
    return sum(random_label_indices_list, [])


def get_class_ratio(
        cardinality_of_target_space: List[int],
        target_information: TargetsInfo,
        perturbation_distribution: str,
) -> List[int]:
    if perturbation_distribution == PerturbationDistribution.BALANCEDCLASSAWARE:
        return [
            round(
                (
                        weight
                        / np.sum(cardinality_of_target_space)
                        * target_information.mutation_value
                )
            )
            for weight in cardinality_of_target_space
        ]
    if perturbation_distribution == PerturbationDistribution.UNIFORMCLASSAWARE:
        return [
            round(
                (
                        target_information.mutation_value
                        / len(cardinality_of_target_space)
                )
            )
            for _weight in cardinality_of_target_space
        ]
    raise RuntimeError("get_class_ratio: none of the checks did match")


@dataclass
class MutatedValidation:
    labels: np.ndarray
    perturbation_ratio: float
    perturbation_distribution: str
    perturbation_type: str = MutationType.SWAPLABEL

    @property
    def __mutated_targets_list(
            self,
    ) -> np.ndarray:
        target_information: TargetsInfo = get_target_info(
            self.labels, self.perturbation_ratio
        )
        random_label_indices: List[int] = random.sample(
            sorted(target_information.label_indices), target_information.mutation_value
        )
        random_instances_labels: List[int] = [
            self.labels[index] for index in random_label_indices
        ]

        swapped_labels: List[int] = [
            np.random.choice(
                [
                    unique_label
                    for unique_label in target_information.unique_labels
                    if unique_label != label
                ]
            )
            for index, label in enumerate(random_instances_labels)
        ]
        original_labels = self.labels.copy()
        mutate_targets(
            self.perturbation_type,
            original_labels,
            random_label_indices,
            swapped_labels,
        )
        return original_labels

    def __mutated_targets_class_level_balanced(
            self, perturbation_distribution: str
    ) -> np.ndarray:
        target_information: TargetsInfo = get_target_info(
            self.labels, self.perturbation_ratio
        )

        sorted_indices_targets = sorted(
            range(len(self.labels)), key=lambda index: self.labels[index]
        )
        sorted_targets = [self.labels[index] for index in sorted_indices_targets]
        cardinality_of_target_space = list(Counter(sorted_targets).values())

        class_aware_mutation_ratio = get_class_ratio(
            cardinality_of_target_space, target_information, perturbation_distribution
        )

        random_label_indices_merged = get_random_label_indices_list(
            cardinality_of_target_space,
            sorted_indices_targets,
            class_aware_mutation_ratio,
        )

        random_instances_labels: List[int] = [
            self.labels[index] for index in random_label_indices_merged
        ]

        swapped_labels: List[int] = [
            np.random.choice(
                [
                    unique_label
                    for unique_label in target_information.unique_labels
                    if unique_label != label
                ]
            )
            for index, label in enumerate(random_instances_labels)
        ]
        original_labels = self.labels.copy()
        mutate_targets(
            self.perturbation_type,
            original_labels,
            random_label_indices_merged,
            swapped_labels,
        )

        return original_labels

    @property
    def get_mutated_label_list(self):
        if self.perturbation_distribution == PerturbationDistribution.GLOBAL:
            return self.__mutated_targets_list
        if self.perturbation_distribution == PerturbationDistribution.BALANCEDCLASSAWARE:
            return self.__mutated_targets_class_level_balanced(
                PerturbationDistribution.BALANCEDCLASSAWARE
            )
        if self.perturbation_distribution == PerturbationDistribution.UNIFORMCLASSAWARE:
            return self.__mutated_targets_class_level_balanced(
                PerturbationDistribution.UNIFORMCLASSAWARE
            )
        raise RuntimeError(
            "get_mutated_label_list: none of the checks did match"
        )
