"""nafes datasets"""

from enum import Enum
import random

import numpy as np
from dataclasses import dataclass
import torch
from sklearn.datasets import load_breast_cancer, make_moons, make_blobs
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10


class Sampling(str, Enum):
    RANDOM = "random"
    FULL = "full"


@dataclass(slots=True)
class RandomInputs:
    random_inputs: torch.Tensor
    random_labels: torch.Tensor


@dataclass(slots=True)
class TensorSet:
    input_date: torch.Tensor
    labels: torch.Tensor


@dataclass(slots=True)
class DATASET:
    input_data: np.ndarray
    labels: np.ndarray


def breast_cancer_dataset() -> DATASET:
    data, labels = load_breast_cancer(return_X_y=True)
    return DATASET(data, labels)


def moons_dataset(random_state: int = None) -> DATASET:
    data, labels = make_moons(
        n_samples=150, shuffle=True, noise=None, random_state=random_state
    )
    return DATASET(data, labels)


def bloobs(random_state: int = None) -> DATASET:
    data, labels = make_blobs(
        n_samples=[120, 80],
        centers=[[0.0, 0.0], [2.0, 2.0]],
        cluster_std=[1.2, 0.5],
        random_state=random_state,
        shuffle=False,
    )
    return DATASET(data, labels)


def mnist_dataset() -> TensorSet:
    train_dataset = MNIST(
        "~/datasets",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )
    test_dataset = MNIST(
        "~/datasets",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        ),
    )
    return TensorSet(
        torch.cat([train_dataset.data, test_dataset.data]),
        torch.cat([train_dataset.targets, test_dataset.targets]),
    )


def cifar_10(sample: Sampling, size: int) -> TensorSet:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    full_train_ds = torch.cat(
        [torch.from_numpy(train_dataset.data), torch.from_numpy(test_dataset.data)]
    )
    full_train_labels = torch.cat(
        [
            torch.from_numpy(np.array(train_dataset.targets)),
            torch.from_numpy(np.array(test_dataset.targets)),
        ]
    )
    classwise_labels = get_classwise_labels(full_train_labels)
    samples = get_random_inputs(
        full_train_ds, full_train_labels, classwise_labels, sample_size=size
    )

    match sample:
        case Sampling.FULL:
            return TensorSet(full_train_ds, full_train_labels)
        case Sampling.RANDOM:
            return TensorSet(samples.random_inputs, samples.random_labels)
        case _:
            raise RuntimeError("cifar-10:none of the cases match")


def get_classwise_labels(full_labels: torch.Tensor, num_class: int = 10) -> np.ndarray:
    classwise_labels = []
    for class_label in range(num_class):
        for index, label in enumerate(full_labels):
            label = label.detach().cpu().numpy()
            if label == class_label:
                classwise_labels.append(index)
    return np.reshape(classwise_labels, (-1, 6000))


def get_random_inputs(
    full_train_ds: torch.Tensor,
    full_train_labels: torch.Tensor,
    classwise_labels: np.ndarray,
    sample_size: int = 1000,
) -> RandomInputs:
    random_labels = []
    for class_ in classwise_labels:
        random.shuffle(class_)
        random_labels.append(class_[:sample_size])
    random_label_indices = np.array(random_labels)
    random_label_indices = random_label_indices.flatten()

    return RandomInputs(
        torch.from_numpy(
            np.array(
                [full_train_ds[index] for index in random_label_indices],
                dtype=float,
            )
        ),
        torch.from_numpy(
            np.array(
                [full_train_labels[index] for index in random_label_indices],
                dtype=float,
            )
        ),
    )


@dataclass(slots=True)
class DATA:
    sample: Sampling = Sampling.FULL
    random: int = 4
    sample_size: int = 1000

    @property
    def S_1(self) -> DATASET:
        return moons_dataset(self.random)

    @property
    def S_2(self) -> DATASET:
        return bloobs(self.random)

    @property
    def breast_cancer(self) -> DATASET:
        return breast_cancer_dataset()

    @property
    def mnist(self) -> TensorSet:
        return mnist_dataset()

    @property
    def cifar_10(self) -> TensorSet:
        return cifar_10(self.sample, self.sample_size)


# if __name__ == "__main__":
#     data = DATA(sample=Sampling.RANDOM, sample_size=10)
#     print(data.mnist.input_date)
