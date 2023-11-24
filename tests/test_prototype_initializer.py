"""nafes prototype initializers test suite"""


import torch
from torch.utils import data
import prototorch.core.initializers as pci
from nafes import PrototypeInitializers, get_prototype_initializers


def test_stratified_means_component_initializer():
    input_features = torch.Tensor(
        [
            [-6, -2, 0],
            [10, 11, 12],
            [0, 0, 0],
            [3, 2, 5],
        ]
    )
    labels = torch.LongTensor([0, 0, 1, 1])
    expected_prototypes = torch.Tensor(
        [
            [2.0, 4.5, 6.0],
            [1.5, 1.0, 2.5],
        ]
    )
    proto_initializer = get_prototype_initializers(
        initializer="SMCI",
        train_ds=data.TensorDataset(input_features, labels),
        pre_initialised_prototypes=None,
    ).generate({0: 1, 1: 1})

    assert torch.allclose(proto_initializer, expected_prototypes)


def test_stratified_selection_component_initializer():
    input_features = torch.Tensor(
        [
            [0, 0, 0],
            [10, 11, 12],
            [0, 0, 0],
            [10, 11, 12],
        ]
    )
    labels = torch.LongTensor([0, 1, 0, 1])
    expected_prototypes = torch.Tensor(
        [
            [0.0, 0.0, 0.0],
            [10, 11, 12],
        ]
    )
    proto_initializer = get_prototype_initializers(
        initializer="SSCI",
        train_ds=data.TensorDataset(input_features, labels),
        pre_initialised_prototypes=None,
    ).generate({0: 1, 1: 1})

    assert torch.allclose(proto_initializer, expected_prototypes)


def test_mean_component_initializer():
    input_features = torch.Tensor(
        [
            [-3, -2, 2],
            [10, 11, 12],
            [1, 0, 0],
            [10, 11, 12],
        ]
    )
    labels = torch.LongTensor([0, 1, 0, 1])
    expected_prototypes = torch.Tensor(
        [
            [4.5, 5.0, 6.5],
            [4.5, 5.0, 6.5],
        ]
    )
    proto_initializer = get_prototype_initializers(
        initializer=PrototypeInitializers.MEANCOMPONENTINITIALIZER.value,
        train_ds=data.TensorDataset(input_features, labels),
        pre_initialised_prototypes=None,
    ).generate(2)

    assert torch.allclose(proto_initializer, expected_prototypes)


def test_zero_component_initializer():
    input_features = torch.Tensor(
        [
            [-3, -2, 2],
            [10, 11, 12],
            [1, 0, 0],
            [10, 11, 12],
        ]
    )
    labels = torch.LongTensor([0, 1, 0, 1])
    expected_prototypes = torch.Tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    proto_initializer = get_prototype_initializers(
        initializer="ZCI",
        train_ds=data.TensorDataset(input_features, labels),
        pre_initialised_prototypes=None,
    ).generate(2)

    assert torch.allclose(proto_initializer, expected_prototypes)


def test_ones_component_initializer():
    input_features = torch.Tensor(
        [
            [-3, -2, 2],
            [10, 11, 12],
            [1, 0, 0],
            [10, 11, 12],
        ]
    )
    labels = torch.LongTensor([0, 1, 0, 1])
    expected_prototypes = torch.Tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    proto_initializer = get_prototype_initializers(
        initializer="OCI",
        train_ds=data.TensorDataset(input_features, labels),
        pre_initialised_prototypes=None,
    ).generate(2)

    assert torch.allclose(proto_initializer, expected_prototypes)


def test_random_normal_component_initializer():
    input_features = torch.Tensor(
        [
            [3, 2, 5],
            [3, 1, 5],
            [1, 4, 1],
            [1, 1, 1],
        ],
    )

    labels = torch.LongTensor([0, 1, 0, 1])
    proto_initializer = get_prototype_initializers(
        initializer="RNCI",
        train_ds=data.TensorDataset(input_features, labels),
        pre_initialised_prototypes=None,
    ).generate(2)

    assert proto_initializer.shape[1] == input_features.shape[1]


def test_uniform_component_initializer():
    input_features = torch.Tensor(
        [
            [3, 2, 5],
            [3, 1, 5],
            [1, 4, 1],
            [1, 1, 1],
        ],
    )

    labels = torch.LongTensor([0, 1, 0, 1])
    proto_initializer = get_prototype_initializers(
        initializer="UCI",
        train_ds=data.TensorDataset(input_features, labels),
        pre_initialised_prototypes=None,
    ).generate(2)

    assert proto_initializer.min() >= -1.0
    assert proto_initializer.max() <= 1.0


def test_literal_component_initializer():
    initial_prototypes = torch.Tensor(
        [
            [3, 2, 5],
            [3, 1, 5],
            [1, 4, 1],
            [1, 1, 1],
        ],
    )

    labels = torch.LongTensor([0, 1, 0, 1])
    proto_initializer = get_prototype_initializers(
        initializer="LCI",
        train_ds=data.TensorDataset(initial_prototypes, labels),
        pre_initialised_prototypes=initial_prototypes,
    ).generate(4)

    assert torch.allclose(proto_initializer, initial_prototypes)


def test_class_aware_component_initializer():
    initial_prototypes = torch.Tensor(
        [
            [3, 2, 5],
            [3, 1, 5],
            [1, 4, 1],
            [1, 1, 1],
        ],
    )

    Prototype_labels = torch.LongTensor([0, 0, 1, 1])
    proto_initializer = get_prototype_initializers(
        initializer="CACI",
        train_ds=data.TensorDataset(initial_prototypes, Prototype_labels),
        pre_initialised_prototypes=None,
    ).generate(4)

    assert torch.allclose(proto_initializer, initial_prototypes)


def test_data_aware_component_initializer():
    initial_prototypes = torch.Tensor(
        [
            [3, 2, 5],
            [3, 1, 5],
            [1, 4, 1],
            [1, 1, 1],
        ],
    )
    Prototype_labels = torch.LongTensor([0, 0, 1, 1])
    proto_initializer = get_prototype_initializers(
        initializer="DACI",
        train_ds=data.TensorDataset(initial_prototypes, Prototype_labels),
        pre_initialised_prototypes=None,
    ).generate(4)

    assert torch.allclose(proto_initializer, initial_prototypes)


def test_selection_component_initializer():
    input_features = torch.Tensor(
        [
            [3, 2, 5],
            [3, 1, 5],
            [1, 4, 1],
            [1, 1, 1],
        ],
    )
    input_labels = torch.LongTensor([0, 0, 1, 1])
    proto_initializer = get_prototype_initializers(
        initializer="SCI",
        train_ds=data.TensorDataset(input_features, input_labels),
        pre_initialised_prototypes=None,
    ).generate(4)

    assert proto_initializer.shape[1] == input_features.shape[1]
    assert len(proto_initializer) == 4


def test_Fill_value_component_initializer():
    initial_prototypes = torch.Tensor(
        [
            [3, 2, 5],
            [3, 1, 5],
            [1, 4, 1],
            [1, 1, 1],
        ],
    )
    Prototype_labels = torch.LongTensor([0, 0, 1, 1])
    proto_initializer = get_prototype_initializers(
        initializer="DACI",
        train_ds=data.TensorDataset(initial_prototypes, Prototype_labels),
        pre_initialised_prototypes=None,
        fill_value=1.0,
    ).generate(4)

    assert torch.allclose(proto_initializer, initial_prototypes)

