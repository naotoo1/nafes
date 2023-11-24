"""nafes omega initializers test suite"""


import torch
import prototorch as pt
import prototorch.core.initializers as pci


def test_zeros_linear_initializer():
    omega_initializer = pt.transforms.LinearTransform(
        in_dim=2,
        out_dim=4,
        initializer=pt.initializers.ZerosLinearTransformInitializer(
            out_dim_first=False
        ),
    ).weights
    expected_initialization = torch.Tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    assert torch.allclose(omega_initializer, expected_initialization)


def test_zeros_linear_initializer11():
    omega_initializer = pt.transforms.LinearTransform(
        in_dim=2,
        out_dim=4,
        initializer=pt.initializers.ZerosLinearTransformInitializer(
            out_dim_first=True,
        ),
    ).weights
    expected_initialization = torch.Tensor(
        [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ]
    )
    assert torch.allclose(omega_initializer, expected_initialization)


def test_ones_linear_tranform_initializer():
    omega_initializer = pt.transforms.LinearTransform(
        in_dim=2,
        out_dim=4,
        initializer=pt.initializers.OLTI(out_dim_first=True),
    ).weights

    expected_initialization = torch.Tensor(
        [
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
        ]
    )
    assert torch.allclose(omega_initializer, expected_initialization)


def test_linear_transform_out_dim_first1():
    omega_initializer = pt.transforms.LinearTransform(
        in_dim=2,
        out_dim=4,
        initializer=pt.initializers.OLTI(out_dim_first=False),
    ).weights

    expected_initialization = torch.Tensor(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    )
    assert torch.allclose(omega_initializer, expected_initialization)


def test_eye_omega_initializer_squre():
    omega_initializer = pt.initializers.EyeLinearTransformInitializer().generate(3, 3)
    expected_initialization = torch.Tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    assert torch.allclose(omega_initializer, expected_initialization)


def test_eye_transform_init_rect():
    omega_initializer = pt.initializers.EyeLinearTransformInitializer().generate(4, 2)
    desired = torch.Tensor(
        [
            [1, 0],
            [0, 1],
            [0, 0],
            [0, 0],
        ]
    )
    assert torch.allclose(omega_initializer, desired)


def test_eye_transform_init_rect_b():
    omega_initializers = pt.initializers.EyeLinearTransformInitializer(
        out_dim_first=True
    ).generate(4, 2)
    desired = torch.Tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]
    )
    assert torch.allclose(omega_initializers, desired)


def test_pca_linear_transform_initializer():
    input_features = torch.Tensor(
        [
            [-6, -2, 0, 1, 2, 0],
            [10, 11, 12, 10, 8, 9],
            [0, 0, 0, 0, 0, 0],
            [3, 2, 5, 4, 2, 4],
            [3, 2, 1, 4, 3, 3],
            [4, 2, 5, 2, 2, 4],
            [3, 1, 5, 4, 2, 4],
            [0, 2, 5, 1, 2, 4],
        ]
    )

    omega_initializer = pt.transforms.LinearTransform(
        in_dim=None,
        out_dim=6,
        initializer=pci.PCALTI(
            data=input_features,
            noise=0.1,
            out_dim_first=False,
        ),
    ).weights
    assert omega_initializer.shape[1] == input_features.shape[1]
    assert omega_initializer.shape[0] == input_features.shape[1]


def test_random_linear_transform_initializer():
    input_features = torch.Tensor(
        [
            [-6, -2, 0, 1],
            [10, 11, 12, 10],
            [0, 0, 0, 0],
            [3, 2, 5, 4],
            [3, 2, 1, 4],
            [4, 2, 5, 2],
            [3, 1, 5, 4],
            [0, 2, 5, 1],
        ]
    )

    omega_initializer = pt.transforms.LinearTransform(
        in_dim=2,
        out_dim=4,
        initializer=pci.RandomLinearTransformInitializer(out_dim_first=False),
    ).weights

    assert omega_initializer.shape[1] == input_features.shape[1]
    assert omega_initializer.shape[0] == input_features.shape[1] - 2
