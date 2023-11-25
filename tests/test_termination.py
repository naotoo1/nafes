"""nafes termination test suite"""

import torch
import torch.linalg as ln
from nafes import (
    get_matrix_stability,
    get_metric_stability,
)


def test_metric_termination_small_difference():
    metric1, metric2, epsilon = 0.833, 0.844, 0.05
    metric_stability = get_metric_stability(
        metric1=metric1,
        metric2=metric2,
        epsilon=epsilon,
    )
    difference = metric2 - metric1
    assert difference < epsilon
    assert metric_stability


def test_metric_termination_large_difference():
    metric1, metric2, epsilon = 0.833, 0.894, 0.05
    metric_stability = get_metric_stability(
        metric1=metric1,
        metric2=metric2,
        epsilon=epsilon,
    )
    difference = metric2 - metric1
    assert difference > epsilon
    assert not metric_stability


def test_metric_termination_no_difference():
    metric1, metric2, epsilon = 0.833, 0.833, 0.05
    metric_stability = get_metric_stability(
        metric1=metric1,
        metric2=metric2,
        epsilon=epsilon,
    )
    difference = metric2 - metric1
    assert difference <= epsilon
    assert metric_stability


def test_matrix_termination_small_difference():
    matrix_1 = torch.Tensor(
        [
            [0.94, -0.34],
            [0.80, 0.12],
        ]
    )

    matrix_2 = torch.Tensor(
        [
            [0.94, -0.34],
            [0.80, 0.10],
        ]
    )
    epsilon, ord = 0.05, "fro"
    matrix_stability = get_matrix_stability(
        matrix1=matrix_1,
        matrix2=matrix_2,
        epsilon=epsilon,
        matrix_ord=ord,
    )
    m = (ln.norm((matrix_2 - matrix_1), ord=ord)).numpy()
    assert m <= epsilon
    assert matrix_stability


def test_matrix_termination_large_differnce():
    matrix_1 = torch.Tensor(
        [
            [0.84, -0.34],
            [0.80, 0.12],
        ]
    )

    matrix_2 = torch.Tensor(
        [
            [0.64, -0.44],
            [0.80, 0.10],
        ]
    )
    epsilon, ord = 0.05, "fro"
    matrix_stability = get_matrix_stability(
        matrix1=matrix_1,
        matrix2=matrix_2,
        epsilon=epsilon,
        matrix_ord=ord,
    )
    m = (ln.norm((matrix_2 - matrix_1), ord=ord)).numpy()
    assert m > epsilon
    assert not matrix_stability


def test_matrix_termination_no_differnce():
    matrix_1 = torch.Tensor(
        [
            [0.84, -0.34],
            [0.80, 0.12],
        ]
    )

    matrix_2 = torch.Tensor(
        [
            [0.84, -0.34],
            [0.80, 0.12],
        ]
    )

    epsilon, ord = None, "fro"
    matrix_stability = get_matrix_stability(
        matrix1=matrix_1,
        matrix2=matrix_2,
        epsilon=epsilon,
        matrix_ord=ord,
    )
    m = (ln.norm((matrix_2 - matrix_1), ord=ord)).numpy()
    assert m <= 0.0
    assert matrix_stability
