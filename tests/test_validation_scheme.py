"""nafes validation scheme test suite"""

import numpy as np
from nafes import (
    mutate_labels,
    mutated_validation,
)

input_data = np.array(
    [
        [1, 2, 3, 4, 1, 3],
        [1, 0, 3, 4, 1, 3],
        [1, 2, 3, 9, 1, 3],
        [6, 2, 5, 4, 1, 3],
        [1, 0, 3, 2, 1, 7],
        [9, 2, 6, 4, 8, 3],
        [2, 2, 1, 4, 7, 3],
        [2, 2, 0, 4, 7, 3],
        [0, 2, 1, 4, 7, 3],
        [2, 2, 3, 4, 7, 3],
        [1, 2, 1, 4, 7, 3],
        [1, 5, 1, 4, 1, 3],
        [3, 1, 1, 2, 6, 1],
        [9, 1, 3, 1, 6, 4],
    ]
)

input_lables = np.array([0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1])
peturbation_distribution = mutate_labels.PerturbationDistribution
perturbation_type = mutate_labels.MutationType
evaluation_metrics = mutated_validation.EvaluationMetricsType


def test_target_info():
    perturbation_ratio = 0.2
    target_info = mutate_labels.get_target_info(
        labels=input_lables,
        perturbation_ratio=perturbation_ratio,
    )

    assert len(target_info.label_indices) == len(input_lables)
    assert target_info.mutation_value == 3
    assert len(target_info.unique_labels) == len(np.unique(input_lables))


def test_mutated_validation_uniform_unique():
    mutation_validation = mutate_labels.MutatedValidation(
        labels=input_lables,
        perturbation_ratio=0.8,
        perturbation_distribution=peturbation_distribution.UNIFORMCLASSAWARE.value,
        perturbation_type=perturbation_type.SWAPLABEL,
    )

    difference = mutation_validation.get_mutated_label_list - input_lables
    assert len(np.unique(difference)) > 1


def test_mutated_validation_uniform():
    mutation_validation = mutate_labels.MutatedValidation(
        labels=input_lables,
        perturbation_ratio=0.8,
        perturbation_distribution=peturbation_distribution.UNIFORMCLASSAWARE.value,
        perturbation_type=perturbation_type.SWAPNEXTLABEL,
    )

    difference = mutation_validation.get_mutated_label_list - input_lables
    assert len(np.unique(difference)) > 1


def test_mutated_validation_balanced_unique():
    mutation_validation = mutate_labels.MutatedValidation(
        labels=input_lables,
        perturbation_ratio=0.8,
        perturbation_distribution=peturbation_distribution.BALANCEDCLASSAWARE.value,
        perturbation_type=perturbation_type.SWAPLABEL,
    )

    difference = mutation_validation.get_mutated_label_list - input_lables
    assert len(np.unique(difference)) > 1


def test_mutated_validation_balanced():
    mutation_validation = mutate_labels.MutatedValidation(
        labels=input_lables,
        perturbation_ratio=0.8,
        perturbation_distribution=peturbation_distribution.BALANCEDCLASSAWARE.value,
        perturbation_type=perturbation_type.SWAPNEXTLABEL,
    )

    difference = mutation_validation.get_mutated_label_list - input_lables
    assert len(np.unique(difference)) > 1


def test_mutated_validation_global_unique():
    mutation_validation = mutate_labels.MutatedValidation(
        labels=input_lables,
        perturbation_ratio=0.8,
        perturbation_distribution=peturbation_distribution.GLOBAL.value,
        perturbation_type=perturbation_type.SWAPLABEL,
    )

    difference = mutation_validation.get_mutated_label_list - input_lables
    assert len(np.unique(difference)) > 1


def test_mutated_validation_global():
    mutation_validation = mutate_labels.MutatedValidation(
        labels=input_lables,
        perturbation_ratio=0.8,
        perturbation_distribution=peturbation_distribution.GLOBAL.value,
        perturbation_type=perturbation_type.SWAPNEXTLABEL,
    )

    difference = mutation_validation.get_mutated_label_list - input_lables
    assert len(np.unique(difference)) > 1


def test_evaluationmetric_accuracy():
    predicted_labels = np.array([0, 1, 1, 1, 0, 0])
    original_labels = np.array([0, 1, 1, 1, 1, 0])
    metric_score = mutated_validation.get_metric_score(
        predicted_labels=predicted_labels,
        original_labels=original_labels,
        metric=evaluation_metrics.ACCURACY.value,
    )

    assert np.allclose(metric_score.metric_score, 0.833, rtol=0.003333)
    assert np.allclose(metric_score.original_labels, original_labels)
    assert np.allclose(metric_score.predicted_labels, predicted_labels)


def test_evaluationmetric_precision():
    predicted_labels = np.array([0, 1, 1, 1, 0, 0])
    original_labels = np.array([0, 1, 1, 1, 1, 0])
    metric_score = mutated_validation.get_metric_score(
        predicted_labels=predicted_labels,
        original_labels=original_labels,
        metric=evaluation_metrics.PRECISION.value,
    )
    assert np.allclose(metric_score.metric_score, 1.0, rtol=0.00001)
    assert np.allclose(metric_score.original_labels, original_labels)
    assert np.allclose(metric_score.predicted_labels, predicted_labels)


def test_evaluationmetric_gmean():
    predicted_labels = np.array([0, 1, 1, 1, 0, 0])
    original_labels = np.array([0, 1, 1, 1, 1, 0])
    metric_score = mutated_validation.get_metric_score(
        predicted_labels=predicted_labels,
        original_labels=original_labels,
        metric=evaluation_metrics.G_MEAN.value,
    )
    assert np.allclose(metric_score.metric_score, 0.866, atol=0.000254)
    assert np.allclose(metric_score.original_labels, original_labels)
    assert np.allclose(metric_score.predicted_labels, predicted_labels)


def test_evaluationmetric_recall():
    predicted_labels = np.array([0, 1, 1, 1, 0, 0])
    original_labels = np.array([0, 1, 1, 1, 1, 0])
    metric_score = mutated_validation.get_metric_score(
        predicted_labels=predicted_labels,
        original_labels=original_labels,
        metric=evaluation_metrics.RECALL.value,
    )
    assert np.allclose(metric_score.metric_score, 0.75)
    assert np.allclose(metric_score.original_labels, original_labels)
    assert np.allclose(metric_score.predicted_labels, predicted_labels)


def test_evaluationmetric_roc_auc():
    predicted_labels = np.array([0, 1, 1, 1, 0, 0])
    original_labels = np.array([0, 1, 1, 1, 1, 0])
    metric_score = mutated_validation.get_metric_score(
        predicted_labels=predicted_labels,
        original_labels=original_labels,
        metric=evaluation_metrics.ROC_AUC.value,
    )
    assert np.allclose(metric_score.metric_score, 0.875, rtol=0.00001)
    assert np.allclose(metric_score.original_labels, original_labels)
    assert np.allclose(metric_score.predicted_labels, predicted_labels)
