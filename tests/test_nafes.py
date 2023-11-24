"""test suite for nafes"""

import numpy as np
from nafes import (
    dataset,
    NafesPy,
    seed_everything,
    OmegaInitializers,
    PrototypeInitializers,
    get_rejection_summary,
    LVQ,
)

# Reproducibility
seed_everything(seed=4)

# Dataset
train_data = dataset.DATA(random=4)

# Input features of the dataset
input_data = train_data.breast_cancer.input_data

# Targets of the input features
labels = train_data.breast_cancer.labels

# Initialise prototypes
proto_init = PrototypeInitializers.STRATIFIEDMEANSCOMPONENTINITIALIZER.value

# Initialise omega matrix
omega_init = OmegaInitializers.ONESLINEARTRANSFORMINITIALIZER.value


def test_nafes_rejection_strategy():
    train = NafesPy(
        input_data=input_data,
        labels=labels,
        batch_size=128,
        model_name=LVQ.LGMLVQ.value,
        optimal_search="cpu",
        input_dim=input_data.shape[1],
        latent_dim=input_data.shape[1],
        num_classes=len(np.unique(labels)),
        num_prototypes=1,
        eval_type="mv",
        evaluation_metric="accuracy",
        perturbation_ratio=0.2,
        perturbation_distribution="balanced",
        epsilon=0.05,
        norm_ord="fro",
        termination="metric",
        verbose=1,
        proto_initializer=proto_init,
        omega_matrix_initializer=omega_init,
        max_epochs=100,
        significance=False,
    )
    summary = train.summary_results

    rejected_strategy = get_rejection_summary(
        significant=summary.significant.features,
        insignificant=summary.insignificant.features,
        significant_hit=summary.significant.hits,
        insignificant_hit=summary.insignificant.hits,
        reject_options=True,
        vis=True,
    )

    significant_features = rejected_strategy.significant
    insignificant_features = list(
        set(rejected_strategy.insignificant) - set(rejected_strategy.tentative)
    )

    tentative_features = rejected_strategy.tentative

    feature_space = significant_features + insignificant_features + tentative_features

    assert input_data.shape[1] == len(feature_space)


def test_nafes():
    train = NafesPy(
        input_data=input_data,
        labels=labels,
        batch_size=128,
        model_name=LVQ.GMLVQ,
        optimal_search="cpu",
        input_dim=input_data.shape[1],
        latent_dim=input_data.shape[1],
        num_classes=len(np.unique(labels)),
        num_prototypes=1,
        eval_type="mv",
        evaluation_metric="accuracy",
        perturbation_ratio=0.2,
        perturbation_distribution="balanced",
        epsilon=0.05,
        norm_ord="fro",
        termination="metric",
        verbose=1,
        proto_initializer=proto_init,
        omega_matrix_initializer=omega_init,
        max_epochs=100,
        significance=False,
    )
    summary = train.summary_results
    significant_features = summary.significant
    insignificant_features = summary.insignificant
    feature_space = list(significant_features) + list(insignificant_features)

    assert input_data.shape[1] == len(feature_space)
