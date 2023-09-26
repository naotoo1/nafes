"""
Prototype-based global feature selection example using the wdbc dataset
"""

import numpy as np
from nafes import (
    dataset,
    NafesPy,
    seed_everything,
    OmegaInitializers,
    PrototypeInitializers,
)

if __name__ == "__main__":
    # Reproducibility
    seed_everything(seed=4)

    # Dataset
    train_data = dataset.DATA(random=4)

    # Input features of the dataset
    input_data = train_data.breast_cancer.input_data

    # Targets of the input features
    labels = train_data.breast_cancer.labels

    # Initialise prototypes
    proto_init = PrototypeInitializers.STRATIFIEDMEANSCOMPONENTINITIALIZER

    # Initialise omega matrix
    omega_init = OmegaInitializers.ONESLINEARTRANSFORMINITIALIZER

    # set up the prototype feature selection
    train = NafesPy(
        input_data=input_data,
        labels=labels,
        batch_size=128,
        model_name='gmlvq',
        optimal_search='cpu',
        input_dim=input_data.shape[1],
        latent_dim=input_data.shape[1],
        num_classes=len(np.unique(labels)),
        num_prototypes=1,
        eval_type='mv',
        evaluation_metric='accuracy',
        perturbation_ratio=0.2,
        perturbation_distribution='balanced',
        epsilon=0.05,
        norm_ord='fro',
        termination='metric',
        verbose=1,
        proto_initializer=proto_init,
        omega_matrix_initializer=omega_init,
        max_epochs=100,
        significance=False
    )

    # Summary of NafesPy
    summary = train.summary_results

    # Summary of significant features
    print(
        'significant_features=',
        summary.significant,
    )

    # Summary of insignificant features
    print(
        'insignificant_features=',
        summary.insignificant,
    )
