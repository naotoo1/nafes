"""Implementation of Prototype Feature Selection  Algorithm"""

# Author: Nana Abeka Otoo <abekaotoo@gmail.com>
# License: MIT

from collections import Counter
from typing import Dict, Any
import logging
import os
import random
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd
import prototorch.models as ps
import prototorch.core.initializers as pci
import pytorch_lightning as pl
import torch
import torch.linalg as ln
from lightning_fabric.utilities.warnings import PossibleUserWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils import data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from .mutate_labels import MutatedValidation
from .mutated_validation import (
    MutatedValidationScore,
    TrainRun,
    EvaluationMetricsType,
)


torch.set_float32_matmul_precision(precision="high")
warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class Verbose(int, Enum):
    YES = 0
    NO = 1


class SavedModelUpdate(str, Enum):
    TRUE = "update"
    FALSE = "keine-update"


class ValidationType(str, Enum):
    MUTATEDVALIDATION = "mv"
    HOLDOUT = "ho"


class LVQ(str, Enum):
    GMLVQ = "gmlvq"
    LGMLVQ = "lgmlvq"


class OmegaInitializers(str, Enum):
    PCALINEARTRANSFORMINITIALIZER = "PCALTI"
    ZEROSCOMPINITIALIZER = "ZCI"
    EYELINEARTRANSFORMINITIALIZER = "ETLI"
    ONESLINEARTRANSFORMINITIALIZER = "OLTI"
    RANDOMLINEARTRANSFORMINITIALIZER = "RLTI"
    LITERALLINEARTRANSFORMINITIALIZER = "LLTI"


class PrototypeInitializers(str, Enum):
    STRATIFIEDMEANSCOMPONENTINITIALIZER = "SMCI"
    STRATIFIEDSELECTIONCOMPONENTINITIALIZER = "SSCI"
    ZEROSCOMPINITIALIZER = "ZCI"
    MEANCOMPONENTINITIALIZER = "MCI"
    ONESCOMPONENTINITIALIZER = "OCI"
    RANDOMNORMALCOMPONENTINITIALIZER = "RNCI"
    LITERALCOMPONENTINITIALIZER = "LCI"
    CLASSAWARECOMPONENTINITIALIZER = "CACI"
    DATAAWARECOMPONENTINITIALIZER = "DACI"
    FILLVALUECOMPONENTINITIALIZER = "FVCI"
    SELECTIONCOMPONENTINITIALIZER = "SCI"
    STRATIFIEDSELECTIONCOMPONENTININTIALIZER = "SSCI"
    UNIFORMCOMPONENTINITIALIZER = "UCI"


@dataclass
class LocalRejectStrategy:
    significant: list[int]
    insignificant: list[int]
    significant_hit: list[int]
    insignificant_hit: list[int]
    tentative: list[int] | None


@dataclass
class SelectedRelevances:
    significant: list[int] | list[str] | np.ndarray
    insignificant: list[str | int] | np.ndarray


@dataclass
class HitsInfo:
    features: list[int]
    hits: list[int]


@dataclass
class SelectedRelevancesExtra:
    significant: HitsInfo
    insignificant: HitsInfo


@dataclass
class BestLearnedResults:
    omega_matrix: list[torch.Tensor]
    evaluation_metric_score: list[float | int]
    num_prototypes: int


@dataclass
class GlobalRelevanceFactorsSummary:
    omega_matrix: torch.Tensor
    lambda_matrix: torch.Tensor
    lambda_diagonal: np.ndarray
    lambda_row_sum: np.ndarray
    feature_relevance_dict: Dict[str, Any]
    weight_significance: np.ndarray


@dataclass
class LocalRelevanceFactorSummary:
    omega_matrix: list[torch.Tensor]
    lambda_matrix: list[torch.Tensor]
    lambda_diagonal: list[torch.Tensor]
    lambda_row_sum: np.ndarray
    feature_relevance_dict: Dict[str, Any]
    weight_significance: np.ndarray
    feature_significance: np.ndarray


@dataclass
class GlobalFeatureSelection:
    relevance: GlobalRelevanceFactorsSummary
    eval_score: list[float | int]
    num_prototypes: int


@dataclass
class LocalFeatureSelection:
    relevance: LocalRelevanceFactorSummary
    eval_score: list[float | int]
    num_prototypes: int


@dataclass
class TrainModelSummary:
    selected_model_evaluation_metrics_scores: list[float | int]
    final_omega_matrix: list[torch.Tensor]
    final_prototypes: list[torch.Tensor]


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, "reset_parameters"):
            print(f"Reset trainable parameters of layer = {layer}")
            layer.reset_parameters()


Path("./evaluation").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename="evaluation/report.txt",
    encoding="utf-8",
    filemode="w",
    level=logging.INFO,
)


def evaluation_metric_logs(
    evaluation_metric_scores: list[float | int],
    model_name: str,
    validation: str,
    log: bool = True,
) -> None:
    report = [{validation: evaluation_metric_scores}]
    match log:
        case True:
            return logging.info("%s:%s", model_name, report)


def train_hold_out(
    input_data: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    optimal_search: str,
    input_dim: int,
    latent_dim: int,
    num_classes: int,
    num_prototypes: int = 1,
    proto_lr: float = 0.01,
    bb_lr: float = 0.01,
    optimizer=torch.optim.Adam,
    proto_initializer: str = "SMCI",
    omega_matrix_initializer: str = "OLTI",
    save_model: bool = False,
    max_epochs: int = 100,
    noise: float = 0.1,
    batch_size: int = 128,
    num_workers: int = 4,
    evaluation_metric: str = EvaluationMetricsType.ACCURACY.value,
) -> TrainModelSummary:
    prototypes, omega_matrix = [], []
    X_train, X_test, y_train, y_test = train_test_split(
        input_data, labels, test_size=0.3, random_state=4
    )

    x_input = torch.from_numpy(X_train).to(torch.float32)
    y_label = torch.from_numpy(y_train).to(torch.float32)
    x_input_test = torch.from_numpy(X_test).to(torch.float32)
    y_label_test = torch.from_numpy(y_test).to(torch.float32)

    train_ds = data.TensorDataset(x_input, y_label)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = matrix_glvq(
        model_name,
        train_ds,
        input_dim,
        latent_dim,
        num_prototypes,
        num_classes,
        proto_lr,
        bb_lr,
        optimizer,
        proto_initializer,
        omega_matrix_initializer,
        noise,
    )

    trainer = model_trainer(
        search=optimal_search,
        max_epoch=max_epochs,
    )

    trainer.fit(model, train_loader)
    prototypes.append(model.prototypes)
    omega_matrix.append(model.omega_matrix)

    outputs = model.predict(torch.Tensor(x_input_test))

    if save_model:
        save_train_model(
            saved_model_dir="./saved_models",
            model_name=model_name + "_" + ValidationType.HOLDOUT,
            estimator=model,
            scoring_metric=evaluation_metric,
        )

    return TrainModelSummary(
        selected_model_evaluation_metrics_scores=[
            accuracy_score(y_label_test, outputs)  # type: ignore
        ],
        final_omega_matrix=omega_matrix,
        final_prototypes=prototypes,
    )  # type: ignore


def train_model_by_mv(
    input_data: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    optimal_search: str,
    input_dim: int,
    latent_dim: int,
    num_classes: int,
    num_prototypes: int = 1,
    proto_lr: float = 0.01,
    bb_lr: float = 0.01,
    optimizer=torch.optim.Adam,
    proto_initializer: str = "SMCI",
    omega_matrix_initializer="OLTI",
    save_model: bool = False,
    max_epochs: int = 100,
    noise: float = 0.1,
    perturbation_distribution: str = "balanced",
    perturbation_ratio: float = 0.2,
    batch_size: int = 128,
    num_workers: int = 4,
    evaluation_metric: str = EvaluationMetricsType.ACCURACY.value,
) -> TrainModelSummary:
    x_input = torch.from_numpy(input_data).to(torch.float32)
    y_label = torch.from_numpy(labels).to(torch.float32)

    train_ds = data.TensorDataset(x_input, y_label)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = matrix_glvq(
        model_name,
        train_ds,
        input_dim,
        latent_dim,
        num_prototypes,
        num_classes,
        proto_lr,
        bb_lr,
        optimizer,
        proto_initializer,
        omega_matrix_initializer,
        noise,
    )

    model_mv = matrix_glvq(
        model_name,
        train_ds,
        input_dim,
        latent_dim,
        num_prototypes,
        num_classes,
        proto_lr,
        bb_lr,
        optimizer,
        proto_initializer,
        omega_matrix_initializer,
        noise,
    )

    model.apply(reset_weights)

    mutated_validation = MutatedValidation(
        labels=labels.astype(np.int64),
        perturbation_ratio=perturbation_ratio,
        perturbation_distribution=perturbation_distribution,
    )

    mutate_list = mutated_validation.get_mutated_label_list
    mutated_labels = torch.from_numpy(mutate_list).to(torch.float32)

    train_ds_mutated = data.TensorDataset(x_input, mutated_labels)
    train_loader_mutated = DataLoader(
        train_ds_mutated,
        # batch_size=128,
        batch_size=batch_size,
        num_workers=2,
    )

    results, mv_score, prototypes, omega_matrix = [], [], [], []
    for train_runs in range(2):
        match train_runs:
            case TrainRun.ORIGINAL:
                trainer = model_trainer(
                    search=optimal_search,
                    max_epoch=max_epochs,
                )
                trainer.fit(model, train_loader)
                prototypes.append(model.prototypes)
                omega_matrix.append(model.omega_matrix)
                results.append(model.predict(x_input))
                if save_model:
                    save_train_model(
                        saved_model_dir="./saved_models",
                        model_name=model_name,
                        estimator=model,
                        scoring_metric=evaluation_metric,
                    )

            case TrainRun.MUTATED:
                model_mv.apply(reset_weights)
                trainer = model_trainer(
                    search=optimal_search,
                    max_epoch=max_epochs,
                )
                trainer.fit(model_mv, train_loader_mutated)
                results.append(model_mv.predict(x_input))

                mv_scorer = MutatedValidationScore(
                    mutated_labels=mutated_validation,
                    mutate=mutate_list,
                    original_training_predicted_labels=results[0],
                    mutated_training_predicted_labels=results[1],
                    evaluation_metric=evaluation_metric,
                )
                mv_score.append(mv_scorer.get_mv_score)

    return TrainModelSummary(
        selected_model_evaluation_metrics_scores=mv_score,
        final_omega_matrix=omega_matrix,
        final_prototypes=prototypes,
    )


def matrix_glvq(
    model: str,
    train_ds: data.TensorDataset,
    input_dim: int,
    latent_dim: int,
    num_prototypes: int,
    num_classes: int,
    proto_lr: float,
    bb_lr: float,
    optimizer=torch.optim.Adam,
    proto_initializer: str = "SMCI",
    omega_matrix_initializer: str = "OLTI",
    noise: float = 0.1,
) -> ps.GMLVQ | ps.LGMLVQ:
    prototype_initializer = get_prototype_initializers(
        initializer=proto_initializer,
        train_ds=train_ds,
        noise=noise,
    )

    omega_initializer = get_omega_initializers(
        initializer=omega_matrix_initializer,
        train_ds=train_ds,
    )
    hparams = dict(
        input_dim=input_dim,
        latent_dim=latent_dim,
        distribution={
            "num_classes": num_classes,
            "per_class": num_prototypes,
        },
        proto_lr=proto_lr,
        bb_lr=bb_lr,
        optimizer=optimizer,
        omega_initializer=omega_initializer,
    )
    match model:
        case LVQ.GMLVQ:
            return ps.GMLVQ(
                hparams,
                prototypes_initializer=prototype_initializer,
            )
        case LVQ.LGMLVQ:
            return ps.LGMLVQ(
                hparams,
                prototypes_initializer=prototype_initializer,
            )
        case _:
            raise RuntimeError(
                "specified_lvq: none of the models did match",
            )


def model_trainer(search: str, max_epoch: int) -> pl.Trainer:  # type: ignore
    return pl.Trainer(
        max_epochs=max_epoch,
        enable_progress_bar=True,
        enable_checkpointing=False,
        logger=False,
        detect_anomaly=False,
        enable_model_summary=True,
        accelerator=search,
    )


def get_omega_initializers(
    initializer: str,
    train_ds: data.TensorDataset,
    out_dim_first: bool = False,
    noise: float = 0,
):
    match initializer:
        case OmegaInitializers.EYELINEARTRANSFORMINITIALIZER:
            return pci.ELTI(out_dim_first)
        case OmegaInitializers.PCALINEARTRANSFORMINITIALIZER:
            return pci.PCALTI(
                data=train_ds.tensors[0],
                noise=noise,
                out_dim_first=out_dim_first,
            )
        case OmegaInitializers.ZEROSCOMPINITIALIZER:
            return pci.ZLTI(out_dim_first)
        case OmegaInitializers.ONESLINEARTRANSFORMINITIALIZER:
            return pci.OLTI(out_dim_first)
        case OmegaInitializers.RANDOMLINEARTRANSFORMINITIALIZER:
            return pci.RLTI(out_dim_first)


def get_prototype_initializers(
    initializer: str,
    train_ds: data.TensorDataset,
    pre_initialised_prototypes: torch.Tensor | None = None,
    scale: float = 1,
    shift: float = 0,
    fill_value: float = 1,
    minimum: float = 0,
    maximum: float = 1,
    noise: float = 0,
):
    match initializer:
        case PrototypeInitializers.STRATIFIEDMEANSCOMPONENTINITIALIZER:
            return pci.SMCI(data=train_ds, noise=noise)
        case PrototypeInitializers.STRATIFIEDSELECTIONCOMPONENTINITIALIZER:
            return pci.SSCI(data=train_ds, noise=noise)
        case PrototypeInitializers.SELECTIONCOMPONENTINITIALIZER:
            return pci.SCI(data=train_ds.tensors[0], noise=noise)
        case PrototypeInitializers.MEANCOMPONENTINITIALIZER:
            return pci.MCI(data=train_ds.tensors[0], noise=noise)
        case PrototypeInitializers.ZEROSCOMPINITIALIZER:
            return pci.ZCI(shape=train_ds.tensors[0].size()[1])
        case PrototypeInitializers.ONESCOMPONENTINITIALIZER:
            return pci.OCI(shape=train_ds.tensors[0].size()[1])
        case PrototypeInitializers.LITERALCOMPONENTINITIALIZER:
            return pci.LCI(components=pre_initialised_prototypes)
        case PrototypeInitializers.CLASSAWARECOMPONENTINITIALIZER:
            return pci.CACI(data=train_ds, noise=noise)
        case PrototypeInitializers.DATAAWARECOMPONENTINITIALIZER:
            return pci.DACI(data=train_ds.tensors[0], noise=noise)
        case PrototypeInitializers.RANDOMNORMALCOMPONENTINITIALIZER:
            return pci.RNCI(
                shape=train_ds.tensors[0].size()[1], shift=shift, scale=scale
            )
        case PrototypeInitializers.FILLVALUECOMPONENTINITIALIZER:
            return pci.FVCI(train_ds.tensors[0].size()[1], fill_value)
        case PrototypeInitializers.UNIFORMCOMPONENTINITIALIZER:
            return pci.UCI(
                shape=train_ds.tensors[0].size()[1],
                minimum=minimum,
                maximum=maximum,
                scale=scale,
            )
        case _:
            raise RuntimeError(
                "get_prototype_initializers:none of the above matches",
            )


def save_train_model(
    *,
    saved_model_dir: str,
    model_name: str,
    estimator: ps.GMLVQ | ps.LGMLVQ,
    scoring_metric: str,
):
    Path(saved_model_dir).mkdir(parents=True, exist_ok=True)
    try:
        torch.save(
            estimator,
            os.path.join(
                saved_model_dir,
                model_name + scoring_metric + ".pt",
            ),
        )
    except AttributeError:
        pass


@dataclass(slots=True)
class NafesPy:
    input_data: np.ndarray
    labels: np.ndarray
    model_name: str
    optimal_search: str
    input_dim: int
    latent_dim: int
    num_classes: int
    num_prototypes: int
    feature_list: list[str] | None = None
    eval_type: str | None = None
    proto_lr: float = 0.01
    bb_lr: float = 0.01
    optimizer = torch.optim.Adam
    proto_initializer: str = "SMCI"
    omega_matrix_initializer: str = "OLTI"
    max_epochs: int = 10
    batch_size: int = 128
    num_workers: int = 4
    save_model: bool = False
    significance: bool = True
    perturbation_distribution: str = "balanced"
    perturbation_ratio: float = 0.2
    evaluation_metric: str = EvaluationMetricsType.ACCURACY.value
    epsilon: float | None = 0.0001
    norm_ord: str = "fro"
    termination: str = "metric"
    patience: int = 1
    verbose: int = 0
    summary_metric_list: list = field(default_factory=lambda: [])

    def train_ho(self, increment: int) -> TrainModelSummary:
        return train_hold_out(
            input_data=self.input_data,
            labels=self.labels,
            model_name=self.model_name,
            optimal_search=self.optimal_search,
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            num_classes=self.num_classes,
            num_prototypes=self.num_prototypes + increment,
            save_model=self.save_model,
            evaluation_metric=self.evaluation_metric,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            optimizer=self.optimizer,
            proto_initializer=self.proto_initializer,
            omega_matrix_initializer=self.omega_matrix_initializer,
            max_epochs=self.max_epochs,
        )

    def train_mv(self, increment: int) -> TrainModelSummary:
        return train_model_by_mv(
            input_data=self.input_data,
            labels=self.labels,
            model_name=self.model_name,
            optimal_search=self.optimal_search,
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            num_classes=self.num_classes,
            num_prototypes=self.num_prototypes + increment,
            save_model=self.save_model,
            evaluation_metric=self.evaluation_metric,
            perturbation_distribution=self.perturbation_distribution,
            perturbation_ratio=self.perturbation_ratio,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            optimizer=self.optimizer,
            proto_initializer=self.proto_initializer,
            omega_matrix_initializer=self.omega_matrix_initializer,
            max_epochs=self.max_epochs,
        )

    @property
    def final(self) -> BestLearnedResults:  # type: ignore
        (metric_list, matrix_list, should_continue, counter) = ([], [], True, -1)
        while should_continue:
            counter += 1
            train_eval_scheme = (
                self.train_mv(increment=counter)
                if self.eval_type == ValidationType.MUTATEDVALIDATION.value
                else self.train_ho(increment=counter)
            )
            validation_score = (
                train_eval_scheme.selected_model_evaluation_metrics_scores
            )
            omega_matrix = train_eval_scheme.final_omega_matrix
            num_prototypes = (
                len((train_eval_scheme.final_prototypes[0]).numpy()) // self.num_classes
            )
            metric_list.append(validation_score[0])
            matrix_list.append(omega_matrix[0])
            if counter < self.patience:
                continue
            condition = counter == self.max_epochs
            stability = get_stability(
                metric1=metric_list[-2],
                metric2=metric_list[-1],
                matrix1=matrix_list[-2],
                matrix2=matrix_list[-1],
                convergence=self.termination,
                epsilon=self.epsilon,
                learner=self.model_name,
                matrix_ord=self.norm_ord,
            )

            match (condition, self.termination, stability):
                case (False, "metric", True):
                    should_continue = False
                    return BestLearnedResults(
                        omega_matrix=omega_matrix,
                        evaluation_metric_score=validation_score,
                        num_prototypes=num_prototypes,
                    )
                case (False, "matrix", True):
                    should_continue = False
                    return BestLearnedResults(
                        omega_matrix=omega_matrix,
                        evaluation_metric_score=validation_score,
                        num_prototypes=num_prototypes,
                    )
                case (True, "metric", True):
                    should_continue = False
                    return BestLearnedResults(
                        omega_matrix=omega_matrix,
                        evaluation_metric_score=validation_score,
                        num_prototypes=num_prototypes,
                    )
                case (True, "matrix", True):
                    should_continue = False
                    return BestLearnedResults(
                        omega_matrix=omega_matrix,
                        evaluation_metric_score=validation_score,
                        num_prototypes=num_prototypes,
                    )
                # case _:
                #     raise RuntimeError("final:none of the above cases match")

    @property
    def feature_selection(self) -> GlobalFeatureSelection | LocalFeatureSelection:  # type: ignore
        match (self.model_name, self.eval_type):
            case (LVQ.GMLVQ, ValidationType.MUTATEDVALIDATION.value):
                train_eval_scheme = self.final
                validation_score = train_eval_scheme.evaluation_metric_score
                omega_matrix = train_eval_scheme.omega_matrix
                relevance = get_lambda_matrix(
                    omega_matrix=omega_matrix, feature_list=self.feature_list
                )
                return GlobalFeatureSelection(
                    relevance=relevance,
                    eval_score=validation_score,
                    num_prototypes=train_eval_scheme.num_prototypes,
                )

            case (LVQ.GMLVQ, ValidationType.HOLDOUT.value):
                train_eval_scheme = self.final
                validation_score = train_eval_scheme.evaluation_metric_score
                omega_matrix = train_eval_scheme.omega_matrix
                relevance = get_lambda_matrix(
                    omega_matrix=omega_matrix, feature_list=self.feature_list
                )
                return GlobalFeatureSelection(
                    relevance=relevance,
                    eval_score=validation_score,
                    num_prototypes=train_eval_scheme.num_prototypes,
                )

            case (LVQ.LGMLVQ, ValidationType.MUTATEDVALIDATION.value):
                train_eval_scheme = self.final
                validation_score = train_eval_scheme.evaluation_metric_score
                omega_matrix = train_eval_scheme.omega_matrix
                num_prototypes = train_eval_scheme.num_prototypes
                relevance = get_local_lambda_matrix(
                    omega_matrix=omega_matrix,
                    feature_list=self.feature_list,
                    num_prototypes=num_prototypes,
                    num_classes=self.num_classes,
                )
                return LocalFeatureSelection(
                    relevance=relevance,
                    eval_score=validation_score,
                    num_prototypes=num_prototypes,
                )

            case (LVQ.LGMLVQ, ValidationType.HOLDOUT.value):
                train_eval_scheme = self.final
                validation_score = train_eval_scheme.evaluation_metric_score
                omega_matrix = train_eval_scheme.omega_matrix
                num_prototypes = train_eval_scheme.num_prototypes
                relevance = get_local_lambda_matrix(
                    omega_matrix=omega_matrix,
                    feature_list=self.feature_list,
                    num_classes=self.num_classes,
                    num_prototypes=num_prototypes,
                )
                return LocalFeatureSelection(
                    relevance=relevance,
                    eval_score=validation_score,
                    num_prototypes=num_prototypes,
                )

            case _:
                raise RuntimeError(
                    "feature_selection: none of the checks did match",
                )

    @property
    def summary_results(self):
        feature_selection = self.feature_selection

        match (self.model_name, self.significance):
            case (LVQ.LGMLVQ, True):
                summary = get_relevance_summary(
                    feature_significance=feature_selection.relevance.feature_significance,  # type: ignore
                    evaluation_metric_score=feature_selection.eval_score[0],
                    verbose=self.verbose,
                )
                return SelectedRelevances(
                    significant=summary.significant,
                    insignificant=summary.insignificant,
                )
            case (LVQ.LGMLVQ, False):
                summary = get_relevance_elimination_summary(
                    weight_significance=feature_selection.relevance.lambda_row_sum,
                    num_protypes_per_class=self.feature_selection.num_prototypes,
                    lambda_row_sum=feature_selection.relevance.lambda_row_sum,
                    evaluation_metric_score=feature_selection.eval_score[0],
                    verbose=self.verbose,
                    input_dim=self.input_dim,
                    num_classes=self.num_classes,
                )

                visualize(
                    features=summary.significant.features,
                    hits=summary.significant.hits,
                    significance=True,
                    eval_score=feature_selection.eval_score[0],
                )
                visualize(
                    features=summary.insignificant.features,
                    hits=summary.insignificant.hits,
                    significance=False,
                    eval_score=feature_selection.eval_score[0],
                )
                return SelectedRelevancesExtra(
                    significant=summary.significant,
                    insignificant=summary.insignificant,
                )
            case (LVQ.GMLVQ, True):
                summary = get_relevance_global_summary(
                    lambda_row_sum=feature_selection.relevance.lambda_row_sum,
                    weight_significance=feature_selection.relevance.weight_significance,
                    significance=self.significance,
                    evaluation_metric_score=feature_selection.eval_score[0],
                    verbose=self.verbose,
                )
                return SelectedRelevances(
                    significant=np.array(summary.significant).flatten(),
                    insignificant=np.array(summary.insignificant).flatten(),
                )
            case (LVQ.GMLVQ, False):
                summary = get_relevance_global_summary(
                    lambda_row_sum=feature_selection.relevance.lambda_row_sum,
                    weight_significance=feature_selection.relevance.weight_significance,
                    significance=self.significance,
                    evaluation_metric_score=feature_selection.eval_score[0],
                    verbose=self.verbose,
                )
                return SelectedRelevances(
                    significant=np.array(summary.significant).flatten(),
                    insignificant=np.array(summary.insignificant).flatten(),
                )
            case _:
                raise RuntimeError("summary_results: none of the above cases match")


def get_lambda_matrix(
    omega_matrix: list[torch.Tensor], feature_list: list[str] | None = None
) -> GlobalRelevanceFactorsSummary:
    omega = omega_matrix[0]
    lambda_matrix = omega.T @ omega
    list_of_features = np.arange(lambda_matrix.size(dim=1))
    lambda_diagonal = torch.diagonal(lambda_matrix, 0).numpy()
    row_sum_squared_omega_ij = torch.sum(lambda_matrix, 1).numpy()
    attributes = feature_list if feature_list is not None else list_of_features

    feature_relevance_elimination_dict = dict(zip(attributes, row_sum_squared_omega_ij))
    weight_features_significance = np.argsort(lambda_diagonal)[::-1]

    return GlobalRelevanceFactorsSummary(
        omega_matrix=omega,
        lambda_matrix=lambda_matrix,
        lambda_diagonal=lambda_diagonal,
        lambda_row_sum=row_sum_squared_omega_ij,
        feature_relevance_dict=feature_relevance_elimination_dict,
        weight_significance=weight_features_significance,
    )


def get_local_lambda_matrix(
    omega_matrix: list[torch.Tensor],
    num_classes: int,
    num_prototypes: int,
    feature_list: list[str] | None = None,
) -> LocalRelevanceFactorSummary:
    omega = omega_matrix[0]
    (
        omega_matrix,
        lambda_matrix,
        lambda_diagonal,
        lambda_row_sum,
        feature_relevance_dict,
        weight_significance,
    ) = ([], [], [], [], [], [])
    for local_matrix in omega:
        relevance_factor_summary = get_lambda_matrix(
            omega_matrix=[local_matrix], feature_list=feature_list
        )
        omega_matrix.append(relevance_factor_summary.omega_matrix)
        lambda_matrix.append(relevance_factor_summary.lambda_matrix)
        lambda_diagonal.append(relevance_factor_summary.lambda_diagonal)
        lambda_row_sum.append(relevance_factor_summary.lambda_row_sum)
        feature_relevance_dict.append(relevance_factor_summary.feature_relevance_dict)
        weight_significance.append(relevance_factor_summary.weight_significance)
    feature_signficance = np.array(
        [f"f{weight[0]}" for _index, weight in enumerate(weight_significance)]
    ).reshape(num_classes, num_prototypes)
    class_labels = [
        f"class_label_{label}_relevances_{i}"
        for label in np.arange(len(omega))
        for i in range(num_classes)
    ]
    feature_relevance_dict = dict(zip(class_labels, feature_relevance_dict))
    return LocalRelevanceFactorSummary(
        omega_matrix=omega_matrix,
        lambda_matrix=lambda_matrix,
        lambda_diagonal=lambda_diagonal,
        lambda_row_sum=lambda_row_sum,  # type: ignore
        feature_relevance_dict=feature_relevance_dict,
        weight_significance=np.array(weight_significance),
        feature_significance=feature_signficance,
    )


def get_relevance_global_summary(
    lambda_row_sum: np.ndarray,
    weight_significance: np.ndarray,
    evaluation_metric_score: float,
    significance: bool,
    verbose: int = 0,
) -> SelectedRelevances:
    significant, insignificant = [], []
    select_vector = weight_significance if significance is True else lambda_row_sum
    get_eval_score(
        evaluation_metric_score=evaluation_metric_score,
        verbose=verbose,
    )
    match significance:
        case True:
            for index, class_relevance in enumerate(select_vector):
                summary = get_global_logs(
                    index=index,
                    class_relevance=class_relevance,
                    verbose=verbose,
                    state="pass",
                )
                significant.append(summary.significant)

        case False:
            for feature_index, feature_label in enumerate(
                np.argsort(lambda_row_sum)[::-1], start=1
            ):
                cond = float(lambda_row_sum[feature_label]) > 0
                match cond:
                    case True:
                        summary = get_global_logs(
                            index=feature_index,
                            class_relevance=feature_label,
                            verbose=verbose,
                            state="pass",
                        )
                        significant.append(summary.significant)
                    case False:
                        summary = get_global_logs(
                            index=feature_index,
                            class_relevance=feature_label,
                            verbose=verbose,
                            state="fail",
                        )
                        insignificant.append(summary.insignificant)
                    case _:
                        raise RuntimeError(
                            "relevance features: none of the above cases match"
                        )
        case _:
            raise RuntimeError(
                "get_relevance_global_summary: none of the above cases match"
            )
    return SelectedRelevances(
        significant=significant,
        insignificant=insignificant,
    )


def get_relevance_summary(
    feature_significance: np.ndarray,
    evaluation_metric_score: float,
    verbose: int = 0,
) -> SelectedRelevances:
    significant_features = []

    get_eval_score(
        evaluation_metric_score=evaluation_metric_score,
        verbose=verbose,
    )
    for index, class_relevance in enumerate(feature_significance):
        for feature_index, feature_label in enumerate(class_relevance):
            match verbose:
                case Verbose.YES:
                    print(
                        "Passes the test: ",
                        f"class {index}",
                        " - Ranking: ",
                        f"prototype {feature_index}.",
                        feature_label,
                        "✔️",
                    )
                    significant_features.append(feature_label)
                case Verbose.NO:
                    logging.info(
                        "%s %s %s %s %s %s",
                        "Passes the test: ",
                        f"class {index}",
                        " - Ranking: ",
                        f"prototype {feature_index}.",
                        feature_label,
                        "✔️",
                    )
                    significant_features.append(feature_label)
                case _:
                    raise RuntimeError(
                        "get_relevance_summary: none of the cases matches"
                    )

    significant_features = [
        int(feature.replace("f", "")) for feature in significant_features
    ]
    return SelectedRelevances(
        significant=significant_features,
        insignificant=[],
    )


def get_eval_score(
    evaluation_metric_score: float,
    verbose: int = 0,
):
    match verbose:
        case Verbose.YES:
            print(
                f"Evaluation Score:{evaluation_metric_score:.2f}",
            )
        case Verbose.NO:
            logging.info(
                "%s %s",
                "Evaluation Score: ",
                evaluation_metric_score,
            )
        case _:
            raise RuntimeError(
                "get_eval_score:none of the cases match",
            )


def get_global_logs(
    index: int,
    class_relevance: int,
    verbose: int = 0,
    state: str = "pass",
) -> SelectedRelevances:
    significant_features, insignificant = [], []
    verbosity = verbose == Verbose.YES
    status = state == "pass"
    match (verbosity, status):
        case (True, True):
            print(
                "Passes the test: ",
                index,
                " - Ranking: ",
                f"f{class_relevance}",
                "✔️",
            )  # log here
            significant_features.append(class_relevance)
        case (False, True):
            logging.info(
                "%s %s %s %s %s",
                "Passes the test: ",
                index,
                " - Ranking: ",
                f"f{class_relevance}",
                "✔️",
            )
            significant_features.append(class_relevance)
        case (True, False):
            print(
                "Fails the test: ",
                index,
                " - Ranking: ",
                f"f{class_relevance}",
                "❌",
            )
            insignificant.append(class_relevance)
        case (False, False):
            logging.info(
                "%s %s %s %s %s",
                "Fails the test: ",
                index,
                " - Ranking: ",
                f"f{class_relevance}",
                "❌",
            )
            insignificant.append(class_relevance)
    return SelectedRelevances(
        significant=significant_features,
        insignificant=insignificant,
    )


def get_relevance_logs(
    label: int,
    index: int,
    feature_label: str,
    verbose: int = 0,
    state: str = "pass",
) -> SelectedRelevances:
    significant_features, insignificant = (
        [],
        [],
    )
    verbosity = verbose == Verbose.YES
    status = state == "pass"

    match (verbosity, status):
        case (True, True):
            print(
                "Passes the test: ",
                f"class {label}",
                " - Ranking: ",
                f"prototype {index}.",
                f"f{feature_label}",
                "✔️",
            )
            significant_features.append(feature_label)
        case (False, True):
            logging.info(
                "%s %s %s %s %s %s",
                "Passes the test: ",
                f"class {label}",
                " - Ranking: ",
                f"prototype {index}.",
                f"f{feature_label}",
                "✔️",
            )
            significant_features.append(feature_label)
        case (True, False):
            print(
                "Fails the test: ",
                f"class {label}",
                " - Ranking: ",
                f"prototype {index}.",
                f"f{feature_label}",
                "❌",
            )
            insignificant.append(feature_label)
        case (False, False):
            logging.info(
                "%s %s %s %s %s %s",
                "Fails the test: ",
                f"class {label}",
                " - Ranking: ",
                f"prototype {index}.",
                f"f{feature_label}",
                "❌",
            )
            insignificant.append(feature_label)
    return SelectedRelevances(
        significant=significant_features,
        insignificant=insignificant,
    )


def get_relevance_elimination_summary(
    weight_significance: np.ndarray,
    num_protypes_per_class: int,
    num_classes: int,
    input_dim: int,
    lambda_row_sum: np.ndarray,
    evaluation_metric_score: float,
    verbose: int = 0,
) -> SelectedRelevancesExtra:
    significant, insignificant = [], []
    get_eval_score(
        evaluation_metric_score=evaluation_metric_score,
        verbose=verbose,
    )

    num_prototypes = len(weight_significance) // num_classes
    # print(num_prototypes)

    weight_significance = np.reshape(
        weight_significance, (num_classes, num_prototypes, input_dim)
    )

    for class_label, weight_summary in enumerate(weight_significance):
        for index, class_relevance in enumerate(weight_summary):
            for _feature_index, feature_label in enumerate(
                np.argsort(class_relevance)[::-1]
            ):
                classwise_relevance = list(lambda_row_sum[index])
                cond_1 = float(classwise_relevance[feature_label]) > 0
                match cond_1:
                    case True:
                        report = get_relevance_logs(
                            label=class_label,
                            index=index,
                            feature_label=feature_label,
                            verbose=verbose,
                            state="pass",
                        )
                        significant.append(report.significant)

                    case False:
                        report = get_relevance_logs(
                            label=class_label,
                            index=index,
                            feature_label=feature_label,
                            verbose=verbose,
                            state="fail",
                        )
                        insignificant.append(report.insignificant)

    return SelectedRelevancesExtra(
        significant=get_hits_significance(np.array(significant).flatten()),
        insignificant=get_hits_significance(np.array(insignificant).flatten()),
    )


def get_hits_significance(summary_list: np.ndarray) -> HitsInfo:
    summary_dict = Counter(summary_list)  # type: ignore
    sorted_feature_keys = sorted(
        summary_dict, key=lambda k: (summary_dict[k], k), reverse=True
    )

    hits = [summary_dict[key] for key in sorted_feature_keys]

    return HitsInfo(
        features=sorted_feature_keys,  # type: ignore
        hits=hits,
    )


def get_matrix_stability(
    matrix1, matrix2, epsilon: float | None, matrix_ord: int | str
) -> bool:
    distance = (ln.norm((matrix2 - matrix1), ord=matrix_ord)).numpy()
    if epsilon is not None:
        return bool(0 < distance <= epsilon)
    return bool(distance == 0)


def get_metric_stability(
    metric1: float,
    metric2: float,
    epsilon: float | None,
) -> bool:
    metric1, metric2 = np.round(metric1, 2), np.round(metric2, 2)
    difference = metric2 - metric1
    if epsilon is not None:
        return bool(0 <= difference <= epsilon)
    return bool(metric2 == metric1)


def get_stability(
    metric1: float,
    metric2: float,
    matrix1: torch.Tensor,
    matrix2: torch.Tensor,
    convergence: str,
    epsilon: float | None,
    learner: str,
    matrix_ord: str,
) -> bool:
    match (convergence, learner):
        case ("metric", LVQ.GMLVQ):
            return get_metric_stability(
                metric1=metric1,
                metric2=metric2,
                epsilon=epsilon,
            )
        case ("matrix", LVQ.GMLVQ):
            return get_matrix_stability(
                matrix1=matrix1,
                matrix2=matrix2,
                epsilon=epsilon,
                matrix_ord=matrix_ord,
            )
        case ("metric", LVQ.LGMLVQ):
            return get_metric_stability(
                metric1=metric1,
                metric2=metric2,
                epsilon=epsilon,
            )
        case ("matrix", LVQ.LGMLVQ):
            raise RuntimeError(
                "get_stability: computational cost may be very high: consider metric case"
            )
        case _:
            raise RuntimeError(
                "get_stability: none of the cases match",
            )


def visualize(
    features: list,
    hits: list,
    significance: bool,
    eval_score: float | None,
):
    relevance = "significant " if significance is True else "insignificant"
    _fig, ax = plt.subplots()
    ax.bar(features, hits)
    ax.set_xticks(features)
    ax.set_ylabel("number of hits per prototype")
    ax.set_xlabel(f"{relevance} features")
    # ax.set_title(f"Feature relevance summary ({np.round(eval_score,4)})")
    ax.set_title("Feature relevance summary")
    plt.savefig(f"evaluation/feature_{relevance}_rank_plot.png")


def reject_strategy(
    significant: list,
    insignificant: list,
    significant_hit: list,
    insignificant_hit: list,
) -> LocalRejectStrategy:
    intersection = list(set(significant) & set(insignificant))

    index_sig_conflicts = [
        index for index, value in enumerate(significant) if value in intersection
    ]

    value_sig_conflicts = [
        value for index, value in enumerate(significant) if value in intersection
    ]

    index_insig_conflicts = [
        index for index, value in enumerate(insignificant) if value in intersection
    ]

    value_insig_conflicts = [
        value for index, value in enumerate(insignificant) if value in intersection
    ]

    index_list, new_val_insig = [], []
    for val_sig in value_sig_conflicts:
        for index_value, value in enumerate(value_insig_conflicts):
            if val_sig == value:
                new_val_insig.append(value)
                index_list.append(index_insig_conflicts[index_value])

    significant_hits = [significant_hit[index] for index in index_sig_conflicts]

    insignificant_hits = [insignificant_hit[index] for index in index_list]

    rejection_strategy_significant = [
        value_sig_conflicts[hit_index]
        for hit_index, hit in enumerate(significant_hits)
        if hit <= insignificant_hits[hit_index]
    ]

    rejection_strategy_insignificant = [
        new_val_insig[hit_index]
        for hit_index, hit in enumerate(insignificant_hits)
        if hit < significant_hits[hit_index]
    ]

    tentative_strategy = [
        value_sig_conflicts[hit_index]
        for hit_index, hit in enumerate(significant_hits)
        if hit == insignificant_hits[hit_index]
    ]

    new_significant = [
        feature
        for feature in significant
        if feature not in rejection_strategy_significant
    ]

    new_insignificant = [
        feature
        for feature in insignificant
        if feature not in rejection_strategy_insignificant
    ]

    new_significant_hit = [
        significant_hit[index]
        for index, value in enumerate(significant)
        if value in new_significant
    ]

    new_insignificant_hit = [
        insignificant_hit[index]
        for index, value in enumerate(insignificant)
        if value in new_insignificant
    ]

    significant = significant if len(new_significant) == 0 else new_significant

    insignificant = insignificant if len(new_insignificant) == 0 else new_insignificant

    significant_hit = (
        significant_hit if len(new_significant) == 0 else new_significant_hit
    )

    insignificant_hit = (
        insignificant_hit if len(new_insignificant) == 0 else new_insignificant_hit
    )

    insignificant = [
        feature
        for _index, feature in enumerate(insignificant)
        if feature not in significant
    ]

    insignificant_hit = [
        insignificant_hit[index]
        for index, feature in enumerate(insignificant)
        if feature not in significant
    ]

    return LocalRejectStrategy(
        significant=significant,
        insignificant=insignificant,
        significant_hit=significant_hit,
        insignificant_hit=insignificant_hit,
        tentative=tentative_strategy,
    )


def reject(
    significant: list,
    insignificant: list,
    significant_hit: list,
    insignificant_hit: list,
    reject_options: bool,
) -> LocalRejectStrategy:
    match reject_options:
        case True:
            strategy = reject_strategy(
                significant,
                insignificant,
                significant_hit,
                insignificant_hit,
            )
            return LocalRejectStrategy(
                significant=strategy.significant,
                insignificant=strategy.insignificant,
                significant_hit=strategy.significant_hit,
                insignificant_hit=strategy.insignificant_hit,
                tentative=strategy.tentative,
            )
        case False:
            return LocalRejectStrategy(
                significant, insignificant, significant_hit, insignificant_hit, None
            )
        case _:
            raise RuntimeError(
                "reject:none of the above matches",
            )


def get_rejection_summary(
    significant: list,
    insignificant: list,
    significant_hit: list,
    insignificant_hit: list,
    reject_options: bool,
    vis: bool,
) -> LocalRejectStrategy:
    rejected_summary = reject(
        significant,
        insignificant,
        significant_hit,
        insignificant_hit,
        reject_options,
    )

    match vis:
        case True:
            visualize(
                features=rejected_summary.significant,
                hits=rejected_summary.significant_hit,
                significance=True,
                eval_score=None,
            )
            visualize(
                features=rejected_summary.insignificant,
                hits=rejected_summary.insignificant_hit,
                significance=False,
                eval_score=None,
            )
        case False:
            pass
        case _:
            raise RuntimeError(
                "get_rejection_summary:none of the cases matches",
            )

    return LocalRejectStrategy(
        significant=rejected_summary.significant,
        insignificant=rejected_summary.insignificant,
        significant_hit=rejected_summary.significant_hit,
        insignificant_hit=rejected_summary.insignificant_hit,
        tentative=rejected_summary.tentative,
    )


def get_ozone_data(file_dir: str) -> tuple[np.ndarray, np.ndarray]:
    dataframe = pd.read_csv(file_dir)
    dataframe.drop(columns="Date")
    dataframe.dropna()
    dataframe = dataframe.to_numpy()[1:2535]
    dataframe = dataframe.tolist()
    dataset = []
    for case in dataframe:
        case.pop(0)
        dataset.append(case)

    features = np.array(
        [
            [eval(value) for _index, value in enumerate(instance[:-1])]
            for instance in dataset
            if "?" not in instance
        ]
    )
    labels = np.array([instance[-1] for instance in dataset if "?" not in instance])
    return features, labels


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


if __name__ == "__main__":
    print("import module to use")
