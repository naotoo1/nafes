"""implementation of the prototypes transform from the projected space"""

import torch

def pseudo_inverse_svd(omega: torch.Tensor) -> torch.Tensor:
    """Computes the pseudo-inverse of A using SVD."""
    U, S, Vh = torch.linalg.svd(omega, full_matrices=False)
    S_inv = torch.diag(1.0 / S)
    return Vh.T @ S_inv @ U.T

def get_transformed_prototypes(
        omega: torch.Tensor,
        p_prototypes: torch.Tensor
) -> torch.Tensor:
    """Computes the transformed prototypes."""
    prototypes = p_prototypes.permute(
        *torch.arange(p_prototypes.ndim - 1, -1, -1)
    )
    return torch.linalg.pinv(omega) @ prototypes

def get_generalised_transformed_prototypes(
        omega: torch.Tensor,
        p_prototypes: torch.Tensor
) -> list[torch.Tensor]:
    """Computes the generalised transformed prototypes."""
    return [
        get_transformed_prototypes(omega, p) for p in p_prototypes
    ]

def get_local_transformed_prototypes(
        omega: torch.Tensor,
        p_prototypes: torch.Tensor
) -> list[torch.Tensor]:
    """Computes the localised transformed prototypes."""
    return [
        get_transformed_prototypes(w, p) for w, p in zip(omega, p_prototypes)
    ]

def transformed_prototypes(
        model: str, omega: torch.Tensor,
        p_prototypes: torch.Tensor
) -> list[torch.Tensor]:
    match model:
        case 'gmlvq':
            return get_generalised_transformed_prototypes(
                omega,
                p_prototypes
            )
        case 'lgmlvq':
            return get_local_transformed_prototypes(
                omega,
                p_prototypes
            )
        case _:
            raise RuntimeError(
                "feature_selection: none of the checks did match",
            )
