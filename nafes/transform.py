"""implementation of the prototypes transform from the projected space"""

import torch

def pseudo_inverse_svd(omega):
    """Computes the pseudo-inverse of A using SVD."""
    U, S, Vh = torch.linalg.svd(omega, full_matrices=False)
    S_inv = torch.diag(1.0 / S) 
    return Vh.T @ S_inv @ U.T

def get_transformed_prototypes(omega,p_prototypes):
    """Computes the transformed prototypes."""
    return torch.linalg.pinv(omega)@p_prototypes.permute(*torch.arange(p_prototypes.ndim - 1, -1, -1))

def get_generalised_transformed_prototypes(omega,p_prototypes):
    """Computes the  generalised transformed prototypes."""
    return [get_transformed_prototypes(omega,prototypes) for prototypes in p_prototypes ]

def get_local_transformed_prototypes(omega, p_prototypes):
    """Computes the localised transformed prototypes."""
    return [get_transformed_prototypes(w, prototypes) for w, prototypes in zip(omega, p_prototypes)]

def transformed_prototypes(model:str,omega,p_prototypes):
    match model:
        case 'gmlvq':
            return get_generalised_transformed_prototypes(omega,p_prototypes)
        case 'lgmlvq':
            return get_local_transformed_prototypes(omega, p_prototypes)
        case _:
            raise RuntimeError(
                    "feature_selection: none of the checks did match",
            )
