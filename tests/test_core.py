"""nafes prototype transform test suite"""


import torch

from nafes.transform import pseudo_inverse_svd 

def test_pseudo_inverse():

    omega = torch.tensor(
        [
            [1.0, 2.0], 
            [3.0, 4.0], 
            [5.0, 6.0]
        ]
    )

    A_pinv_torch = torch.linalg.pinv(omega)
    A_pinv_svd = pseudo_inverse_svd(omega)

    assert torch.allclose(A_pinv_torch, A_pinv_svd, atol=1e-6)

def test_transformed_prototypes():

    omega = torch.tensor(
        [
            [1.0, 2.0], 
            [3.0, 4.0], 
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],

        ]
    )

    projected_prototypes = torch.Tensor(
        [
            [3, 2, 5, 4, 2, 4]
        ]
    )
    
    A_pinv_torch = torch.linalg.pinv(omega)
    A_pinv_svd = pseudo_inverse_svd(omega)

    prototypes_transform_1 = A_pinv_svd @ projected_prototypes.T
    prototypes_transform_2 = A_pinv_torch @ projected_prototypes.T
    

    assert torch.allclose(projected_prototypes.T, projected_prototypes.permute(*torch.arange(projected_prototypes.ndim - 1, -1, -1)))
    assert torch.allclose(prototypes_transform_1, prototypes_transform_2, atol=1e-6)

