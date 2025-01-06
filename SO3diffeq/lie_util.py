import torch

def SO3exp(w: torch.Tensor, eps = 1e-6)->torch.Tensor:
    """w shape (..., 3)"""
    theta = torch.norm(w, dim = -1, keepdim = True)
    small_theta_mask = theta[...,0] < eps
    Identity = torch.eye(3).expand(w.shape[:-1]+(3,3)).to(w.device)

    unit_w = w[~small_theta_mask] / theta[~small_theta_mask]
    s = torch.sin(theta[~small_theta_mask]).unsqueeze(-1)
    c = torch.cos(theta[~small_theta_mask]).unsqueeze(-1)

    Rotation = torch.zeros_like(Identity)
    Rotation[small_theta_mask] = Identity[small_theta_mask] + so3hat(w[small_theta_mask])
    # outer product is used here (follow Timothy Barfoot formulation, not conventional Rodrigues formula), also used in code from M. Brossard
    Rotation[~small_theta_mask] = c * Identity[~small_theta_mask] + (1-c) * outer_product(unit_w, unit_w) + s * so3hat(unit_w)
    return Rotation

def outer_product(v1: torch.Tensor, v2: torch.Tensor)->torch.Tensor:
    """v1, v2 shape (..., 3)"""
    return torch.einsum('...i,...j->...ij', v1, v2)

def so3hat(w: torch.Tensor)->torch.Tensor:
    """w shape (..., 3)"""
    return torch.stack([torch.zeros_like(w[..., 0]), -w[..., 2], w[..., 1],
                        w[..., 2], torch.zeros_like(w[..., 0]), -w[..., 0],
                        -w[..., 1], w[..., 0], torch.zeros_like(w[..., 0])], dim=-1).reshape(w.shape[:-1]+(3,3))
    
def so3vee(W: torch.Tensor)->torch.Tensor:
    """W shape (..., 3, 3)"""
    return torch.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], dim=-1)


#TODO: cayley map
