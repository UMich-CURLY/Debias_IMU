import torch

def SO3exp(w: torch.Tensor, eps = 1e-6)->torch.Tensor:
    """w shape (..., 3)"""
    theta = torch.norm(w, dim = -1, keepdim = True)
    small_theta_mask = theta[...,0] < eps
    Identity = torch.eye(3).expand(w.shape[:-1]+(3,3)).to(w.device).to(w.dtype)

    unit_w = w[~small_theta_mask] / theta[~small_theta_mask]
    s = torch.sin(theta[~small_theta_mask]).unsqueeze(-1)
    c = torch.cos(theta[~small_theta_mask]).unsqueeze(-1)

    Rotation = torch.zeros_like(Identity)
    Rotation[small_theta_mask] = Identity[small_theta_mask] + so3hat(w[small_theta_mask])
    # outer product is used here (follow Timothy Barfoot formulation, not conventional Rodrigues formula), also used in code from M. Brossard
    Rotation[~small_theta_mask] = c * Identity[~small_theta_mask] + (1-c) * outer_product(unit_w, unit_w) + s * so3hat(unit_w)
    return Rotation

def SO3log(R: torch.Tensor, eps = 1e-6)->torch.Tensor:
    """R shape (..., 3, 3)"""
    trace = torch.einsum('...ii->...', R)
    # trace = torch.clamp(trace, -1 + eps, 3-eps)
    # assert torch.all(trace > -1-eps) and torch.all(trace < 3 + eps), "trace should be in the range of [-1,3]"
    theta = torch.acos( ((trace - 1.0) / 2.0).clamp(-1+1e-5,1-1e-5) )
    small_theta_mask = theta < eps
    Identity = torch.eye(3).expand(R.shape[:-2]+(3,3)).to(R.device).to(R.dtype)

    w_so3 = torch.zeros_like(R)
    w_so3[small_theta_mask] = R[small_theta_mask] - Identity[small_theta_mask]
    w_so3[~small_theta_mask] = (0.5 * theta[~small_theta_mask] / torch.sin(theta[~small_theta_mask])).unsqueeze(-1).unsqueeze(-1) * (R[~small_theta_mask] - R[~small_theta_mask].transpose(-1,-2))
    w = so3vee(w_so3)
    return w
    
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

def sen3hat(xi: torch.Tensor)->torch.Tensor:
    """xi shape (..., 3*n), n = 1,2,3..."""
    dim_mat = round(xi.shape[-1] // 3) +2 
    output = torch.zeros(xi.shape[:-1]+(dim_mat,dim_mat)).to(xi.device)
    output[..., :3, :3] = so3hat(xi[..., :3])
    output[..., :3, 3:] = xi[...,3:].reshape(*xi.shape[:-1], -1,3).transpose(-1,-2)
    return output

def SO3toEuler(R: torch.Tensor, order = "xyz")->torch.Tensor:
    """R shape (..., 3, 3)"""
    if order == "xyz":
        phi = torch.atan2(R[..., 2, 1], R[..., 2, 2])
        theta = -torch.asin(R[..., 2, 0])
        psi = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    elif order == "zyx":
        phi = torch.atan2(R[..., 1, 0], R[..., 0, 0])
        theta = -torch.asin(R[..., 2, 0])
        psi = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    else:
        raise ValueError("order should be 'xyz' or 'zyx'")
    return torch.stack((phi, theta, psi), dim = -1)

def SEn3fromSO3Rn(R: torch.Tensor, x: torch.Tensor)->torch.Tensor:
    """R shape (..., 3, 3), x shape (..., 3n)"""
    dim_mat = 3 + round(x.shape[-1] // 3)
    output = torch.eye(dim_mat).expand(R.shape[:-2]+(dim_mat,dim_mat)).clone().to(R.device)
    output[..., :3, :3] = R
    output[..., :3, 3:] = x.reshape(*x.shape[:-1], -1,3).transpose(-1,-2)
    return output

def sen3vee(X: torch.Tensor)->torch.Tensor:
    """X shape (..., 3+n, 3+n), n = 1,2,3..."""
    reshaped_tensor = X[...,:3,3:].transpose(-1,-2).reshape(*X.shape[:-2], -1)
    return torch.cat((so3vee(X[..., :3, :3]), reshaped_tensor), dim = -1)

def SEn3exp(xi: torch.Tensor, eps = 1e-6)->torch.Tensor:
    """xi shape (..., 3*n), n = 1,2,3..."""
    phi = xi[..., :3]
    R = SO3exp(phi)
    dim_mat = round(xi.shape[-1]//3)+2
    output = torch.eye(dim_mat).expand(xi.shape[:-1]+(dim_mat,dim_mat)).clone().to(xi.device)
    output[..., :3, :3] = R
    Jl =SO3leftJaco(phi)
    temp_rest = Jl @ xi[..., 3:].reshape(*xi.shape[:-1], -1,3).transpose(-1,-2)
    output[..., :3, 3:] = temp_rest
    return output

def SEn3log(X: torch.Tensor, eps = 1e-6)->torch.Tensor:
    """X shape (..., 3+n, 3+n), n = 1,2,3..."""
    """output shape (..., 3*n)"""
    phi = SO3log(X[..., :3, :3])
    xi = torch.zeros(X.shape[:-2]+((X.shape[-1]-2)*3,)).to(X.device)
    xi[..., :3] = phi
    temp_rest = SO3leftJacoInv(phi) @ X[..., :3, 3:]
    xi[..., 3:] = temp_rest.transpose(-1,-2).reshape(*temp_rest.shape[:-2], -1)
    return xi

def SO3leftJaco(phi: torch.Tensor, eps = 1e-6)->torch.Tensor:
    """left jacobian of SO(3), phi shape (..., 3)"""
    theta = torch.norm(phi, dim = -1, keepdim = True)
    small_theta_mask = theta[...,0] < eps
    Identity = torch.eye(3).expand(phi.shape[:-1]+(3,3)).to(phi.device)

    unit_phi = phi[~small_theta_mask] / theta[~small_theta_mask]
    sss = (torch.sin(theta[~small_theta_mask])/theta[~small_theta_mask]).unsqueeze(-1)
    ccc = ((1.0- torch.cos(theta[~small_theta_mask]))/theta[~small_theta_mask]).unsqueeze(-1)

    Jaco = torch.zeros_like(Identity)
    Jaco[small_theta_mask] = Identity[small_theta_mask] + 0.5 * so3hat(phi[small_theta_mask])
    Jaco[~small_theta_mask] = sss * Identity[~small_theta_mask] + (1.0 - sss) * outer_product(unit_phi, unit_phi) + ccc * so3hat(unit_phi)
    return Jaco

def SO3leftJacoInv(phi: torch.Tensor, eps = 1e-6)->torch.Tensor:
    """inverse of left jacobian of SO(3)"""
    theta = torch.norm(phi, dim = -1, keepdim = True)
    small_theta_mask = theta[...,0] < eps
    ## check singularity
    lage_theta_mask = theta[...,0] > 1.8 * torch.pi 
    remaider = theta[lage_theta_mask] % (2 * torch.pi)
    assert torch.all( torch.min(remaider, torch.abs(2 * torch.pi - remaider)) > 1e-1 ), "theta should not be a multiple of 2pi"
    ## end check singularity
    Identity = torch.eye(3).expand(phi.shape[:-1]+(3,3)).to(phi.device)

    unit_phi = phi[~small_theta_mask] / theta[~small_theta_mask]
    sss_cot = (theta[~small_theta_mask]/(2.0 * torch.tan(theta[~small_theta_mask]/2.0))).unsqueeze(-1)

    Jaco = torch.zeros_like(Identity)
    Jaco[small_theta_mask] = Identity[small_theta_mask] - 0.5 * so3hat(phi[small_theta_mask])
    Jaco[~small_theta_mask] = sss_cot * Identity[~small_theta_mask] + (1.0 - sss_cot) * outer_product(unit_phi, unit_phi) - 0.5 * so3hat(phi[~small_theta_mask])
    return Jaco

def SO3rightJaco(phi: torch.Tensor, eps = 1e-6)->torch.Tensor:
    """right jacobian of SO(3), phi shape (..., 3)"""
    return SO3leftJaco(-phi)

def SO3rightJacoInv(phi: torch.Tensor, eps = 1e-6)->torch.Tensor:
    """inverse of right jacobian of SO(3)"""
    return SO3leftJacoInv(-phi)

def SEn3inverse(X: torch.Tensor, numerial_invese = False)->torch.Tensor:
    """inverse of SEn(3) matrix, X, shape (..., 3+n, 3+n) n = 1,2,3..."""
    if numerial_invese:
        X_inv = torch.linalg.inv(X)
    else:
        X_inv = X.clone()
        X_inv[..., :3, :3] = X[..., :3, :3].transpose(-2, -1)
        # temp_rest = - torch.matmul(X_inv[..., :3, :3], X[..., :3, 3:])
        # X_inv[..., :3, 3:] = temp_rest # TODO: check why this cause an error of auto-grad
        X_inv[..., :3, 3:] = - torch.matmul(X_inv[..., :3, :3], X[..., :3, 3:])
    return X_inv


def SEn3leftJaco(xi: torch.Tensor, eps = 1e-6)->torch.Tensor:
    """left jacobian of SEn(3), phi shape (..., 3*n), n = 1,2,3..., Order: xi_R, xi_v, xi_p ..."""
    """                         phi should be (m1,m2,3*n) or (m1,3*n)"""
    phi = xi[..., :3]

    theta = torch.norm(phi, dim = -1, keepdim = True)
    mask_small_theta = theta[..., 0] < eps
    Identity = torch.eye(xi.shape[-1]).expand(phi.shape[:-1]+(xi.shape[-1],xi.shape[-1])).to(phi.device)

    Jaco = torch.zeros_like(Identity)
    Jaco[mask_small_theta] = Identity[mask_small_theta] + 0.5 * ad_sen3(xi[mask_small_theta])
    Jaco_left_SO3 = SO3leftJaco(phi[~mask_small_theta])
    temp = Jaco[~mask_small_theta]
    for i in range(round(xi.shape[-1] // 3)):
        temp[:, 3*i:3*(i+1), 3*i:3*(i+1)] = Jaco_left_SO3
    temp[:, 3:, :3] = Ql_forSE3Jaco(xi[~mask_small_theta])
    Jaco[~mask_small_theta] = temp
    return Jaco


def SEn3leftJaco_inv(xi: torch.Tensor, eps = 1e-6)->torch.Tensor:
    phi = xi[..., :3]

    theta = torch.norm(phi, dim = -1, keepdim = True)
    mask_small_theta = theta[..., 0] < eps
    Identity = torch.eye(xi.shape[-1]).expand(phi.shape[:-1]+(xi.shape[-1],xi.shape[-1])).to(phi.device).to(xi.dtype)

    Jaco = torch.zeros_like(Identity)
    # Jaco = torch.zeros(xi.shape[:-1]+(xi.shape[-1],xi.shape[-1])).to(xi.device)
    Jaco[mask_small_theta] = Identity[mask_small_theta] - 0.5 * ad_sen3(xi[mask_small_theta])
    Jaco_left_SO3_inv = SO3leftJacoInv(phi[~mask_small_theta])
    temp = Jaco[~mask_small_theta]
    Ql = Ql_forSE3Jaco(xi[~mask_small_theta])
    temp[:, :3, :3] = Jaco_left_SO3_inv
    for i in range(1, round(xi.shape[-1] // 3)):
        temp[:, 3*i:3*(i+1), 3*i:3*(i+1)] = Jaco_left_SO3_inv
        temp[:, 3*i:3*(i+1), :3] = - Jaco_left_SO3_inv @ Ql[:, 3*(i-1):3*i, :3] @ Jaco_left_SO3_inv
    Jaco[~mask_small_theta] = temp
    return Jaco

def SEn3rightJaco(xi: torch.Tensor, eps = 1e-6)->torch.Tensor:
    """right jacobian of SEn(3), phi shape (..., 3*n), n = 1,2,3..., Order: xi_R, xi_v, xi_p ..."""
    return SEn3leftJaco(-xi, eps)

def SEn3rightJaco_inv(xi: torch.Tensor, eps = 1e-6)->torch.Tensor:
    """inverse of right jacobian of SEn(3)"""
    return SEn3leftJaco_inv(-xi, eps)
    

def Ql_forSE3Jaco(xi: torch.Tensor)->torch.Tensor:
    """Ql for SEn(3) left jacobian, phi shape (..., 3*n) Order: xi_R, xi_v, xi_p ..."""
    """Assume xi is free of singularity, this should be checked before calling this function"""
    N = round(xi.shape[-1] // 3)
    Ql = torch.zeros(xi.shape[:-1]+(xi.shape[-1]-3,3)).to(xi.device)
    phi = xi[..., :3]
    phi_wedge = so3hat(phi)

    theta = torch.norm(phi[..., :3], dim = -1, keepdim = True)
    theta2 = theta * theta
    theta3 = theta2 * theta
    theta4 = theta3 * theta
    theta5 = theta4 * theta

    s_theta = torch.sin(theta)
    c_theta = torch.cos(theta)

    m1 = 0.5
    m2 = ((theta - s_theta) / theta3).unsqueeze(-1)
    m3 = ((theta2 + 2 * c_theta - 2) / (2 * theta4)).unsqueeze(-1)
    m4 = ((2*theta - 3*s_theta + theta * c_theta) / (2 * theta5)).unsqueeze(-1)

    for i in range(N-1):
        nu_wedge = so3hat(xi[..., 3*(i+1):3*(i+2)])
        v1 =nu_wedge
        v2 = phi_wedge @ nu_wedge + nu_wedge @ phi_wedge + phi_wedge @ nu_wedge @ phi_wedge
        v3 = phi_wedge @ phi_wedge @ nu_wedge + nu_wedge @ phi_wedge @ phi_wedge - 3 * phi_wedge @ nu_wedge @ phi_wedge
        v4 = phi_wedge @ nu_wedge @ phi_wedge @ phi_wedge + phi_wedge @ phi_wedge @ nu_wedge @ phi_wedge
        Ql[..., 3*i:3*(i+1), :] = m1 * v1 + m2 * v2 + m3 * v3 + m4 * v4

    return Ql

def ad_sen3(xi: torch.Tensor)->torch.Tensor:
    """
    Compute the adjoint matrix from a SEn(3) matrix
    :param xi: sen3 lie algebra, shape (...,3*n) Order: xi_R, xi_v, xi_p ...
    :return: the adjoint matrix, shape (...,3*n,3*n)
    """
    adm = torch.zeros(xi.shape[:-1]+(xi.shape[-1],xi.shape[-1])).to(xi.device)
    N = round(xi.shape[-1] // 3)
    phi_wedge = so3hat(xi[..., :3])
    adm[..., :3, :3] = phi_wedge
    for i in range(1,N):
        adm[..., 3*i:3*(i+1), 3*i:3*(i+1)] = phi_wedge
        adm[..., 3*i:3*(i+1), :3] = so3hat(xi[..., 3*i:3*(i+1)])
    return adm

def quat_to_SO3(q: torch.Tensor, order = "wxyz")->torch.Tensor:
    """q shape (..., 4)"""
    if order == "wxyz":
        q0 = q[..., 0]
        q1 = q[..., 1]
        q2 = q[..., 2]
        q3 = q[..., 3]
    elif order == "xyzw":
        q0 = q[..., 3]
        q1 = q[..., 0]
        q2 = q[..., 1]
        q3 = q[..., 2]
    q0_2 = q0 * q0
    q1_2 = q1 * q1
    q2_2 = q2 * q2
    q3_2 = q3 * q3
    q0q1 = q0 * q1
    q0q2 = q0 * q2
    q0q3 = q0 * q3
    q1q2 = q1 * q2
    q1q3 = q1 * q3
    q2q3 = q2 * q3
    R = torch.zeros(q.shape[:-1]+(3,3)).to(q.device).to(q.dtype)
    R[..., 0, 0] = q0_2 + q1_2 - q2_2 - q3_2
    R[..., 0, 1] = 2 * (q1q2 - q0q3)
    R[..., 0, 2] = 2 * (q1q3 + q0q2)
    R[..., 1, 0] = 2 * (q1q2 + q0q3)
    R[..., 1, 1] = q0_2 - q1_2 + q2_2 - q3_2
    R[..., 1, 2] = 2 * (q2q3 - q0q1)
    R[..., 2, 0] = 2 * (q1q3 - q0q2)
    R[..., 2, 1] = 2 * (q2q3 + q0q1)
    R[..., 2, 2] = q0_2 - q1_2 - q2_2 + q3_2
    return R

def SO3_to_quat(Rots, ordering='wxyz'):
    """Convert a rotation matrix to a unit length quaternion.
    Valid orderings are 'xyzw' and 'wxyz'.
    """
    tmp = 1 + Rots[:, 0, 0] + Rots[:, 1, 1] + Rots[:, 2, 2]
    tmp[tmp < 0] = 0
    qw = 0.5 * torch.sqrt(tmp)
    qx = qw.new_empty(qw.shape[0])
    qy = qw.new_empty(qw.shape[0])
    qz = qw.new_empty(qw.shape[0])

    near_zero_mask = qw.abs() < 1e-8

    if near_zero_mask.sum() > 0:
        cond1_mask = near_zero_mask * \
            (Rots[:, 0, 0] > Rots[:, 1, 1])*(Rots[:, 0, 0] > Rots[:, 2, 2])
        cond1_inds = cond1_mask.nonzero()

        if len(cond1_inds) > 0:
            cond1_inds = cond1_inds.squeeze()
            R_cond1 = Rots[cond1_inds].view(-1, 3, 3)
            d = 2. * torch.sqrt(1. + R_cond1[:, 0, 0] -
                R_cond1[:, 1, 1] - R_cond1[:, 2, 2]).view(-1)
            qw[cond1_inds] = (R_cond1[:, 2, 1] - R_cond1[:, 1, 2]) / d
            qx[cond1_inds] = 0.25 * d
            qy[cond1_inds] = (R_cond1[:, 1, 0] + R_cond1[:, 0, 1]) / d
            qz[cond1_inds] = (R_cond1[:, 0, 2] + R_cond1[:, 2, 0]) / d

        cond2_mask = near_zero_mask * (Rots[:, 1, 1] > Rots[:, 2, 2])
        cond2_inds = cond2_mask.nonzero()

        if len(cond2_inds) > 0:
            cond2_inds = cond2_inds.squeeze()
            R_cond2 = Rots[cond2_inds].view(-1, 3, 3)
            d = 2. * torch.sqrt(1. + R_cond2[:, 1, 1] -
                            R_cond2[:, 0, 0] - R_cond2[:, 2, 2]).squeeze()
            tmp = (R_cond2[:, 0, 2] - R_cond2[:, 2, 0]) / d
            qw[cond2_inds] = tmp
            qx[cond2_inds] = (R_cond2[:, 1, 0] + R_cond2[:, 0, 1]) / d
            qy[cond2_inds] = 0.25 * d
            qz[cond2_inds] = (R_cond2[:, 2, 1] + R_cond2[:, 1, 2]) / d

        cond3_mask = near_zero_mask & cond1_mask.logical_not() & cond2_mask.logical_not()
        cond3_inds = cond3_mask

        if len(cond3_inds) > 0:
            R_cond3 = Rots[cond3_inds].view(-1, 3, 3)
            d = 2. * \
                torch.sqrt(1. + R_cond3[:, 2, 2] -
                R_cond3[:, 0, 0] - R_cond3[:, 1, 1]).squeeze()
            qw[cond3_inds] = (R_cond3[:, 1, 0] - R_cond3[:, 0, 1]) / d
            qx[cond3_inds] = (R_cond3[:, 0, 2] + R_cond3[:, 2, 0]) / d
            qy[cond3_inds] = (R_cond3[:, 2, 1] + R_cond3[:, 1, 2]) / d
            qz[cond3_inds] = 0.25 * d

    far_zero_mask = near_zero_mask.logical_not()
    far_zero_inds = far_zero_mask
    if len(far_zero_inds) > 0:
        R_fz = Rots[far_zero_inds]
        d = 4. * qw[far_zero_inds]
        qx[far_zero_inds] = (R_fz[:, 2, 1] - R_fz[:, 1, 2]) / d
        qy[far_zero_inds] = (R_fz[:, 0, 2] - R_fz[:, 2, 0]) / d
        qz[far_zero_inds] = (R_fz[:, 1, 0] - R_fz[:, 0, 1]) / d

    # Check ordering last
    if ordering == 'xyzw':
        quat = torch.stack([qx, qy, qz, qw], dim=1)
    elif ordering == 'wxyz':
        quat = torch.stack([qw, qx, qy, qz], dim=1)
    return quat

def quat_normlize(q: torch.Tensor)->torch.Tensor:
    """q shape (..., 4)"""
    return q / q.norm(dim=-1, keepdim=True)

def quat_slerp(q0: torch.Tensor, q1: torch.Tensor, tau: torch.Tensor, DOT_THRESHOLD = 0.9995):
        """Spherical linear interpolation."""
        assert q0.shape == q1.shape and q0.shape[1] == 4 and tau.dim() == 1 and tau.shape[0] == q0.shape[0], "q0, q1, tau should have shape (N, 4), (N, 4), (N, ) respectively"

        q0 = q0 / q0.norm(dim=-1, keepdim=True)
        q1 = q1 / q1.norm(dim=-1, keepdim=True)

        dot = (q0*q1).sum(dim=1)
        q1[dot < 0] = -q1[dot < 0]
        dot[dot < 0] = -dot[dot < 0]

        q = torch.zeros_like(q0)
        tmp = q0 + tau.unsqueeze(1) * (q1 - q0)
        tmp = tmp[dot > DOT_THRESHOLD]
        q[dot > DOT_THRESHOLD] = tmp / tmp.norm(dim=1, keepdim=True)

        theta_0 = dot.clamp(-1., 1.).acos()
        sin_theta_0 = theta_0.sin()
        theta = theta_0 * tau
        sin_theta = theta.sin()
        s0 = (theta.cos() - dot * sin_theta / sin_theta_0).unsqueeze(1)
        s1 = (sin_theta / sin_theta_0).unsqueeze(1)
        q[dot < DOT_THRESHOLD] = ((s0 * q0) + (s1 * q1))[dot < DOT_THRESHOLD]
        return q / q.norm(dim=1, keepdim=True)

def SO3_to_RPY(Rots):
    """Convert a rotation matrix to RPY Euler angles. Copy from https://github.com/mbrossar/denoise-imu-gyro.git
    Args:  Rots: shape (batch_size, 3, 3)
    """

    pitch = torch.atan2(-Rots[:, 2, 0],
        torch.sqrt(Rots[:, 0, 0]**2 + Rots[:, 1, 0]**2))
    yaw = pitch.new_empty(pitch.shape)
    roll = pitch.new_empty(pitch.shape)

    near_pi_over_two_mask = (pitch - torch.pi / 2.).abs() < 1e-8 
    near_neg_pi_over_two_mask = (pitch + torch.pi / 2.).abs() < 1e-8

    remainder_inds = ~(near_pi_over_two_mask | near_neg_pi_over_two_mask)

    yaw[near_pi_over_two_mask] = 0
    roll[near_pi_over_two_mask] = torch.atan2(
        Rots[near_pi_over_two_mask, 0, 1],
        Rots[near_pi_over_two_mask, 1, 1])

    yaw[near_neg_pi_over_two_mask] = 0.
    roll[near_neg_pi_over_two_mask] = -torch.atan2(
        Rots[near_neg_pi_over_two_mask, 0, 1],
        Rots[near_neg_pi_over_two_mask, 1, 1])

    sec_pitch = 1/pitch[remainder_inds].cos()
    remainder_mats = Rots[remainder_inds]
    yaw = torch.atan2(remainder_mats[:, 1, 0] * sec_pitch,
                        remainder_mats[:, 0, 0] * sec_pitch)
    roll = torch.atan2(remainder_mats[:, 2, 1] * sec_pitch,
                        remainder_mats[:, 2, 2] * sec_pitch)
    rpys = torch.cat([roll.unsqueeze(dim=1),
                    pitch.unsqueeze(dim=1),
                    yaw.unsqueeze(dim=1)], dim=1)
    return rpys

if __name__ == '__main__':
    print("Test lie_algebra.py")
    print("---------------------------------")
    torch.set_default_dtype(torch.float64)

    ## test SO3exp
    device = 'cuda'
    w = torch.tensor([[1.6469, 3.7091, 1.3493],[1e-7,0,0],[0,torch.pi/3,0],[0,0,torch.pi/4]]).to(device).unsqueeze(0)
    w = w.repeat(2,1,1)
    # print("w.shape", w.shape)
    R = SO3exp(w)
    R_t = torch.linalg.matrix_exp(so3hat(w))
    error = torch.norm(R - R_t)
    if error > 1e-6:
        raise ValueError("error: ", error)
    else:
        print("SO3exp test passed, error: ", error)

    ## test SO3log
    w_log = SO3log(R)
    R2 = SO3exp(w_log)
    error = torch.norm(R - R2)
    # w_norm = torch.norm(w, dim = -1, keepdim = True)
    # w_unit = w / w_norm
    # w_clampwith2pi = w_unit * (w_norm % (2. *torch.pi))
    # print("w_clampwith2pi \n", w_clampwith2pi)
    # error = torch.norm(w_unit * (w_norm % (2. *torch.pi)) - w_log)
    # for i in range(2):
    #     for j in range(4):
    #         print("error at ", i, j, ":", torch.norm(w[i,j,...] - w_log[i,j,...]), "w norm: ", torch.norm(w[i,j,...]))
    if error > 1e-6:
        raise ValueError("error: ", error)
    else:
        print("SO3log test passed, error: ", error)



    # test SEn3inverse
    A = torch.eye(5).to(device).unsqueeze(0).unsqueeze(0)
    A = A.repeat(10,2,1,1)
    temp = SO3exp(torch.randn(10,2,3))
    A[..., :3, :3] = temp
    A[..., :3, 3] = torch.randn(10,2,3)
    A[...,:3,4] = torch.randn(10,2,3)
    A_inv = SEn3inverse(A, numerial_invese = False)
    temp = torch.matmul(A, A_inv)
    error = torch.norm(temp - torch.eye(5).to(device))
    if error > 1e-6:
        raise ValueError("error: ", error)
    else:
        print("SEn3inverse test passed, error: ", error)


    ## test SO3leftJaco
    import math
    phi = torch.tensor([[torch.pi/3,0,0],[torch.pi*1.95,0,0],[1e-4,0,0],[0,0,0.]]).to(device).unsqueeze(0)
    # phi = phi.repeat(2,1,1)
    # phi = phi.repeat(2,1,1)
    # print("phi.shape", phi.shape)
    temp1 = SO3leftJaco(phi)
    # def left_Jaco_true(xi: torch.Tensor):
    #     Jaco = torch.eye(3).repeat(*so3hat(xi).shape[:-2]+(1,1)).to(xi.device)
    #     ad_mult = Jaco.clone()
    #     for i in range(1,20):
    #         ad_mult = ad_mult @ so3hat(xi)
    #         Jaco += ad_mult / math.factorial(i+1)
    #     return Jaco
    # temp1_true = left_Jaco_true(phi)
    # error = torch.norm(temp1 - temp1_true)
    # if error > 1e-6:
    #     print("temp1 \n", temp1)
    #     print("temp1_true \n", temp1_true)
    #     raise ValueError("error: ", error)
    # else:
    #     print("SO3leftJaco test passed, error: ", error)
    # print(temp1)
    temp2 = SO3leftJacoInv(phi)
    temp3 = torch.matmul(temp1,temp2)
    error = torch.norm(temp3 - torch.eye(3).to(device).unsqueeze(0).unsqueeze(0).repeat(2,4,1,1))
    if error > 1e-6:
        raise ValueError("error: ", error)
    else:
        print("SO3leftJaco and SO3leftJacoInv test passed, error: ", error)
    pass
    # phi = torch.tensor([torch.pi * 1.8,0,0])
    # temp1 = SO3leftJaco(phi)
    # print("temp1.shape", temp1.shape)
    # print("temp1 \n", temp1)
    # import math
    # Jaco_true = torch.eye(3).to(phi.device)
    # ad_mult = torch.eye(3).to(phi.device)
    # for i in range(1,20):
    #     ad_mult = ad_mult @ so3hat(phi)
    #     Jaco_true = Jaco_true + ad_mult / math.factorial(i+1)
    # print("Jaco_true \n", Jaco_true)
    ## debug
    xi = torch.randn(3,)
    dd = torch.randn(3,)
    tmp1 = SO3exp(xi) @ so3hat(SO3leftJacoInv(xi) @ dd)
    tmp2 = so3hat(SO3leftJacoInv(xi) @ dd) @ SO3exp(xi)
    print("tmp1: ", tmp1)
    print("tmp2: ", tmp2)

    ## test SEn3leftJaco
    
    xi = torch.tensor([[torch.pi/3,0,0,1,0,0],[torch.pi,0,0,1,0,0],[1e-4,0,0,1,0,0],[0,0,torch.pi/4,1,0,0]]).to(device).unsqueeze(0)
    xi = xi.repeat(2,1,1)
    xi = torch.randn(2,4,9).to(device)
    # print("phi.shape", xi.shape)
    Jaco_SE3_true = torch.eye(xi.shape[-1]).expand(xi.shape[:-1]+(xi.shape[-1],xi.shape[-1])).clone().to(xi.device)
    ad_mult = Jaco_SE3_true.clone()
    for i in range(1,20):
        ad_mult = ad_mult @ ad_sen3(xi)
        Jaco_SE3_true += ad_mult / math.factorial(i+1)
    temp1 = SEn3leftJaco(xi)
    error = torch.norm(temp1 - Jaco_SE3_true)
    if error > 1e-6:
        raise ValueError("error: ", error)
    else:
        print("SEn3leftJaco test passed, error: ", error)

    ## test SEn3leftJaco_inv
    xi = torch.randn(2,4,9).to(device)
    temp1 = SEn3leftJaco_inv(xi)
    temp2 = torch.matmul(temp1, SEn3leftJaco(xi))
    error = torch.norm(temp2 - torch.eye(xi.shape[-1]).to(xi.device))
    if error > 1e-6:
        raise ValueError("error: ", error)
    else:
        print("SEn3leftJaco_inv test passed, error: ", error)

    # test SEn3exp
    xi = torch.randn(2,4,6).to(device)
    temp1 = SEn3exp(xi)
    temp2 = torch.linalg.matrix_exp(sen3hat(xi))

    error = torch.norm(temp1 - temp2)
    if error > 1e-6:
        raise ValueError("error: ", error)
    else:
        print("SEn3exp test passed, error: ", error)

    # test SEn3log
    xi_log = SEn3log(temp1)
    temp3 = SEn3exp(xi_log)
    error = torch.norm(temp1 - temp3)
    if error > 1e-6:
        raise ValueError("error: ", error)
    else:
        print("SEn3log test passed, error: ", error)

    # test quat_to_SO3
    from scipy.spatial.transform import Rotation as scipy_rot
    import numpy as np
    q_numpy = np.array([[0,0,np.sin(np.pi/4),np.cos(np.pi/4)]])
    q  = scipy_rot.from_quat(q_numpy) # x, y, z, w

    R_true = q.as_matrix()
    R_true = torch.tensor(R_true).to(device)
    R_test = quat_to_SO3(torch.tensor(q_numpy).to(device), order = "xyzw")
    error = torch.norm(R_true - R_test)
    if error > 1e-6:
        raise ValueError("error: ", error)
    else:
        print("quat_to_SO3 test passed, error: ", error)

    q0 = torch.tensor([[0,0,0,1.]]).to(device)
    q1 = torch.tensor([[0,0,0,1.]]).to(device)
    qt = quat_slerp(q0, q1, torch.tensor([0.5]).to(device))
    print("qt: ", qt)

    # test SEn3fromSO3Rn
    R = torch.eye(3).to(device)
    x = torch.arange(6)
    X = SEn3fromSO3Rn(R, x)
    print("X: \n", X)
    print("---------------------------------")
    


    

