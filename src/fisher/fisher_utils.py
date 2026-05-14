from os.path import abspath, dirname, join
import sys
sys.path.append(dirname(abspath(__file__)))
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch_norm_factor
import between_bingham_fisher as bbf
import bingham_utils
import torch.nn.functional as F

def vmf_loss(net_out, R, overreg=1.05):
    A = net_out.view(-1, 3, 3)
    if R == None:
        return batch_torch_A_to_R(A)
    else:
        R = R.view(-1,3,3)
        loss_v = KL_Fisher(A, R, overreg=overreg)
        Rest = batch_torch_A_to_R(A)
        return loss_v, Rest


def KL_Fisher(A, R, overreg=1.05):
    """
    @param A: (b, 3, 3)
    @param R: (b, 3, 3)
    We find torch.svd() on cpu much faster than that on gpu in our case, so we apply svd operation on cpu.
    """
    A, R = A.cpu(), R.cpu()
    U, S, V = torch.svd(A)
    with torch.no_grad():  # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
        s3sign = torch.det(torch.matmul(U, V.transpose(1, 2)))
    S_sign = torch.cat((S[:, :2], S[:, 2:] * s3sign[:, None]), -1)
    log_normalizer = torch_norm_factor.logC_F(S_sign)
    log_exponent = -torch.matmul(A.view(-1, 1, 9), R.view(-1, 9, 1)).view(-1)
    log_nll = log_exponent + overreg * log_normalizer
    log_nll = log_nll.cuda()
    return log_nll


def batch_torch_A_to_R(A):
    A = A.cpu()
    A = A.reshape(-1, 3, 3)
    U, S, V = torch.svd(A)
    with torch.no_grad():  # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
        s3sign = torch.det(torch.matmul(U, V.transpose(1, 2)))
    U = torch.cat((U[:, :, :2], U[:, :, 2:] * s3sign[:, None][:, None]), -1)
    R = torch.matmul(U, V.transpose(1, 2))
    R = R.cuda()
    return R


def fisher_log_pdf(A, R):
    """
    @param A: (b, 3, 3)
    @param R: (b, 3, 3)
    """
    A, R = A.cpu(), R.cpu()
    U, S, V = torch.svd(A)
    with torch.no_grad():  # sign can only change if the 3rd component of the svd is 0, then the sign does not matter
        s3sign = torch.det(torch.matmul(U, V.transpose(1, 2)))
    S_sign = torch.cat((S[:, :2], S[:, 2:] * s3sign[:, None]), -1)
    log_normalizer = torch_norm_factor.logC_F(S_sign)
    log_exponent = torch.matmul(A.view(-1, 1, 9), R.view(-1, 9, 1)).view(-1)

    logp = -log_normalizer + log_exponent
    logp = logp.cuda()

    return logp


def fisher_entropy(A):
    """
    @param A: (b, 9) or (b, 3, 3)
    @return entropy: (b, )
    """
    A = A.reshape(-1, 3, 3)
    V, Lam = bbf.A_to_V_Lam(A)
    VB, LamB = bbf.convert_bingham_convention(V, Lam)
    # appr F
    entropy = bingham_utils.bingham_entropy(LamB)
    entropy = entropy - torch.tensor([np.log(2 * np.pi**2)]).float().to(entropy.device)
    return entropy


def fisher_CE(A1, A2):
    """
    A1 is gt
    A2 is prediction
    """
    A1 = A1.reshape(-1, 3, 3)
    A2 = A2.reshape(-1, 3, 3)
    V1, Lam1 = bbf.A_to_V_Lam(A1)
    V2, Lam2 = bbf.A_to_V_Lam(A2)
    VB1, LamB1 = bbf.convert_bingham_convention(V1, Lam1)
    VB2, LamB2 = bbf.convert_bingham_convention(V2, Lam2)
    CE = bingham_utils.bingham_CE(VB1, LamB1, VB2, LamB2)
    CE = CE - torch.tensor([np.log(2 * np.pi**2)]).float().to(CE.device)

    assert not torch.isnan(CE).any() and not torch.isinf(CE).any()
    return CE


####################### From omega, predicting rotation velocity

def hat_so3(w):  # (B,3) -> (B,3,3)
    B = w.shape[0]
    wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
    O = torch.zeros((B, 3, 3), device=w.device, dtype=w.dtype)
    O[:, 0, 1] = -wz
    O[:, 0, 2] =  wy
    O[:, 1, 0] =  wz
    O[:, 1, 2] = -wx
    O[:, 2, 0] = -wy
    O[:, 2, 1] =  wx
    return O

def so3_exp(phi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor: # Rodrigues exponential map.
    """
        phi: (B,3) rotation vector (axis * angle), in radians
        return: (B,3,3) rotation matrices
    """
    B = phi.shape[0]
    theta = torch.linalg.norm(phi, dim=1, keepdim=True)  # (B,1)

    I = torch.eye(3, device=phi.device, dtype=phi.dtype).unsqueeze(0).repeat(B, 1, 1)
    # avoid divide-by-zero
    a = phi / (theta + eps)  # (B,3) unit axis
    K = hat_so3(a)           # (B,3,3)

    theta2 = theta * theta

    # stable Taylor for small angles
    sin_t_over_t = torch.where(
        theta < 1e-4,
        1 - theta2 / 6 + theta2 * theta2 / 120,
        torch.sin(theta) / (theta + eps),
    )
    one_minus_cos_over_t2 = torch.where(
        theta < 1e-4,
        0.5 - theta2 / 24 + theta2 * theta2 / 720,
        (1 - torch.cos(theta)) / (theta2 + eps),
    )

    sin_t_over_t = sin_t_over_t.view(B, 1, 1)
    one_minus_cos_over_t2 = one_minus_cos_over_t2.view(B, 1, 1)

    R = I + sin_t_over_t * K + one_minus_cos_over_t2 * (K @ K)
    return R

def vmf_loss_omega_k(net_out, R_gt, dt, overreg=1.05, k_min=0.0):
    """
        net_out: (B,4) => omega(3) + k_raw(1)
        R_gt: (B,3,3) or None
        dt: (B,) or (B,1) seconds
    """
    B = net_out.shape[0]
    omega = net_out[:, :3].contiguous()              # (B,3)
    k_raw = net_out[:, 3:4].contiguous()             # (B,1)
    kappa = F.softplus(k_raw) + k_min                # (B,1)

    if dt.ndim == 1:
        dt = dt[:, None]
    dt = dt.view(B, 1)

    phi = omega * dt                                 # (B,3) radians
    R_mode = so3_exp(phi)                            # (B,3,3)
    A = kappa.view(B, 1, 1) * R_mode                 # (B,3,3)

    if R_gt is None:
        return R_mode
    else:
        R_gt = R_gt.view(B, 3, 3)
        loss_v = KL_Fisher(A, R_gt, overreg=overreg)
        return loss_v, R_mode