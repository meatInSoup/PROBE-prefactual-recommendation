import numpy as np
import scipy.special as special
import torch
from scipy.stats import entropy

from scipy.linalg import sqrtm


def compute_M(x_i, x_j, x_0):
    M = x_i @ x_i.T - x_j @ x_j.T + (x_j - x_i) @ x_0.T + x_0 @ (x_j - x_i).T
    return M


def compute_mahalanobis(x, x_0, A):
    dist = (x - x_0).T @ A @ (x - x_0)
    return dist.item()


def transpose_kron(d):
    # iden = np.eye(d)
    # sum = None
    # for i in range(d):
    #     for j in range(d):
    #         e_i = iden[:, i].reshape(-1, 1)
    #         e_j = iden[:, j].reshape(-1, 1)
    #         kron_1 = np.kron(e_j, e_i)
    #         kron_2 = np.kron(e_i, e_j)
    #         prod = kron_1 @ kron_2.T
    #         sum = prod if sum is None else sum + prod
    # return sum
    iden = np.eye(d)
    kron_1 = np.kron(iden, iden)
    return kron_1 + kron_1.T


def compute_covariance(z_i, z_j, Sigma, m):
    # d = z_i.shape[0]
    # z_j_kron = np.kron(z_j, z_j)
    # z_i_kron = np.kron(z_i, z_i)
    # Sigma_kron = np.kron(Sigma, Sigma)
    # transpose = transpose_kron(d)
    # const = np.eye(d**2) + transpose
    # cov = z_i_kron.T @ Sigma_kron @ const @ z_j_kron
    # return (m * cov).item()
    
    z_i, z_j = z_i.reshape(-1, 1), z_j.reshape(-1, 1)

    # Compute quadratic forms
    Sigma_z_j = Sigma @ z_j
    Sigma_z_i = Sigma @ z_i
    
    # Compute covariance
    covariance = ((z_i.T @ Sigma_z_j) ** 2).item()

    return covariance


def compute_rho(x_i, x_j, x_0, Sigma):
    z_i = (x_i - x_0).reshape(-1, 1)
    z_j = (x_j - x_0).reshape(-1, 1)
    
    # Compute quadratic forms
    Sigma_z_j = Sigma @ z_j
    Sigma_z_i = Sigma @ z_i
    
    covariance = ((z_i.T @ Sigma_z_j) ** 2).item()

    # covariance_ij = compute_covariance(z_i, z_j, Sigma, m)
    variance_i = (z_i.T @ Sigma_z_i).item()
    variance_j = (z_j.T @ Sigma_z_j).item()

    return covariance / (variance_i * variance_j)


def params_McKayII(alpha, beta_x, beta_y, rho):
    rho = np.clip(rho, -0.99, 0.99)
    a = alpha - 1 / 2
    # b = (
    #     2
    #     * beta_x
    #     * beta_y
    #     * (1 - rho)
    #     / np.sqrt((beta_x - beta_y) ** 2 + 4 * beta_x * beta_y * (1 - rho))
    # )
    
    c = -(beta_x - beta_y) / np.sqrt(
        (beta_x - beta_y) ** 2 + 4 * beta_x * beta_y * (1 - rho)
    )
    return a, c


def zero_cdf_McKayII(a, c):
    epsilon1 = max(1 - c, 1e-3)
    epsilon2 = max(1 + c, 1e-6)
    first_term = epsilon2 / epsilon1
    first_term = np.exp((a + 1/2) * np.log(first_term))
    # first_term = first_term ** (a + 1 / 2)
    second_term = (special.gamma(2 * a + 1) * special.gamma(1)) / (
        special.gamma(a + 3 / 2) * special.gamma(a + 1 / 2)
    )
    third_term = special.hyp2f1(2 * a + 1, a + 1 / 2, a + 3 / 2, - epsilon2 / epsilon1)
    return first_term * second_term * third_term
    
    # epsilon = max(1 - c, 1e-12)

    # if epsilon < 1e-6:  # Handle c ≈ 1 separately with smooth approximation
    #     # first_term = np.sqrt(2 / epsilon) * special.gamma(a+3/2) / special.gamma(a+1/2)
    #     first_term = np.exp((a + 1/2) * np.log(2 / epsilon)) * special.gamma(a+3/2) / special.gamma(a+1/2)
    #     third_term = 1  # Since hypergeometric function is approximated
    # else:
    #     first_term = ((1 + c) / (1 - c)) ** (a + 1/2)
    #     third_term = special.hyp2f1(2*a+1, a+1/2, a+3/2, -(1 + c) / (1 - c))

    # second_term = (special.gamma(2 * a + 1) * special.gamma(1)) / (
    #     special.gamma(a + 3 / 2) * special.gamma(a + 1 / 2)
    # )

    return first_term * second_term * third_term


def entropy_McKay(x_i, x_j, x_0, Sigma, m):
    # determine the parameters of 2 gamma random var
    alpha = m / 2

    sigma_i_sq = compute_mahalanobis(x_i, x_0, Sigma)
    beta_i = 2 * sigma_i_sq

    sigma_j_sq = compute_mahalanobis(x_j, x_0, Sigma)
    beta_j = 2 * sigma_j_sq

    # compute rho
    rho = compute_rho(x_i, x_j, x_0, Sigma)

    # determine the parameters of McKay distribution
    a, c = params_McKayII(alpha, beta_i, beta_j, rho)

    # compute gamma =  P[<A, M> <=0]
    gamma = zero_cdf_McKayII(a, c)

    # compute the entropy
    # entropy = -gamma * np.log2(gamma) - (1 - gamma) * np.log2((1 - gamma))
    pk = [gamma, 1 - gamma]
    ent = entropy(pk, base=2)
    return ent


def expected_graph_cost(post_Sigma, post_m, x_i, x_j):
    x_i = x_i.reshape(-1, 1)
    x_j = x_j.reshape(-1, 1)
    M = (x_i - x_j) @ (x_i - x_j).T
    expected_A = post_Sigma * post_m
    expected_inner = np.trace(expected_A @ M)
    return expected_inner.item()


def generate_A_0(d, scl=0.4):
    A = np.random.normal(loc=0, scale=scl, size=(d, d))
    A_0 = A @ A.T
    return A_0


def evaluate_cost_diag(x_0, x, A):
    diag_A = np.eye(A.shape[0])
    np.fill_diagonal(diag_A, A.diagonal())
    cost = (x - x_0).T @ diag_A @ (x - x_0)
    return np.sqrt(cost.item())


def multivariate_digamma(z, d):
    digamma = [special.digamma(z + (1 - i) / 2) for i in range(1, d + 1)]
    digamma = np.array(digamma)
    return np.sum(digamma)


def l1_norm_diag(recourse, x_0, A_0, diag=True):
    if diag:
        dim = A_0.shape[0]
        A = np.eye(dim)
        diag_A_0 = sqrtm(A_0).diagonal()
        np.fill_diagonal(A, diag_A_0)
    else:
        A = A_0

    argument = A @ x_0 - A @ recourse
    cost = np.linalg.norm(argument, ord=1)

    return cost
