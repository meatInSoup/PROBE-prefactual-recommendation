import numpy as np
import scipy.special as special
from scipy.linalg import cho_factor, cho_solve
from methods.probe import bayesian_utils


def is_positive_definite(matrix):
    """Check if a matrix is positive definite."""
    # try:
    #     np.linalg.cholesky(matrix)
    #     return True
    # except np.linalg.LinAlgError:
    #     return False
    
    eigenvalues = np.linalg.eigvalsh(matrix)  # eigvalsh is faster for symmetric matrices
    return np.all(eigenvalues > 0)

def matrix_projection(A, threshold=1e-3):
    """
    Project a symmetric matrix A into a space where its eigenvalues are larger than the threshold.

    Parameters:
    A (numpy.ndarray): A symmetric matrix.
    threshold (float): The minimum eigenvalue allowed.

    Returns:
    numpy.ndarray: The projected matrix.
    """
    # Check if A is symmetric
    # if not np.allclose(A, A.T):
    #     raise ValueError("Matrix A is not symmetric")

    eigenvalues, eigenvectors = np.linalg.eigh(A)
    eigenvalues_projected = np.maximum(eigenvalues, threshold)
    A_projected = eigenvectors @ np.diag(eigenvalues_projected) @ eigenvectors.T
    A_projected = (A_projected + A_projected.T) / 2  # Ensure symmetry
    np.fill_diagonal(A_projected, np.maximum(np.diagonal(A_projected), threshold))
    
    return A_projected

def gd_linesearch(
    response_list,
    prior_Sigma,
    prior_m,
    m,
    tau,
    d,
    iterations,
    lr,
    gamma,  # Step size decay factor (γ)
    eta,  # Parameter for line search condition (0 < η ≤ 0.5)
    beta=0.6,  # Momentum factor
    relative_error=1e-6,
):
    curr_Sigma = prior_Sigma
    lst_loss = []

    inv_prior_Sigma = np.linalg.inv(prior_Sigma)
    # X = inv_prior_Sigma
    X = prior_Sigma
    
    # Initialize momentum
    momentum = np.zeros_like(X)

    # Precompute the gradient and the loss for comparison
    loss = l_posterior(X, m, prior_Sigma, prior_m, response_list, tau)

    for i in range(iterations):
        # if i != 0 and i % 50 == 0:
        #     lr = lr * 0.8
        #     lr = max(lr, 0.2) 
        grad = gradient_l(X, response_list, prior_Sigma, prior_m, m, tau)
        # grad = grad / np.linalg.norm(grad, ord="fro")
        # momentum = beta * momentum + (1 - beta) * grad
        # X_next = matrix_projection(X - lr * grad)
        # loss_next = l_posterior(X_next, m, prior_Sigma, prior_m, response_list, tau)
        learning_rate = lr
        grad_norm_squared = np.linalg.norm(grad, "fro") ** 2
        while True:
            grad = gradient_l(X, response_list, prior_Sigma, prior_m, m, tau)
            # Momentum update: previous update influences current update
            # momentum = beta * momentum + (1 - beta) * grad
            X_next = X - learning_rate * grad

            loss_next = l_posterior(X_next, m, prior_Sigma, prior_m, response_list, tau)

            if (
                is_positive_definite(X_next)
                and loss_next
                <= loss - eta * learning_rate * grad_norm_squared
            ):
                break

            learning_rate *= gamma
        res = np.max(np.abs(X - X_next))

        # store the loss
        loss = loss_next
        lst_loss.append(loss)
        X = X_next

        # if res < relative_error:
        #     break

    curr_Sigma = np.linalg.inv(X)
    big_l_loss = big_l_posterior(loss, m, prior_m, d)
    return curr_Sigma, lst_loss, big_l_loss
    # return X, lst_loss, big_l_loss

def gd_diag(
    response_list,
    prior_Sigma,
    prior_m,
    m,
    tau,
    d,
    iterations,
    lr,
    gamma,  # Step size decay factor (γ)
    eta,  # Parameter for line search condition (0 < η ≤ 0.5)
    beta=0.6,  # Momentum factor
    relative_error=1e-6,
):
    curr_Sigma = prior_Sigma
    lst_loss = []

    inv_prior_Sigma = np.linalg.inv(prior_Sigma)
    X = prior_Sigma
    
    # Initialize momentum
    momentum = np.zeros_like(X)

    # Precompute the gradient and the loss for comparison
    loss = l_posterior(X, m, prior_Sigma, prior_m, response_list, tau)

    for i in range(iterations):
        grad = gradient_l(X, response_list, prior_Sigma, prior_m, m, tau)
        X_next = matrix_projection(X - lr * grad)
        X_next = np.diag(np.diag(X_next))
        loss_next = l_posterior(X_next, m, prior_Sigma, prior_m, response_list, tau)
        
        # store the loss
        loss = loss_next
        lst_loss.append(loss)
        X = X_next

        # if res < relative_error:
        #     break

    curr_Sigma = np.linalg.inv(X)
    big_l_loss = big_l_posterior(loss, m, prior_m, d)
    return curr_Sigma, lst_loss, big_l_loss
    # return X, lst_loss, big_l_loss
    


def compute_lambda_star(R_ij, z_i, z_j, X_inv):
    """Compute λ*_ij(X) based on the given formula."""
    # num = R_ij * (z_j.T @ X_inv @ z_j - z_i.T @ X_inv @ z_i)
    # denom = 2 * (
    #     (z_i.T @ X_inv @ z_i) * (z_j.T @ X_inv @ z_j) - (z_j.T @ X_inv @ z_i) ** 2
    # )
    # Compute necessary projections
    z_i_X_inv = X_inv @ z_i
    z_j_X_inv = X_inv @ z_j

    z_i_norm = z_i.T @ z_i_X_inv
    z_j_norm = z_j.T @ z_j_X_inv
    z_i_z_j = z_j.T @ z_i_X_inv

    denom = 2 * (z_i_norm * z_j_norm - z_i_z_j ** 2) + 1e-6
    lambda_star = (R_ij * (z_j_norm - z_i_norm)) / denom
    return lambda_star


def gradient_l(X, response_list, prior_Sigma, prior_m, m, tau):
    d = X.shape[0]
    # X_inv = np.linalg.inv(X)
    cho_L, lower = cho_factor(X)
    X_inv = cho_solve((cho_L, lower), np.eye(d))  # Efficient X^{-1}
    grad = -X_inv + prior_m / m * prior_Sigma
    # grad = -X_inv + m / prior_m * np.linalg.inv(prior_Sigma)

    for k in range(len(response_list)):
        response = response_list[k]
        z_i, z_j = response["vector_pair"]
        R_ij = response["R_ij"]
        M_ij = response["M_ij"]

        lambda_star = compute_lambda_star(R_ij, z_i, z_j, X_inv)

        trace_val = np.trace(M_ij @ X_inv)
        alignment  = R_ij * trace_val

        correction = R_ij * M_ij
        grad += tau * correction

        if alignment <= 0:
            # Use Sherman-Morrison-Woodbury formula for low-rank update
            # C = lambda_star * R_ij * M_ij

            # # Woodbury formula: (X - C)^{-1} ≈ X^{-1} + X^{-1} C X^{-1} (for small-rank updates)
            # X_C_inv = X_inv - X_inv @ C @ X_inv

            # correction = -X_C_inv + X_inv  # Efficient update (approximation)
            # grad += tau * correction
            
            correction = -np.linalg.inv(X - lambda_star * R_ij * M_ij) + X_inv
            grad += tau * correction
        
    return grad


def l_posterior(pos_X, pos_m, prior_Sigma, prior_m, response_list, tau):
    det_pos_X = np.linalg.det(pos_X)
    ell = -np.log(det_pos_X) + prior_m / pos_m * np.trace(prior_Sigma @ pos_X)
    X_inv = np.linalg.inv(pos_X)
    for k in range(len(response_list)):
        response = response_list[k]
        z_i, z_j = response["vector_pair"]
        R_ij = response["R_ij"]
        M_ij = response["M_ij"]

        lambda_star = compute_lambda_star(R_ij, z_i, z_j, X_inv)

        if R_ij * np.trace(M_ij @ X_inv) < 0:
            det_pos_XM = np.linalg.det(pos_X - lambda_star * R_ij * M_ij)
            ell += tau * (det_pos_X - det_pos_XM)

    return ell


def big_l_posterior(l_loss, pos_m, prior_m, d):
    multivariate_digamma = bayesian_utils.multivariate_digamma(prior_m / 2, d)
    l_loss = (pos_m / 2) * l_loss
    big_ell = (
        l_loss
        + special.multigammaln(pos_m / 2, d)
        + (prior_m - pos_m) / 2 * multivariate_digamma
    )
    return big_ell


def posterior_inference(
    set_m, response_list, prior_Sigma, prior_m, tau, d, iterations, lr, gamma=0.9, eta=0.1
):
    opt_Sigma, opt_m, opt_lst_loss = None, None, None
    opt_loss = np.inf

    for m in set_m:
        pos_Sigma, lst_loss, big_l_loss = gd_linesearch(
            response_list, prior_Sigma, prior_m, m, tau, d, iterations, lr, gamma, eta
        )

        if big_l_loss < opt_loss:
            opt_loss = big_l_loss
            opt_m = m
            opt_Sigma = pos_Sigma
            opt_lst_loss = lst_loss
    return opt_Sigma, opt_m, opt_lst_loss


def posterior_inference_diag(
    set_m, response_list, prior_Sigma, prior_m, tau, d, iterations, lr, gamma=0.9, eta=0.1
):
    opt_Sigma, opt_m, opt_lst_loss = None, None, None
    opt_loss = np.inf

    for m in set_m:
        pos_Sigma, lst_loss, big_l_loss = gd_diag(
            response_list, prior_Sigma, prior_m, m, tau, d, iterations, lr, gamma, eta
        )

        if big_l_loss < opt_loss:
            opt_loss = big_l_loss
            opt_m = m
            opt_Sigma = pos_Sigma
            opt_lst_loss = lst_loss
    return opt_Sigma, opt_m, opt_lst_loss
