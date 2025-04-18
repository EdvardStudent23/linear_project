import numpy as np


def gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    This function orthogonalizes an input array.
    """
    n, m = A.shape 
    Q = np.zeros((n, m))
    R = np.zeros((m, n))

    for j in range(m):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            v = v - R[i, j] * Q[:, i] 
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    
    return Q, R


def qr_algorithm(
    A: np.ndarray, numx_iterations: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function computes the eigenvalues and eigenvectors of a matrix using the QR algorithm.
"""
    n, _ = A.shape
    Q_total = np.eye(
        n
    )

    for _ in range(50):
        Q, R = gram_schmidt(A)
        A = R @ Q
        Q_total = Q_total @ Q

    eigenvalues = np.diag(A)

    return eigenvalues, Q_total


def custom_svd(A: np.ndarray):
    """
    Just custom svd
    """

    m, n = A.shape
    ATA = A.T @ A
    eigenvalues, V = qr_algorithm(ATA)
    singular_values = np.sqrt(np.abs(eigenvalues))


    Sigma_inv = np.diag([1/s if s > 1e-10 else 0 for s in singular_values])
    U = A @ V @ Sigma_inv


    idx = np.argsort(singular_values)[::-1]
    singular_values = singular_values[idx]
    U = U[:, idx]
    V = V[:, idx]

    r = len(singular_values)
    U = U[:, :r]
    Vt = V.T[:r, :]

    return U, singular_values, Vt

