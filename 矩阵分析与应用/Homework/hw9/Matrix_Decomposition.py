import numpy as np

np.set_printoptions(suppress=True)


def solve(A, b, method="LU"):
    """
    :param A: The coefficient matrix
    :param b: The right-hand side vector
    :param method: decomposition method
    :return: The solution vector
    """
    if method == "LU":
        L, U, P = LU_decomposition(A)
        y = np.linalg.solve(L, P @ b)
        x = np.linalg.solve(U, y)
    elif method == "QR":
        Q, R = Gram_Schmidt_QR_decomposition(A)
        y = np.linalg.solve(Q, b)
        x = np.linalg.solve(R, y)
    elif method == "HR":
        P, T = Householder_reduction(A)
        x = np.linalg.solve(T, P @ b)
    elif method == "GR":
        P, T = Givens_reduction(A)
        x = np.linalg.solve(T, P @ b)
    elif method == "URV":
        U, R, V_T = URV_decomposition(A)
        y = np.linalg.solve(R, U.T @ b)
        x = V_T.T @ y
    else:
        raise ValueError("The method is not supported!")
    return x


def determinant(A, method="LU"):
    """
    :param A: n*n matrix
    :param method: decomposition method
    :return: determinant of A
    """
    if method == "LU":
        L, U, P = LU_decomposition(A)
        det = np.linalg.det(P) * np.prod(np.diag(U))
    elif method == "QR":
        Q, R = Gram_Schmidt_QR_decomposition(A)
        det = np.linalg.det(Q) * np.prod(np.diag(R))
    elif method == "HR":
        P, T = Householder_reduction(A)
        det = np.linalg.det(P) * np.prod(np.diag(T))
    elif method == "GR":
        P, T = Givens_reduction(A)
        det = np.linalg.det(P) * np.prod(np.diag(T))
    elif method == "URV":
        U, R, V_T = URV_decomposition(A)
        det = np.linalg.det(U) * np.linalg.det(R) * np.linalg.det(V_T)
    else:
        raise ValueError("The method is not supported!")
    return det


# LU decomposition
def LU_decomposition(A):
    """
    :param A: n*n matrix
    :return: L, U, P
    """
    _A = A.copy()
    if _A.shape[0] != _A.shape[1]:
        print("The matrix is not square! No LU decomposition!")
        return None
    n = _A.shape[0]
    det = np.linalg.det(_A)
    if det == 0:
        print("The matrix is singular! No LU decomposition!")
        return

    P = np.eye(n)

    for i in range(n):
        pivot_idx = np.argmax(np.abs(_A[i:, i])) + i
        _A[i], _A[pivot_idx] = _A[pivot_idx].copy(), _A[i].copy()
        p_i = np.eye(n)
        p_i[i], p_i[pivot_idx] = p_i[pivot_idx].copy(), p_i[i].copy()
        P = p_i @ P
        for j in range(i + 1, n):
            _A[j, i] = _A[j, i] / _A[i, i]
            _A[j, i + 1:n] = _A[j, i + 1:n] - _A[j, i] * _A[i, i + 1:n]

    L = np.tril(_A, -1) + np.eye(n)
    U = np.triu(_A)
    return L, U, P


# Gram-Schmidt QR decomposition
def Gram_Schmidt_QR_decomposition(A):
    """
    :param A: m*n matrix with independent columns
    :return: Q, R
    """
    if np.linalg.matrix_rank(A) != A.shape[1]:
        print("The matrix's columns are linear dependent! No QR decomposition!")
        return

    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for i in range(n):
        for j in range(0, i):
            R[j, i] = np.dot(Q[:, j], A[:, i])
        q_i = A[:, i] - np.dot(Q[:, 0:i], R[0:i, i])
        R[i, i] = np.linalg.norm(q_i)
        Q[:, i] = q_i / R[i, i]

    return Q, R


# Householder reduction
def Householder_reduction(A):
    """
    :param A: m*n matrix
    :return: P, T
    """
    m, n = A.shape
    P = np.eye(m)
    T = A.copy()

    for i in range(m - 1):
        x = T[i:, i]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x - e
        u = u / np.linalg.norm(u)
        R = np.eye(m)
        R[i:, i:] -= 2 * np.outer(u, u)
        P = R @ P
        T = R @ T

    return P, T


# Givens reduction
def Givens_reduction(A):
    """
    :param A: m*n matrix
    :return: P, T
    """
    m, n = A.shape
    P = np.eye(m)
    T = A.copy()

    for i in range(n):
        for j in range(i + 1, m):
            G = np.eye(m)
            G[i, i] = T[i, i] / np.sqrt(T[i, i] ** 2 + T[j, i] ** 2)
            G[j, j] = G[i, i]
            G[i, j] = T[j, i] / np.sqrt(T[i, i] ** 2 + T[j, i] ** 2)
            G[j, i] = -G[i, j]
            P = G @ P
            T = G @ T

    return P, T


# URV decomposition
def URV_decomposition(A):
    """
    :param A: m*n matrix
    :return: U, R, V_T
    """
    P, B = Householder_reduction(A)
    Q, T = Householder_reduction(B.T)
    return P.T, T.T, Q


# Determine whether the matrices are equal with a certain degree of precision
def is_equal(A, B, eps=1e-8):
    """
    :param A: matrix A
    :param B: matrix B
    :param eps: degree of precision
    :return: whether A and B are equal
    """
    return np.linalg.norm(A - B) < eps


def decomposition(A, method="LU"):
    """
    :param A: matrix A to be decomposed
    :param method: decomposition method
    :return: decomposition result
    """
    if method == "LU":
        ans = LU_decomposition(A)
        if ans:
            L, U, P = ans
            print("LU decomposition:")
            print("L:\n", L)
            print("U:\n", U)
            print("P:\n", P)
            print("A:\n", A)
            print("P * A:\n", P @ A)
            print("L * U:\n", L @ U)
            if is_equal(P @ A, L @ U):
                print("PA=LU, LU decomposition is correct!")
            else:
                print("PA!=LU, LU decomposition is wrong!")
            return L, U, P
    elif method == "QR":
        ans = Gram_Schmidt_QR_decomposition(A)
        if ans:
            Q, R = ans
            print("Gram-Schmidt QR decomposition:")
            print("Q:\n", Q)
            print("R:\n", R)
            print("A:\n", A)
            print("Q * R:\n", np.dot(Q, R))
            if is_equal(np.dot(Q, R), A):
                print("A=QR, QR decomposition is correct!")
            else:
                print("A!=QR, QR decomposition is wrong!")
            return Q, R
    elif method == "HR":
        P, T = Householder_reduction(A)
        print("Householder reduction:")
        print("P:\n", P)
        print("T:\n", T)
        print("A:\n", A)
        print("P.T * T:\n", P.T @ T)
        if is_equal(P @ A, T):
            print("PA=T, Householder reduction is correct!")
        else:
            print("PA!=T, Householder reduction is wrong!")
        return P, T
    elif method == "GR":
        P, T = Givens_reduction(A)
        print("Givens reduction:")
        print("P:\n", P)
        print("T:\n", T)
        print("A:\n", A)
        print("P.T * T:\n", P.T @ T)
        if is_equal(P @ A, T):
            print("PA=T, Givens reduction is correct!")
        else:
            print("PA!=T, Givens reduction is wrong!")
        return P, T
    elif method == "URV":
        U, R, V_T = URV_decomposition(A)
        print("URV decomposition:")
        print("U:\n", U)
        print("R:\n", R)
        print("V.T:\n", V_T)
        print("A:\n", A)
        print("U * R * V.T:\n", U @ R @ V_T)
        if is_equal(U @ R @ V_T, A):
            print("A=URV.T, URV decomposition is correct!")
        else:
            print("A!=URV.T, URV decomposition is wrong!")
        return U, R, V_T
    else:
        print("Wrong decomposition method!")
        return


if __name__ == '__main__':
    decompositions = ["LU", "QR", "HR", "GR", "URV"]
    A = np.array([[1, 2, -3, 4], [4, 8, 12, -8], [2, 3, 2, 1], [-3, -1, 1, -4]], dtype=float)
    b = np.array([3, 60, 1, 5], dtype=float)
    print("A:\n", A)
    print("b:\n", b.T, end='\n\n')

    for method in decompositions:
        decomposition(A, method)
        print("Determinant of A:\n", determinant(A, method))
        print("Solution of {}:\n".format(method), solve(A, b, method))
        print("################################################################################")
