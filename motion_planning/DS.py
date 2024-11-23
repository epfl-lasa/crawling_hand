import numpy as np


class linear_system:
    def __init__(self, attractor, A=None, max_vel=2, scale_vel=2, eps=1e-8, dt=0.001):

        self.dim = len(attractor)
        if A is None:
            A = np.eye(self.dim) * (-1)

        self.A = A * scale_vel
        self.attractor = attractor
        self.max_vel = max_vel

        # self.scale_vel = scale_vel
        self.eps = eps
        self.dt = 0.001

    def eval(self, q: np.ndarray, k=1) -> np.ndarray:
        A = self.A / (np.linalg.norm(q - self.attractor) + self.eps)
        dq = self.A.dot((q - self.attractor)**k)
        if self.max_vel is not None:
            if np.linalg.norm(dq) > self.max_vel:
                dq = dq / np.linalg.norm(dq) * self.max_vel

        return dq


# def get_independent_basis(vector: np.ndarray) -> np.ndarray:
#     assert np.linalg.norm(vector) != 0
#     dim = len(vector)
#     E_matrix = np.zeros((dim, dim))
#     a0 = np.nonzero(vector)[0][0]  # the first nonzero component in vector
#     E_matrix[:, 0] = vector
#     for i in range(1, dim):
#         E_matrix[0, i] = - vector[i]
#         E_matrix[i, i] = a0
#     return E_matrix


class Modulation:
    def __init__(self, dim):
        self.dim = dim
        self.E = np.zeros((dim, dim))
        self.D = np.zeros((dim, dim))

    def get_M(self, vector: np.ndarray, gamma: float, dq=None, rho: float = 1.) -> np.ndarray:
        a0 = vector[np.nonzero(vector)[0][0]]  # the first nonzero component in vector (gradient)
        self.E[:, 0] = vector

        tmp = np.power(np.abs(gamma), 1 / rho)
        # solve the tail-effect
        if dq is None:
            lambda_0 = 1 - 1 / tmp
        else:
            if np.dot(vector, dq) >= 0:
                lambda_0 = 1
            else:
                lambda_0 = 1 - 1 / tmp
        lambda_1 = 1 + 1 / tmp

        self.D[0, 0] = lambda_0
        for i in range(1, self.dim):
            if self.dim in [2,4]:
                self.E[0, i] = - vector[i]
                self.E[i, i] = a0
            else:
                self.E[0, i] = - vector[0, i]
                self.E[i, i] = a0[i]

            self.D[i, i] = lambda_1

        return self.E @ self.D @ np.linalg.inv(self.E)








