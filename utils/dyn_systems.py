import numpy as np
from numpy.random import multivariate_normal as gauss


class DynSystem():

    def __init__(self, A, mu):

        self.A = A
        self.mu = mu

    def __call__(self, x):
        """
        Computes the new state of teh system
        :param x: current state
        :return: new state
        """
        diff = self.mu - x
        new_x = np.matmul(self.A, diff) + x
        return new_x

    def __str__(self):
        return f"Matrix A: {self.A}\n Threshold mu: {self.mu}"

    def evolve(self, x0, T):
        """
        Evolution of the system for T time steps from the initial state included.
        :param x0: initial state
        :param T: number of time steps to consider
        :return: list of system state [x(0),...,x(T-1)]
        """

        states = []
        curr_state = x0
        for t in range(T):
            states.append(curr_state)
            curr_state = self.__call__(curr_state)

        return states

    def evolve_stoch(self, x0, T, mean, cov):
        """
        Evolution of the system for T time steps from the initial state included.

        :param x0: initial state
        :param T: number of time steps to consider
        :param mean: mean of the multivariate normal distribution modeling the noise
        :param cov: covariance of the multivariate normal distribution modeling the noise
        :return: list of system state [x(0),...,x(T-1)]
        """

        states = []
        noises = []
        curr_state = x0
        for t in range(T):
            noise = gauss(mean, cov, 1)[0]
            states.append(curr_state)
            noises.append(noise)
            curr_state = self.__call__(curr_state) + noise

        return states, np.array(noises)


class DynSystemMin(DynSystem):

    def __call__(self, x):
        """
        Computes the new state of teh system
        :param x: current state
        :return: new state
        """
        diff = self.mu - x
        minimum = np.min((np.zeros_like(diff), diff), axis=0)
        new_x = np.matmul(self.A, minimum) + x
        return new_x

    def __str__(self):
        return f"Matrix A: {self.A}\n Threshold mu: {self.mu} with min"


def diag_matrix(eigens):
    dim = eigens.shape[0]
    A = np.zeros((dim, dim))
    np.fill_diagonal(A, eigens)
    return A
