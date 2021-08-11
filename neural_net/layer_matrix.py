import numpy as np


class Layer:
    def __init__(
            self,
            num_inputs=0,
            num_nodes=0,
            initial_theta=None,
            alpha = 1,
            ):
        if initial_theta is None:
            rng = np.random.default_rng()
            self.theta_mtx = rng.uniform(
                low=-1.0,
                high=1.0,
                size=(num_nodes, num_inputs))
        else:
            self.theta_mtx = initial_theta.copy()

        self.deltas = np.zeros(self.theta_mtx.shape)
        self.alpha = alpha
        self.m = 0

    def thetas(self):
        return self.theta_mtx.copy()

    def activation(self, x):
        self.input = x
        z = np.sum(x * self.theta_mtx, axis=1)
        self.a = 1 / (1.0 + (np.e ** (-z)))
        return self.a

    def calc_error(self, next_layer):
        f_prime = self.a * (1 - self.a)
        self.error = (np.matmul(next_layer.error.reshape(1, -1), next_layer.thetas()))
        self.error = self.error * f_prime
        return self.error

    def accumulate_delta(self):
        delta = np.matmul(self.error.T, self.input.reshape(1, -1))
        self.deltas += delta
        self.m += 1

        return self.deltas

    def step_theta(self):
        self.theta_mtx += (self.alpha/self.m * self.deltas)

        self.deltas = np.zeros(self.theta_mtx.shape)
        self.m = 0


class OutputLayer(Layer):
    def calc_error(self, label):
        f_prime = self.a * (1 - self.a)
        self.error = ((label - self.a) * f_prime).reshape(1, -1)
        return self.error

