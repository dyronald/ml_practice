import numpy as np


class Layer:
    def __init__(
            self,
            num_inputs,
            num_nodes,
            alpha = 1,
            ):
        self.thetas = np.random.rand(num_nodes, num_inputs)
        self.deltas = np.zeros(self.thetas.shape)
        self.alpha = alpha
        self.m = 0

    def activation(self, x):
        self.input = x
        z = np.sum(x * self.thetas, axis=1)
        self.a = 1 / (1.0 + (np.e ** (-z)))
        return self.a

    def calc_error(self, next_layer):
        f_prime = self.a * (1 - self.a)
        self.error = (np.matmul(next_layer.error.reshape(1, -1), next_layer.thetas))
        self.error = self.error * f_prime
        return self.error

    def accumulate_delta(self):
        delta = np.matmul(self.error.T, self.input.reshape(1, len(self.input)))
        self.deltas += delta
        self.m += self.error.shape[1]

        return self.deltas

    def step_theta(self):
        self.thetas += (self.alpha/self.m * self.deltas)

        self.deltas = np.zeros(self.thetas.shape)
        self.m = 0


class OutputLayer(Layer):
    def calc_error(self, label):
        f_prime = self.a * (1 - self.a)
        self.error = ((label - self.a) * f_prime).reshape(1, -1)
        return self.error

