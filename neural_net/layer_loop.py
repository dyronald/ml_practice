import numpy as np


class LogisticNode:
    def __init__(
            self,
            num_features,
            ):
        rng = np.random.default_rng()
        self.theta = rng.uniform(
            low=-1.0,
            high=1.0,
            size=(num_features,))

    def activation(self, x):
        z = np.sum(self.theta * x)
        a = 1 / (1.0 + (np.e ** (-z)))
        return a


class Layer:
    def __init__(
            self,
            nodes,
            alpha=1,
            ):
        self.nodes = nodes
        self.deltas = np.zeros((len(self.nodes), self.nodes[0].theta.shape[0]))
        self.m = 0
        self.alpha = alpha

    def activation(self, a):
        assert(a.shape == self.nodes[0].theta.shape)
        self.input = a

        # Apply input to each layer
        self.a = np.array([n.activation(a) for n in self.nodes])
        return self.a

    def thetas(self):
        return np.array([n.theta for n in self.nodes])

    def calc_error(self, next_layer):
        f_prime = self.a * (1 - self.a)
        self.error = (np.matmul(next_layer.error.reshape(1, -1), next_layer.thetas()))
        self.error = self.error * f_prime
        return self.error

    def accumulate_delta(self):
        e = self.error
        e.shape = (-1, 1) # need error to as 2-dimensional matrix

        delta = np.matmul(e, self.input.reshape(1, -1))
        self.deltas += delta
        self.m += 1

        # delta is a 2-d matrix (num layers, num inputs)
        return self.deltas

    def step_theta(self):
        # Apply each delta to its corresponding node
        for i in range(len(self.nodes)):
            old = self.nodes[i].theta
            self.nodes[i].theta = old + (self.alpha/self.m * self.deltas[i])

        self.deltas = np.zeros((len(self.nodes), self.nodes[0].theta.shape[0]))
        self.m = 0


class OutputLayer(Layer):
    def calc_error(self, label):
        f_prime = self.a * (1 - self.a)
        self.error = (label - self.a) * f_prime
        return self.error
