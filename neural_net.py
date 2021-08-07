import numpy as np

class LogisticNode:
    def __init__(
            self,
            num_features,
            ):
        self.theta = np.random.rand(num_features)

    def activation(self, x):
        z = np.sum(self.theta * x)
        a = 1 / (1.0 + (np.e ** (-z)))
        return a

class OutputLayer:
    def __init__(
            self,
            nodes,
            ):
        self.nodes = nodes
        self.deltas = np.zeros(len(self.nodes), self.nodes[0].theta.shape[0])

    def activation(self, a):
        self.input = a
        self.a = np.array([n.activation(a) for n in self.nodes])
        return self.a

    def thetas(self):
        return np.array([n.theta for n in self.nodes])

    def error_term(self, label):
        f_prime = self.a * (1 - self.a)
        self.error = (label - self.a) * f_prime
        return self.error

    def accumulate_delta(self):
        delta = self.error * self.input.reshape(1, self.nodes[0].theta.shape[0])
        self.delta += delta

    def step_theta(self, delta):
        for d in delta:
            for i in range(len(self.nodes)):
                old = self.nodes[i].theta
                self.nodes[i].theta = old + (0.1 * d[i])

        self.deltas = np.zeros(len(self.nodes), self.nodes[0].theta.shape[0])


class Layer:
    def __init__(
            self,
            nodes
            ):
        self.nodes = nodes

    def activation(self, a):
        a = np.array([n.activation(a) for n in self.nodes])
        return a

    def thetas(self):
        return np.array([n.theta for n in self.nodes])

    def step_theta(self, delta):
        # print('...')
        for d in delta:
            for i in range(len(self.nodes)):
                # print(d[i])
                old = self.nodes[i].theta
                self.nodes[i].theta = old + (d[i])

class Network:
    def __init__(
            self, 
            layers,
            features,
            labels,
            ):
        self.layers = layers
        self.features = features
        self.labels = labels

    def cost(self, hyp):
        m = hyp.shape[0]
        diff_true = self.labels * np.log(hyp)
        diff_false = (1-self.labels) * np.log(1-hyp)
        return -1 / m * np.sum(diff_true + diff_false)

    def f_prime(self, a):
        return a * (1 - a)

    def hypothesis(self):
        predictions = []
        deltas = [[], []]
        errors = [[], []]
        
        for i in range(len(self.features)):
            a = self.features[i]
            activations = []
            for layer in self.layers:
                a = layer.activation(a)
                activations.append(a)
            hyp = a
            predictions.append(hyp)

            error = (self.labels[i] - hyp) * self.f_prime(hyp)
            delta = error * activations[-2].reshape(1, 4)
            errors[-1].append(error)
            deltas[-1].append(delta)

            error = error * self.layers[1].thetas() * self.f_prime(activations[-2])
            delta = np.matmul(error.T, self.features[i].reshape(1,3))
            errors[-2].append(error)
            deltas[-2].append(delta)

        return np.array(predictions), deltas

    def train(self, iterations=10):
        for _ in range(iterations):
            hyp, deltas = self.hypothesis()
            print(self.cost(hyp))

            for i in range(len(deltas)):
                self.layers[i].step_theta(deltas[i])


X = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 0],
    [1, 0, 0],
])
y = np.array([[0, 1, 1, 0, 0, 0]]).T
num_features = X.shape[1]
layers = [
    Layer(
        nodes = [
            LogisticNode(num_features=num_features,),
            LogisticNode(num_features=num_features,),
            LogisticNode(num_features=num_features,),
            LogisticNode(num_features=num_features,),
        ]),
    Layer(
        nodes = [
            LogisticNode(num_features=4,),
        ])
]
network = Network(
    layers=layers,
    features=X,
    labels=y,
)

network.train(1000)

hyp, deltas = network.hypothesis()
# for d in deltas[0]:
#     print(d)
# print('')
# print('')
# for d in deltas[1]:
#     print(d)

print('')
print(hyp)
