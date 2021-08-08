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
        self.deltas = np.zeros((len(self.nodes), self.nodes[0].theta.shape[0]))

    def activation(self, a):
        self.input = a
        self.a = np.array([n.activation(a) for n in self.nodes])
        return self.a

    def thetas(self):
        return np.array([n.theta for n in self.nodes])

    def calc_error(self, label):
        f_prime = self.a * (1 - self.a)
        self.error = (label - self.a) * f_prime
        return self.error

    def accumulate_delta(self):
        delta = self.error * self.input.reshape(1, self.nodes[0].theta.shape[0])
        self.deltas += delta

    def step_theta(self):
        for i in range(len(self.nodes)):
            old = self.nodes[i].theta
            self.nodes[i].theta = old + (0.1 * self.deltas[i])

        self.deltas = np.zeros((len(self.nodes), self.nodes[0].theta.shape[0]))


class Layer:
    def __init__(
            self,
            nodes
            ):
        self.nodes = nodes
        self.deltas = np.zeros((len(self.nodes), self.nodes[0].theta.shape[0]))

    def activation(self, a):
        self.input = a
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
        for e in self.error:
            e.shape = (-1, 1)
            delta = np.matmul(e, self.input.reshape(1, len(self.input)))
            self.deltas += delta

    def step_theta(self):
        for i in range(len(self.nodes)):
            old = self.nodes[i].theta
            self.nodes[i].theta = old + (self.deltas[i])

        self.deltas = np.zeros((len(self.nodes), self.nodes[0].theta.shape[0]))


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
        
        for i in range(len(self.features)):
            a = self.features[i]
            activations = []
            for layer in self.layers:
                a = layer.activation(a)
                activations.append(a)
            hyp = a
            predictions.append(hyp)

            self.layers[-1].calc_error(self.labels[i])
            self.layers[-1].accumulate_delta()

            l = len(self.layers) - 2
            while l >= 0:
                self.layers[l].calc_error(self.layers[l+1])
                self.layers[l].accumulate_delta()
                l -= 1

        return np.array(predictions)

    def train(self, iterations=10):
        for _ in range(iterations):
            hyp = self.hypothesis()
            print(f'cost: {self.cost(hyp)}')

            for l in self.layers:
                l.step_theta()


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
            LogisticNode(num_features=4,),
            LogisticNode(num_features=4,),
        ]),
    OutputLayer(
        nodes = [
            LogisticNode(num_features=3,),
        ])
]
network = Network(
    layers=layers,
    features=X,
    labels=y,
)

network.train(5000)

hyp = network.hypothesis()

print('')
print(f'last hyp: {hyp}')
