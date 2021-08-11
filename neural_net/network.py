import numpy as np


def cost(hyps, labels):
    m = hyps.shape[0]
    diff_true = labels * np.log(hyps)
    diff_false = (1-labels) * np.log(1-hyps)
    return -1 / m * np.sum(diff_true + diff_false)


def hypothesis(input, layers):
    a = input
    for layer in layers:
        a = layer.activation(a)
    hyp = a
    return hyp


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

    def iterate(self):
        predictions = []
        
        for i in range(len(self.features)):
            hyp = hypothesis(self.features[i], self.layers)
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
        costs = []
        for _ in range(iterations):
            hyps = self.iterate()
            costs.append(cost(hyps, self.labels))
            print(f'cost {len(costs)}: {costs[-1]}')

            for l in self.layers:
                l.step_theta()

        return costs
