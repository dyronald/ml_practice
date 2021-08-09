import numpy as np


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
        costs = []
        for _ in range(iterations):
            hyp = self.hypothesis()
            costs.append(self.cost(hyp))
            print(f'cost {len(costs)}: {costs[-1]}')

            for l in self.layers:
                l.step_theta()

        return costs

