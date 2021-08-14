import numpy as np
from neural_net.network import Network, cost, hypothesis
from neural_net.layer_matrix import Layer, OutputLayer

"""
References:
https://towardsdatascience.com/how-to-debug-a-neural-network-with-gradient-checking-41deec0357a9
https://www.youtube.com/watch?v=I-X8_EcGYik&list=PLZ9qNFMHZ-A4rycgrgOYma6zxF4BZGGPW&index=55
"""

class GradientCheck:
    def __init__(
            self, 
            layers,
            features,
            labels,
            epsilon=1e-8,
            ):
        self.layers = layers
        self.features = features
        self.labels = labels
        self.epsilon = epsilon

    def cost(self, layers):
        predictions = []
        for x in self.features:
            predictions.append(hypothesis(x, layers))
        hyps = np.array(predictions)
        return cost(hyps, self.labels)

    def calc_deltas(self):
        for l in range(len(self.layers)):
            curr_mtx = self.layers[l].thetas()

            # loop through each element in the theta matrix
            for i in range(curr_mtx.shape[0]):
                for j in range(curr_mtx.shape[1]):
                    temp_layers = self.layers.copy()

                    # cost for theta minus epsilon
                    l1 = Layer(initial_theta=curr_mtx)
                    l1.theta_mtx[i][j] -= self.epsilon
                    temp_layers[l] = l1
                    l1_cost = self.cost(temp_layers)

                    # cost for theta plus epsilon
                    l2 = Layer(initial_theta=curr_mtx)
                    l2.theta_mtx[i][j] += self.epsilon
                    temp_layers[l] = l2
                    l2_cost = self.cost(temp_layers)

                    derivative = (l1_cost - l2_cost)/(2*self.epsilon)
                    self.layers[l].deltas[i][j] = derivative
            
            print(f'derivative layer {l}: {self.layers[l].deltas}')

    def train(self, iterations=1):
        for i in range(iterations):
            self.calc_deltas()
            for l in self.layers:
                l.m = 1
                l.step_theta()

            print(f'cost {i}: {self.cost(self.layers)}')


if __name__ == '__main__':    
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

    # Initialize first network; Uses back propagation
    n1l1 = Layer(num_features, 5)
    n1l2 = Layer(5, 4)
    n1l3 = OutputLayer(4, 1)
    net1 = Network(
        layers = [n1l1, n1l2, n1l3],
        features = X,
        labels = y,
    )

    # Initialize second network; Uses Gradient check
    # Layers take their thetas from the first network so we have identical
    #   start states
    n2l1 = Layer(initial_theta=n1l1.thetas())
    n2l2 = Layer(initial_theta=n1l2.thetas())
    n2l3 = Layer(initial_theta=n1l3.thetas())
    net2 = GradientCheck(
        layers = [n2l1, n2l2, n2l3],
        features = X,
        labels = y,
    )

    pre_train_1 = n2l1.thetas()
    pre_train_2 = n2l2.thetas()
    pre_train_3 = n2l3.thetas()

    net1.train(1)
    net2.train(1)

    print('post train:')
    print(n1l1.thetas())
    print(n2l1.thetas())

    d1 = np.concatenate((
        (pre_train_1 - n1l1.thetas()).reshape(-1,),
        (pre_train_2 - n1l2.thetas()).reshape(-1,),
        (pre_train_3 - n1l3.thetas()).reshape(-1,),
        ))
    d2 = np.concatenate((
        (pre_train_1 - n2l1.thetas()).reshape(-1,),
        (pre_train_2 - n2l2.thetas()).reshape(-1,),
        (pre_train_3 - n2l3.thetas()).reshape(-1,),
        ))
    numerator = np.linalg.norm(d1 - d2)
    denominator = np.linalg.norm(d1) + np.linalg.norm(d2)
    difference = numerator / denominator

    print(numerator)
    print(denominator)
    print(f'difference: {difference}')
