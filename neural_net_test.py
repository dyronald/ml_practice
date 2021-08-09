import numpy as np
from neural_net.network import Network
from neural_net.layer_loop import LogisticNode, Layer, OutputLayer

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
