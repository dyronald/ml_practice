import numpy as np
from matplotlib import pyplot as plt
from neural_net.network import Network
from neural_net.layer_matrix import Layer, OutputLayer

num_samples = int(input('num samples: '))
num_features = int(input('num features: '))

def init_data():
    """
    Generate a randomized features and labels.
    Feature elements are either 0 or 1.
    Some proportion of elements are designated "white" or "black".
    Label is set to 1 if both are true:
      1. 1's in white elements reaches some threshold
      2. 0's in black elements reaches some threshold
    """
    rng = np.random.default_rng()
    features = rng.choice(a=[0,1], size=(num_samples, num_features))

    labels = []
    for f in features:
        white = 0
        black = 0
        white_factor = num_features / 2
        for i in range(len(f)):
            if (i+1) % white_factor == 0:
                white += f[i]
            else:
                black += f[i]

        white_region_size = num_features / white_factor
        if white >= (white_region_size/2) and black <= ((num_features-white_region_size)/2):
            labels.append(1)
        else:
            labels.append(0)

    labels = np.array([labels]).T
    return features, labels


X, y = init_data()
print(f'x shape: {X.shape}')
print(f'y shape: {y.shape}')

positives = np.sum(y)
print(f'positives: {positives}')

if positives > (2/3 * num_samples) or positives < (1/3 * num_samples):
    print('poor samples ratio. aborting.')
    exit()

layers = []
num_layers = int(input('num hidden layers: '))
num_inputs = num_features
for _ in range(num_layers):
    num_nodes = int(input('num nodes: '))
    alpha = float(input('alpha: '))
    layers.append(Layer(
        num_inputs=num_inputs,
        num_nodes=num_nodes,
        alpha=alpha,
    ))

    num_inputs = num_nodes

output_alpha = float(input('output alpha: '))
layers.append(
    OutputLayer(
        num_inputs=num_inputs,
        num_nodes=1, 
        alpha=output_alpha)
)

network = Network(
    layers=layers,
    features=X,
    labels=y,
)

iterations = int(input('iterations: '))
costs = network.train(iterations)
hyp = network.iterate()

plt.plot(costs, '.')
plt.show()

plt.plot(hyp, '.b')
plt.plot(y, '.r')
plt.show()

hit = 0
maybe = 0
miss = 0
for i in range(len(hyp)):
    if y[i] == 1:
        if hyp[i] > 0.8:
            hit += 1
        elif hyp[i] > 0.5:
            maybe += 1
        else:
            miss += 1
    else:
        if hyp[i] < 0.2:
            hit += 1
        elif hyp[i] < 0.5:
            maybe += 1
        else:
            miss += 1

print(f'num_samples: {num_samples}')
print(f'num_features: {num_features}')
print(f'hit: {hit}')
print(f'maybe: {maybe}')
print(f'miss: {miss}')

print('thetas:')
for l in range(len(layers)):
    print(f'{l}:')
    print(layers[l].thetas())
    print('---')
