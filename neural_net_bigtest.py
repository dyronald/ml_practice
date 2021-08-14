import numpy as np
from matplotlib import pyplot as plt
from neural_net.network import Network
from neural_net.layer_matrix import Layer, OutputLayer

num_samples = int(input('num samples: '))
num_features = int(input('num features: '))
num_outputs = int(input('num outputs: '))


def init_multi_data():
    """
    Generates data with variable output nodes.
    The features are divided equally to correlate with an output node.
    The output node with the largest sum is set to 1. All others are zero.
    """
    rng = np.random.default_rng()
    features = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(num_samples, num_features),
        )

    labels = []
    for f in features:
        totals = [0] * num_outputs
        i = 0
        for fi in f:
            totals[i] += fi
            i = (i+1) % num_outputs

        label = [0] * num_outputs
        label[totals.index(max(totals))] = 1
        labels.append(label)

    labels = np.array(labels)
    return features, labels


X, y = init_multi_data()
num_samples, num_features = X.shape
print(f'x shape: {X.shape}')
print(f'y shape: {y.shape}')

print(y)

for i in range(num_outputs):
    positives = np.sum(y[:,i])
    print(f'positives[{i}]: {positives}')
    if positives < (num_features / num_outputs * 0.75):
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
        num_nodes=y.shape[1], 
        alpha=output_alpha)
)

network = Network(
    layers=layers,
    features=X,
    labels=y,
)

iterations = int(input('iterations: '))
costs = network.train(iterations)

print('thetas:')
for l in range(len(layers)):
    print(f'{l}:')
    print(layers[l].thetas())
    print('---')

X_verify, y_verify = init_multi_data()
print(f'x shape: {X_verify.shape}')
print(f'y shape: {y_verify.shape}')
print(f'verify positives: {np.sum(y_verify)}')

network.features = X_verify
network.labels = y_verify
hyp = network.iterate()

plt.plot(costs, '.')
plt.show()

plt.plot(hyp, '.b')
plt.plot(y_verify, '.r')
plt.show()

hit = 0
miss = 0
for i in range(len(hyp)):
    match = True
    for k in range(len(y_verify[i])):
        if y_verify[i][k] == 1 and hyp[i][k] < 0.5:
            match = False
            break
        elif y_verify[i][k] == 0 and hyp[i][k] > 0.5:
            match = False
            break

    if match:
        hit += 1
    else:
        miss += 1

print(f'num_samples: {num_samples}')
print(f'num_features: {num_features}')
print(f'hit: {hit}')
print(f'miss: {miss}')
