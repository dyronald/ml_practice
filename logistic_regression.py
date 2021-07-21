import numpy as np
import math
from matplotlib import pyplot as plt


def hypothesis(gz):
    return 1 / (1.0 + (np.e ** (-gz)))


def calc_gz(theta, features):
    prod = theta * features
    return np.sum(prod, axis=1)


def cost(hyp, labels):
    m = hyp.shape[0]
    return -1 / m * np.sum((labels * np.log(hyp)) + ((1-labels) * np.log(1-hyp)))


def step(theta, hyp, labels, features, alpha=0.01):
    m = hyp.shape[0]
    diff = (hyp - labels).reshape(-1, 1)
    prod = alpha / m * np.sum(diff * features, axis=0)
    return (theta - prod)


def train(
        features,
        labels, 
        theta_fill=0.5, 
        alpha=0.5,
        epochs=1000
        ):
    theta = np.full(features.shape[1], theta_fill)
    costs = []
    for _ in range(epochs):
        gz = calc_gz(theta, features)
        hyp = hypothesis(gz)
        theta = step(theta, hyp, labels, features, alpha=alpha)
        costs.append(cost(hyp, labels))

    return theta, costs
    

def init_data():
    data = np.genfromtxt(
        './data/email_spam.csv',
        delimiter=',',
        names=True)
    samples = data.shape[0]
    data_array = data.view(np.float64).view().reshape(samples, -1)

    split = np.split(data_array, [-1], axis=1)
    features = split[0]
    labels = split[1].reshape(-1)

    f_scale = []
    for f in range(features.shape[1]):
        f_scale.append(features[:, f].std() * 3.0)
    f_scale = np.array(f_scale)

    features /= f_scale
    return features, labels


features, labels = init_data()
print(f'features: {features}')
print(features.shape)

theta, costs = train(
    features=features,
    labels=labels,
    alpha=0.5,
    epochs=2500,
    )
print(theta)

plt.plot(costs, '.')
plt.show()

plt.plot(hypothesis(calc_gz(theta, features)), '.b')
plt.plot(labels, '.r')
plt.show()
