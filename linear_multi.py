import numpy as np
from matplotlib import pyplot as plt


class MultiFeatureLinearRegression:
    def __init__(self, features, labels, alpha=0.1, epochs=3000):
        self.samples = features.shape[0]
        self.features = np.concatenate(
            (np.full((self.samples, 1), 1.0), features),
            axis=1
        )
        self.labels = labels
        self.alpha = alpha
        self.epochs = epochs

    def prediction(self, theta):
        return np.sum(self.features * theta, axis=1)

    def cost(self, prediction):
        return 1 / 2 / self.samples * np.sum(np.square(prediction - self.labels))

    def step_theta(self, prediction, theta):
        theta_next = []
        for j in range(len(theta)):
            next = theta[j] - (self.alpha / self.samples *
                               np.sum((prediction - self.labels) * self.features[:, j]))
            theta_next.append(next)
        return np.array(theta_next)

    def train(self):
        theta = np.full((self.features.shape[1],), 1.0)
        for _ in range(self.epochs):
            p = self.prediction(theta)
            theta = self.step_theta(p, theta)
            print(self.cost(p))
        return theta


def clean_data(data, f_scale=None):
    features = np.array(
        (
            data['housing_median_age'],
            data['total_rooms'],
            data['total_bedrooms'],
            data['population'],
            data['households'],
            data['median_income'],
        ),
    ).transpose()

    labels = data['median_house_value']

    if f_scale is None:
        f_scale = []
        for f in range(features.shape[1]):
            f_scale.append(features[:, f].std() * 3.0)

    features /= f_scale
    return features, labels, f_scale


data = np.genfromtxt(
    './data/california_housing_train.csv',
    delimiter=',',
    names=True)
features, labels, f_scale = clean_data(data)
print(labels)
print(features)
print(features.shape)

trainer = MultiFeatureLinearRegression(
    features=features,
    labels=labels,
    alpha=0.03,
    epochs=5000)
theta = trainer.train()
print(theta)


# Plot prediction and label against each of the feature
p = trainer.prediction(theta)
for f in range(features.shape[1]):
    x = features[:, f]
    plt.scatter(x, labels)
    plt.scatter(x, p)
    plt.show()


# Plot prediction and label using test data
# Test data is sorted according to the label to emphasize trend
test_data = np.genfromtxt(
    './data/california_housing_test.csv',
    delimiter=',',
    names=True)
test_features, test_labels, _ = clean_data(test_data, f_scale)

order = np.argsort(test_labels)
sorted_labels = test_labels[order]
sorted_features = test_features[order]

tester = MultiFeatureLinearRegression(
    features=sorted_features,
    labels=sorted_labels)
p = tester.prediction(theta)

x = range(sorted_labels.shape[0])
plt.scatter(x, p)
plt.scatter(x, sorted_labels)
plt.show()
