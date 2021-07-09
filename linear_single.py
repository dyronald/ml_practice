import numpy as np
from matplotlib import pyplot as plt


def plot_theta(theta, data, feature, label):
    # Label the axes.
    plt.xlabel(feature)
    plt.ylabel(label)

    # Plot the feature values vs. label values.
    plt.scatter(data[feature], data[label])

    # Create a red line representing the model. The red line starts
    # at coordinates (x0, y0) and ends at coordinates (x1, y1).
    x0 = 0
    y0 = theta[0]
    x1 = data[feature][-1]
    y1 = theta[0] + (theta[1] * x1)
    plt.plot([x0, x1], [y0, y1], c='r')

    plt.show()


def sort_by_key(data, key):
    order = np.argsort(data[key])
    return data[order]


class SingleFeatureLinearRegression:
    def __init__(self, data, feature, label, alpha=0.1, epochs=300):
        self.feature_vector = data[feature]
        self.label_vector = data[label]
        self.alpha = alpha
        self.epochs = epochs

    def prediction(self, theta):
        return theta[0] + theta[1] * self.feature_vector

    def cost(self, prediction):
        size = self.label_vector.shape[0]
        return 1 / 2 / size * np.sum(np.square(prediction - self.label_vector))

    def partial_derivative_0(self, prediction):
        size = self.label_vector.shape[0]
        return 1 / size * np.sum(prediction - self.label_vector)

    def partial_derivative_1(self, prediction):
        size = self.feature_vector.shape[0]
        derivative = (
            prediction - self.label_vector) * self.feature_vector
        return 1 / size * np.sum(derivative)

    def train(self):
        theta = [1, 1]

        for _ in range(self.epochs):
            p = self.prediction(theta)

            d0 = self.partial_derivative_0(p)
            d1 = self.partial_derivative_1(p)

            theta[0] = theta[0] - (self.alpha * d0)
            theta[1] = theta[1] - (self.alpha * d1)

            print(f'cost: {self.cost(p)}')

        print(f't: {theta}')
        return theta


data = np.genfromtxt(
    './data/california_housing_train.csv',
    delimiter=',',
    names=True)
test_data = np.genfromtxt(
    './data/california_housing_test.csv',
    delimiter=',',
    names=True)
feature = 'median_income'
label = 'median_house_value'

trainer = SingleFeatureLinearRegression(
    data=data,
    feature=feature,
    label=label,
)

theta = trainer.train()

sorted = sort_by_key(data, feature)
plot_theta(theta, sorted, feature, label)

sorted = sort_by_key(test_data, feature)
plot_theta(theta, sorted, feature, label)
