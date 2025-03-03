import numpy as np
import matplotlib.pyplot as plt
from cost_functions.losses import mean_squared_error

def compute_cost(X, y, theta):
    """
    Compute the mean squared error cost
    """
    predictions = X.dot(theta)
    standard_mse = mean_squared_error(y, predictions)
    return (1/2) * standard_mse


def gradient_descent(X, y, theta, alpha, iterations, standardize=False):
    """
    Perform gradient descent to learn theta parameters
    """
    m = len(y)
    cost_history = np.zeros(iterations)

    if standardize:
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        gradient = (1 / m) * X.T.dot(errors)
        theta -= alpha * gradient
        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history


def plot_cost_history(cost_history):
    """
    Plot the cost function history
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, 'b-', linewidth=3)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function History')
    plt.grid(True)
    plt.show()