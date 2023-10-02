import numpy as np


def loss(X, D):
    N = X.shape[0]
    total_loss = 0
    
    for i in range(N):
        for j in range(N):
            difference = X[i] - X[j]
            squared_distance = np.dot(difference.T, difference)
            total_loss += (squared_distance - D[i, j]**2)**2
            
    return total_loss


def compute_gradient(X, D):
    N = X.shape[0]
    grad = np.zeros_like(X)

    for i in range(N):
        for j in range(N):
            difference = X[i] - X[j]
            squared_distance = np.dot(difference.T, difference)
            grad[i] += 4 * (squared_distance - D[i, j]**2) * difference

    return grad


def gradient_descent(D, learning_rate=0.0001, num_iterations=1000):
    dim = 2
    N = D.shape[0]
    X = np.random.rand(N, dim)

    for i in range(num_iterations):
        grad = compute_gradient(X, D)
        X -= learning_rate * grad

    return X


def gradient_descent_cache(D, learning_rate=0.001, num_iterations=1000):
    dim = 2
    N = D.shape[0]
    X = np.random.rand(N, dim)

    positions_over_time = []
    loss_over_time = []

    for i in range(num_iterations):
        positions_over_time.append(X.copy())
        loss_over_time.append(loss(X, D))

        grad = compute_gradient(X, D)
        X -= learning_rate * grad

    positions_over_time.append(X.copy())
    loss_over_time.append(loss(X, D))

    return positions_over_time, loss_over_time

