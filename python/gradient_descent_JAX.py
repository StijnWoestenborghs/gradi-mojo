import jax.numpy as jnp
import jax
from functools import partial

def loss(X, D):
    N = X.shape[0]
    total_loss = 0
    
    for i in range(N):
        for j in range(N):
            difference = X[i] - X[j]
            squared_distance = jnp.dot(difference.T, difference)
            total_loss += (squared_distance - D[i, j]**2)**2
            
    return total_loss

# ----- Jax Method 1 -----
@partial(jax.jit, static_argnums=(2,3))
def gradient_descent_JAX(X, D, learning_rate=0.0001, num_iterations=1000):
    D = jnp.array(D)
    X = jnp.array(X)
    
    iterations = jnp.arange(num_iterations)
    (X, learning_rate, D), _ = jax.lax.scan(grad_step, (X, learning_rate, D), iterations)
    return X

def grad_step(carry, x):
    X, learning_rate, D = carry
    
    grad = compute_gradient(X, D)
    X -= learning_rate * grad
    return (X, learning_rate, D), None

def compute_gradient(X, D):
    iterations = jnp.arange(X.shape[0])
    (X, D), grad = jax.lax.scan(iter1, (X, D), iterations)
    return grad

def iter1(carry, row1):
    X, D = carry
    diff = X[row1] - X
    diff_squared = diff ** 2
    squared_distance = jnp.sum(diff_squared, axis=diff_squared.ndim - 1)
    squared_distance_diff = 4 * (squared_distance - (D[row1] ** 2).T)
    squared_distance_diff_reshaped = jnp.reshape(squared_distance_diff, (squared_distance_diff.shape[0], 1))
    return (X, D), jnp.sum(squared_distance_diff_reshaped * diff, axis=0)

# ----- Jax cache method for plotting -----
def gradient_descent_cache_JAX(X, D, learning_rate=0.001, num_iterations=1000):
    D = jnp.array(D)
    X = jnp.array(X)
    
    iterations = jnp.arange(num_iterations)
    _, (positions_over_time, loss_over_time) = jax.lax.scan(grad_step_with_time_evolution, (X, learning_rate, D), iterations)

    return positions_over_time, loss_over_time

def grad_step_with_time_evolution(carry, _):
    X, learning_rate, D = carry
    loss_val = loss(X, D)
    grad = compute_gradient(X, D)
    X -= learning_rate * grad
    return (X, learning_rate, D), (X, loss_val)
