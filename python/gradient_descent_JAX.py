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
    iterations = jnp.arange(X.shape[0])
    (X, D, row1), grad = jax.lax.scan(calc_single_grad, (X, D, row1), iterations)
    grad = jnp.sum(grad, axis=0)    
    return (X, D), grad

def calc_single_grad(carry, row2):
    X, D, row1 = carry
    
    difference = X[row1] - X[row2]
    squared_distance = difference @ difference.T
    grad = 4 * (squared_distance - D[row1, row2]**2) * difference
    return (X, D, row1), grad

# ----- Jax Method 3 -----
@partial(jax.jit, static_argnums=(2,3))
def gradient_descent_JAX3(X, D, learning_rate=0.0001, num_iterations=1000):
    D = jnp.array(D)
    X = jnp.array(X)
    
    iterations = jnp.arange(num_iterations)
    (X, learning_rate, D), _ = jax.lax.scan(grad_step_jax3, (X, learning_rate, D), iterations)
    return X

def grad_step_jax3(carry, x):
    X, learning_rate, D = carry
    
    grad = compute_gradient_JAX3(X, D)
    X -= learning_rate * grad
    return (X, learning_rate, D), None

def compute_gradient_JAX3(X, D):
    N = X.shape[0]
    grad = []

    for i in range(N):
        grad_i = 0
        for j in range(N):
            difference = X[i] - X[j]
            squared_distance = jnp.dot(difference.T, difference)
            grad_i += 4 * (squared_distance - D[i, j]**2) * difference
        grad.append(grad_i)
    grad_arr = jnp.array(grad)
    return grad_arr

# ----- Jax cache method for plotting -----
def gradient_descent_cache_JAX(X, D, learning_rate=0.001, num_iterations=1000):
    D = jnp.array(D)
    X = jnp.array(X)
    
    iterations = jnp.arange(num_iterations)
    (X, learning_rate, D), (positions_over_time, loss_over_time) = jax.lax.scan(grad_step_with_time_evolution, (X, learning_rate, D), iterations)

    #positions_over_time.append(X.copy())
    #loss_over_time.append(loss(X, D))

    return positions_over_time, loss_over_time


def grad_step_with_time_evolution(carry, x):
    X, learning_rate, D = carry
    loss_val = loss(X, D)
    grad = compute_gradient(X, D)
    X -= learning_rate * grad
    return (X, learning_rate, D), (X,loss_val)