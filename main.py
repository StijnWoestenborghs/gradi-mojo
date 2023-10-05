import numpy as np

from python.gradient_descent import gradient_descent, gradient_descent_cache
from python.gradient_descent_native import gradient_descent_native, gradient_descent_native_cache, PyMatrix
from python.gradient_descent_JAX import gradient_descent_JAX, gradient_descent_cache_JAX
from python.utils import plot_gradient_descent, animate_gradient_descent

from timeit import timeit

def benchmark_gradient_descent(X, D):
    secs = timeit(lambda: gradient_descent(X, D), number=10) / 2
    print(f"Average time python numpy: {secs}")

def benchmark_gradient_descent_JAX(X, D):
    secs = timeit(lambda: gradient_descent_JAX(X, D), number=10) / 2
    print(f"Average time JAX: {secs}")
    
def benchmark_gradient_descent_native(X, D):
    N = D.shape[0]
    D_native = PyMatrix(D.tolist(), N, N)
    secs = timeit(lambda: gradient_descent_native(D_native), number=10) / 2
    print(f"Average time python native: {secs}")


### 2D
def generate_circle_points(N):
    points = []
    r = 3
    for i in range(N):
        angle = 2 * np.pi * i / N
        points.append([r*np.cos(angle), r*np.sin(angle)])
    return points


def generate_distance_matrix(points):
    n = len(points)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    
    return distance_matrix



if __name__ == "__main__":
    N = 10
    circle = generate_circle_points(N)
    D = np.array(generate_distance_matrix(circle), dtype=np.float64)
    D_native = PyMatrix(D.tolist(), N, N)

    np.random.seed(42)
    dim = 2
    X = np.random.rand(N, dim)  # use same starting values X for np and jax

    ### Without visuals
    p = gradient_descent_JAX(X, D)
    p2 = gradient_descent(X, D)
    p3 = gradient_descent_native(D_native)

    ### Benchmarks
    benchmark_gradient_descent(X, D)
    benchmark_gradient_descent_native(X, D)  
    benchmark_gradient_descent_JAX(X, D)  

    ### PLOTTING
    P, L = gradient_descent_cache(D, learning_rate=0.0001, num_iterations=1000)
    plot_gradient_descent(P, L, title="Gradient Descent in python numpy")
    
    P_native, L_native = gradient_descent_native_cache(D_native, learning_rate=0.0001, num_iterations=1000)
    plot_gradient_descent(P_native, L_native, title="Gradient Descent in native python")

    animate_gradient_descent(P, L)