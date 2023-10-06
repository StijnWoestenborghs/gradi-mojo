from python.python import Python

from mojo.gradient_descent import gradient_descent, compute_gradient, loss
from mojo.gradi.matrix import Matrix


def plot_gradient_descent_cache[dtype: DType](
        inout X: Matrix[dtype], 
        D: Matrix[dtype], 
        learning_rate: SIMD[dtype, 1], 
        num_iterations: Int
    ):

    # Import python modules
    Python.add_to_path("./python")
    np = Python.import_module("numpy")
    utils = Python.import_module("utils")

    # Gradient descent cach
    var grad = Matrix[dtype](X.rows, X.cols)

    positions_over_time = np.zeros((num_iterations + 1, X.rows, X.cols), np.float64)
    loss_over_time = np.zeros((num_iterations + 1, ), np.float64)

    for i in range(num_iterations):
    
        # Cache to numpy arrays
        set_element[dtype](positions_over_time, loss_over_time, X, D, i)

        grad.zeros()
        compute_gradient[dtype](grad, X, D)
        for r in range(X.rows):
            for c in range(X.cols):
                X[r, c] -= learning_rate * grad[r, c]

    # Add last element
    set_element[dtype](positions_over_time, loss_over_time, X, D, num_iterations)

    utils.plot_gradient_descent(positions_over_time, loss_over_time)


def set_element[dtype: DType](inout positions_over_time: PythonObject, inout loss_over_time: PythonObject, X: Matrix[dtype], D: Matrix[dtype], i: Int):
    loss_over_time.itemset((i,), loss[dtype](X, D).cast[DType.float64]())
    for r in range(X.rows):
        for c in range(X.cols):
            positions_over_time.itemset((i, r, c), X[r, c])
