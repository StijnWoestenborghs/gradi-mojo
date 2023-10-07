from python.python import Python
from tensor import Tensor
from utils.index import Index

from mojo.gradient_descent import gradient_descent, compute_gradient, loss
from mojo.tensorutils import zero



def plot_gradient_descent_cache[dtype: DType](
        inout X: Tensor[dtype], 
        D: Tensor[dtype], 
        learning_rate: SIMD[dtype, 1], 
        num_iterations: Int
    ):

    # Import python modules
    Python.add_to_path("./python")
    np = Python.import_module("numpy")
    utils = Python.import_module("utils")

    # Gradient descent cach
    let shape = X.shape()
    var grad = Tensor[dtype](shape[0], shape[1])

    positions_over_time = np.zeros((num_iterations + 1, shape[0], shape[1]), np.float64)
    loss_over_time = np.zeros((num_iterations + 1, ), np.float64)

    for i in range(num_iterations):
    
        # Cache to numpy arrays
        set_element[dtype](positions_over_time, loss_over_time, X, D, i)

        zero(grad)
        compute_gradient[dtype](grad, X, D)
        for r in range(shape[0]):
            for c in range(shape[1]):
                X[Index(r, c)] -= learning_rate * grad[r, c]

    # Add last element
    set_element[dtype](positions_over_time, loss_over_time, X, D, num_iterations)

    utils.plot_gradient_descent(positions_over_time, loss_over_time)


def set_element[dtype: DType](inout positions_over_time: PythonObject, inout loss_over_time: PythonObject, X: Tensor[dtype], D: Tensor[dtype], i: Int):
    let shape = X.shape()
    loss_over_time.itemset((i,), loss[dtype](X, D).cast[DType.float64]())
    for r in range(shape[0]):
        for c in range(shape[1]):
            positions_over_time.itemset((i, r, c), X[r, c])
