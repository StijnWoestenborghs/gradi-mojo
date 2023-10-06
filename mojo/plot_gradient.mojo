from python.python import Python

from mojo.gradient_descent import gradient_descent, compute_gradient, loss
from mojo.gradi.matrix import Matrix


fn plot_gradient_descent_cache[dtype: DType](
        inout X: Matrix[dtype], 
        D: Matrix[dtype], 
        learning_rate: SIMD[dtype, 1], 
        num_iterations: Int
    ):

    var grad = Matrix[dtype](X.rows, X.cols)

    var positions_over_time: PythonObject = []
    var loss_over_time: PythonObject = []

    try:
        for i in range(num_iterations):
            
            positions_over_time += flatten(X)                                   # ? Can't seem to find another way but to flatten and reshape
            loss_over_time += [loss[dtype](X, D).cast[DType.float64]()]         # ? Mandatory cast to float64
            
            grad.zeros()
            compute_gradient[dtype](grad, X, D)
            for r in range(X.rows):
                for c in range(X.cols):
                    X[r, c] -= learning_rate * grad[r, c]

        positions_over_time += flatten(X)
        loss_over_time += [loss(X, D).cast[DType.float64]()]
    except e:
        print(e)

    # Flat array shape
    let shape: Tuple[Int, Int, Int] = (num_iterations + 1, X.rows, X.cols)
    
    try:
        Python.add_to_path("./python")
        let np = Python.import_module("numpy")
        let utils = Python.import_module("utils")
    
        positions_over_time = np.reshape(positions_over_time, shape)
        let a = utils.plot_gradient_descent(positions_over_time, loss_over_time)
    except e:
        print(e)


fn flatten[dtype: DType](X: Matrix[dtype]) -> PythonObject:
    var flat_array: PythonObject = []
    for i in range(X.rows):
        for j in range(X.cols):
            try:
                flat_array += [X[i, j].cast[DType.float64]()]
            except e:
                print(e)

    return flat_array