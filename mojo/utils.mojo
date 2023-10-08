from python.python import Python
from utils.vector import InlinedFixedVector

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
    visuals = Python.import_module("visuals")

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

    visuals.plot_gradient_descent(positions_over_time, loss_over_time)
    visuals.animate_gradient_descent(positions_over_time, loss_over_time, "Gradient Descent Animation: Mojo", False)


def set_element[dtype: DType](inout positions_over_time: PythonObject, inout loss_over_time: PythonObject, X: Matrix[dtype], D: Matrix[dtype], i: Int):
    loss_over_time.itemset((i,), loss[dtype](X, D).cast[DType.float64]())
    for r in range(X.rows):
        for c in range(X.cols):
            positions_over_time.itemset((i, r, c), X[r, c])




# Read and parse shape files (.csv format with very specific structure)
# Into the Matrix type

fn count_lines(s: String) -> Int:
    var count: Int = 0
    for i in range(len(s)):
        if s[i] == "\n":
            count += 1
    return count


fn find_first(s: String, delimiter: String) -> Int:
    for i in range(len(s)):
        if s[i] == delimiter:
            return i
    return -1


fn cast_string[dtype: DType](s: String) raises -> SIMD[dtype, 1]:
    let idx = find_first(s, delimiter=".")
    var x: SIMD[dtype, 1] = -1

    if idx == -1:
        x = atol(s)
        return x
    else:
        let c_int: SIMD[dtype, 1]
        let c_frac: SIMD[dtype, 1]
        c_int = atol(s[:idx])
        c_frac = atol(s[idx+1:])
        x = c_int + c_frac / (10 ** len(s[idx+1:]))
        return x


fn read_shape[dtype: DType](file_name: String) raises -> Matrix[dtype]: 
    var s: String

    with open(file_name, "r") as f:
        s = f.read() 

    let N: Int = count_lines(s)
    var points = Matrix[dtype](N, 2)    # Both modular.csv and flame.csv are 2D

    let x_str: String
    let y_str: String
    let coord_idx: Int
    var line_idx: Int = find_first(s, "\n")
    var point_idx: Int = 0
    while line_idx != -1:
        # Read coordinate strings of the line
        coord_idx = find_first(s[:line_idx], ",")
        x_str = s[:coord_idx]
        y_str = s[coord_idx+1:line_idx]
        
        # Update point matrix
        points[point_idx, 0] = cast_string[dtype](x_str)
        points[point_idx, 1] = cast_string[dtype](y_str)

        # Cut line and update line_idx
        s = s[line_idx+1:]
        line_idx = find_first(s, "\n")
        point_idx += 1

    return points
