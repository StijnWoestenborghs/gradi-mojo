from benchmark import Benchmark
from python.python import Python
from math import sin, cos, sqrt, acos

from mojo.gradi.matrix import Matrix
from mojo.gradient_descent import gradient_descent
from mojo.utils import plot_gradient_descent_cache, read_shape

alias PI = 3.141592653589793


fn generate_radial_points[dtype: DType](N: Int, dim: Int) -> Matrix[dtype]:
    var points = Matrix[dtype](N, dim)
    let angle: SIMD[dtype, 1]
    let r: SIMD[dtype, 1] = 0.5
    
    if dim == 2:
        for i in range(N):
            angle = 2 * PI * i / N
            points[i, 0] = r * cos(angle)
            points[i, 1] = r * sin(angle)
    elif dim == 3:
        let phi: SIMD[dtype, 1]
        let theta: SIMD[dtype, 1]
        for i in range(N):
            angle = (1 - 2 * (i / N)).cast[dtype]()
            phi = acos[dtype, 1](angle)
            theta = sqrt[dtype, 1](N * PI) * phi    
            points[i, 0] = r * sin(phi) * cos(theta)
            points[i, 1] = r * sin(phi) * sin(theta)
            points[i, 2] = r * cos(phi)
    else:
        print("Only supports 2D and 3D !")
            
    return points


fn generate_distance_matrix[dtype: DType](points: Matrix[dtype]) -> Matrix[dtype]:  
    let N = points.rows
    let dim = points.cols
    var distance: SIMD[dtype, 1]
    var D = Matrix[dtype](N, N)
    D.zeros()    

    for i in range(N):
        for j in range(i+1, N):
            distance = 0
            for d in range(dim):
                distance += (points[j, d] - points[i, d])**2
            distance = sqrt(distance)
            D[i, j] = distance
            D[j, i] = distance
    
    return D


@always_inline
fn benchmark[dtype: DType, nelts: Int](D: Matrix[dtype], dim: Int, lr: SIMD[dtype, 1], niter: Int):

    # Initial starting point
    var X = Matrix[dtype](D.rows, dim)
    X.rand()

    @parameter
    fn test_fn():
        _ = gradient_descent[dtype, nelts](X, D, learning_rate = lr, num_iterations = niter)

    let secs = Benchmark().run[test_fn]() / 1e9
    # Prevent the matrices from being freed before the benchmark run
    _ = (X, D)

    print("Average time Mojo: ", secs)


fn main():

    alias dtype = DType.float32
    alias nelts = simdwidthof[dtype]()

    # Generate optimization target
    var points: Matrix[dtype]
    alias n_circle = 100
    alias dim_circle = 3
    points = generate_radial_points[dtype](n_circle, dim_circle)
    
    try:
        points = read_shape[dtype]("./shapes/flame.csv")
        # points = read_shape[dtype]("./shapes/modular.csv")
    except e:
        print("Failed to parse shape: ", e)


    # Optimization input
    alias dim = 2
    alias lr = 0.0001
    alias niter = 1000
    alias plots = True

    let D: Matrix[dtype]
    D = generate_distance_matrix[dtype](points)

    ### Benchmarks from python
    # [python native, numpy, jax, C++ (python binding)]
    try:
        Python.add_to_path(".")
        let pymain = Python.import_module("main")
        _ = pymain.benchmarks(
            D.to_python(),
            dim,
            lr,
            niter,
            plots
        )
    except e:
        print("Error: ", e)


    # Initial starting point
    var X = Matrix[dtype](D.rows, dim)
    X.rand()

    ### Without visuals
    gradient_descent[dtype, nelts](X, D, learning_rate=lr, num_iterations=niter)

    ### Benchmark Mojo
    benchmark[dtype, nelts](D, dim, lr=lr, niter=niter)

    ### PLOTTING  
    try:
        if plots:
            X.rand()
            _ = plot_gradient_descent_cache[dtype](X, D, learning_rate=lr, num_iterations=niter)
    except e:
        print("Error: ", e)



