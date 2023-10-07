from benchmark import Benchmark
from python.python import Python
from math import sin, cos, sqrt, acos
from tensor import Tensor
from utils.index import Index
from random import rand

from mojo.gradient_descent import gradient_descent
from mojo.plot_gradient import plot_gradient_descent_cache
from mojo.tensorutils import *

alias PI = 3.141592653589793


fn generate_radial_points[dtype: DType](N: Int, dim: Int) -> Tensor[dtype]:
    var points = Tensor[dtype](N, dim)
    let angle: SIMD[dtype, 1]
    let r: Int = 3
    
    if dim == 2:
        for i in range(N):
            angle = 2 * PI * i / N
            points[Index(i, 0)] = r * cos(angle)
            points[Index(i, 1)] = r * sin(angle)
    elif dim == 3:
        let phi: SIMD[dtype, 1]
        let theta: SIMD[dtype, 1]
        for i in range(N):
            angle = (1 - 2 * (i / N)).cast[dtype]()
            phi = acos[dtype, 1](angle)
            theta = sqrt[dtype, 1](N * PI) * phi    
            points[Index(i, 0)] = r * sin(phi) * cos(theta)
            points[Index(i, 1)] = r * sin(phi) * sin(theta)
            points[Index(i, 2)] = r * cos(phi)
    else:
        print("Only supports 2D and 3D !")
            
    return points


fn generate_distance_matrix[dtype: DType](points: Tensor[dtype]) -> Tensor[dtype]:  
    let N = points.shape()[0]
    let dim = points.shape()[1]
    var distance: SIMD[dtype, 1]
    var D = Tensor[dtype](N, N)
    zero(D)

    for i in range(N):
        for j in range(i+1, N):
            distance = 0
            for d in range(dim):
                distance += (points[j, d] - points[i, d])**2
            distance = sqrt(distance)
            D[Index(i, j)] = distance
            D[Index(j, i)] = distance
    
    return D


@always_inline
fn benchmark[dtype: DType, nelts: Int](N: Int, dim: Int, lr: SIMD[dtype, 1], niter: Int):
    let points: Tensor[dtype]
    let D: Tensor[dtype]
    
    points = generate_radial_points[dtype](N, dim)
    D = generate_distance_matrix[dtype](points)

    var X = Tensor[dtype](N, dim)
    randt(X)

    @parameter
    fn test_fn():
        _ = gradient_descent[dtype, nelts](X, D, learning_rate = lr, num_iterations = niter)

    let secs = Benchmark().run[test_fn]() / 1e9
    # Prevent the matrices from being freed before the benchmark run
    _ = (X, D)

    print("Average time Mojo: ", secs)


fn main():
    #### TODO: implement with tensor ? 
    # let t = Tensor[DType.float64](10, 2)

    alias N = 100
    alias dim = 3
    alias lr = 0.00001
    alias niter = 1000
    alias dtype = DType.float32
    alias nelts = simdwidthof[dtype]()
    let plots: Bool = True


    ### Benchmarks from python
    # [python native, numpy, jax, C++ (python binding)]
    try:
        Python.add_to_path(".")
        let pymain = Python.import_module("main")
        _ = pymain.benchmarks(
            N,
            dim,
            lr,
            niter,
            plots
        )
    except e:
        print("Error: ", e)


    # Generate optimization target
    let points: Tensor[dtype]
    let D: Tensor[dtype]
    
    points = generate_radial_points[dtype](N, dim)
    D = generate_distance_matrix[dtype](points)

    # Initial starting point
    var X = Tensor[dtype](N, dim)
    randt(X)

    ### Without visuals
    gradient_descent[dtype, nelts](X, D, learning_rate=lr, num_iterations=niter)

    ### Benchmark Mojo
    benchmark[dtype, nelts](N, dim, lr=lr, niter=niter)

    ### PLOTTING  
    try:
        if plots:
            randt(X)
            _ = plot_gradient_descent_cache[dtype](X, D, learning_rate=lr, num_iterations=niter)
    except e:
        print("Error: ", e)
