from benchmark import Benchmark
from math import sin, cos, sqrt
from runtime.llcl import Runtime

from mojo.gradi.matrix import Matrix
from mojo.gradient_descent import gradient_descent #, gradient_descent_vp
from mojo.plot_gradient import plot_gradient_descent_cache


alias N = 100
alias dim = 2
alias dtype = DType.float32
alias nelts = simdwidthof[dtype]()

alias PI = 3.141592653589793


### 2D
fn generate_circle_points(N: Int, r: Int) -> Matrix[dtype]:
    var points = Matrix[dtype](N, 2)
    let angle: SIMD[dtype, 1]
    
    for i in range(N):
        angle = 2 * PI * i / N
        points[i, 0] = r*cos(angle)
        points[i, 1] = r*sin(angle)

    return points


fn generate_distance_matrix(points: Matrix[dtype]) -> Matrix[dtype]:
    let distance: SIMD[dtype, 1]
    let N = points.rows
    var D = Matrix[dtype](N, N)
    D.zeros()    
    
    for i in range(N):
        for j in range(i+1, N):
            distance = sqrt((points[j, 0] - points[i, 0])**2 + (points[j, 1] - points[i, 1])**2)
            D[i, j] = distance
            D[j, i] = distance
    
    return D


@always_inline
fn benchmark(N: Int, dim: Int):
    let points: Matrix[dtype]
    let D: Matrix[dtype]
    let radius: Int = 3
    points = generate_circle_points(N, radius)
    D = generate_distance_matrix(points)

    var X = Matrix[dtype](N, dim)
    X.rand()

    with Runtime() as rt:

        @always_inline
        @parameter
        fn test_fn():
            _ = gradient_descent(X, D, rt, learning_rate = 0.0001, num_iterations = 1000)
            # _ = gradient_descent_vp(X, D, rt, learning_rate=0.00001)

        let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
        # Prevent the matrices from being freed before the benchmark run
        _ = (X, D)

        print("Average time Mojo: ", secs)


fn main():
    let points: Matrix[dtype]
    let D: Matrix[dtype]
    let radius: Int = 3

    points = generate_circle_points(N, radius)
    D = generate_distance_matrix(points)

    var X = Matrix[dtype](N, dim)
    X.rand()
    with Runtime() as rt:
        gradient_descent[dtype](X, D, rt, learning_rate = 0.00001, num_iterations = 1000)

    benchmark(N, dim)


    # var X = Matrix(N, dim)
    X.rand()
    plot_gradient_descent_cache[dtype](X, D, learning_rate = 0.00001, num_iterations = 1000)
  

