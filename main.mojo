from benchmark import Benchmark
from math import sin, cos, sqrt
from runtime.llcl import Runtime

from mojo.gradient_descent import Matrix, gradient_descent, gradient_descent_vp
from mojo.plot_gradient import plot_gradient_descent_cache


alias PI = 3.141592653589793

### 2D
fn generate_circle_points(inout points: Matrix):
    let N = points.rows
    let angle: Float32
    let r = 3
    
    for i in range(N):
        angle = 2 * PI * i / N
        points[i, 0] = r*cos(angle)
        points[i, 1] = r*sin(angle)

fn generate_distance_matrix(inout D: Matrix, N: Int):
    let distance: Float32
    var points = Matrix(N, 2)
    generate_circle_points(points)

    D.zero()
    for i in range(N):
        for j in range(i+1, N):
            distance = sqrt((points[j, 0] - points[i, 0])**2 + (points[j, 1] - points[i, 1])**2)
            D[i, j] = distance
            D[j, i] = distance


@always_inline
fn benchmark(N: Int, dim: Int):
    var X = Matrix(N, dim)
    var D = Matrix(N, N)
    generate_distance_matrix(D, N)

    with Runtime() as rt:

        @always_inline
        @parameter
        fn test_fn():
            # _ = gradient_descent(X, D, rt, learning_rate=0.00001)
            _ = gradient_descent_vp(X, D, rt, learning_rate=0.00001)

        let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
        # Prevent the matrices from being freed before the benchmark run
        _ = (X, D)

        print("Average time Mojo: ", secs)


fn main():
    alias N = 100
    alias dim = 2

    var D = Matrix(N, N)
    generate_distance_matrix(D, N)

    # var X = Matrix(N, dim)
    # with Runtime() as rt:
    #     gradient_descent(X, D, rt)
    # print(X.__str__())

    benchmark(N, dim)

    var X = Matrix(N, dim)
    plot_gradient_descent_cache(X, D)
  

