from algorithm import vectorize, parallelize, vectorize_unroll
from runtime.llcl import Runtime
from python.python import Python

from mojo.gradi.matrix import Matrix


fn loss[dtype: DType](X: Matrix[dtype], D: Matrix[dtype]) -> SIMD[dtype, 1]:
    let N = X.rows
    let dim = X.cols
    var squared_distance: SIMD[dtype, 1] = 0
    var total_loss: SIMD[dtype, 1] = 0

    for i in range(N):
        for j in range(N):

            squared_distance = 0
            for d in range(dim):
                squared_distance += (X[i, d] - X[j, d])**2

            total_loss += (squared_distance - D[i, j]**2)**2
    
    return total_loss


fn compute_gradient[dtype: DType](inout grad: Matrix[dtype], X: Matrix[dtype], D: Matrix[dtype], _rt: Runtime):
    let N = X.rows
    let dim = X.cols
    var squared_distance: SIMD[dtype, 1] = 0

    for i in range(N):
        for j in range(N):
            squared_distance = 0
            for d in range(dim):
                squared_distance += (X[i, d] - X[j, d])**2

            for d in range(dim):
                grad[i, d] += 4 * (squared_distance - D[i, j] ** 2) * (X[i, d] - X[j, d])


fn gradient_descent[dtype: DType, nelts: Int](
        inout X: Matrix[dtype], 
        D: Matrix[dtype],
        rt: Runtime, 
        learning_rate: SIMD[dtype, 1], 
        num_iterations: Int
    ):

    let N = X.rows
    let dim = X.cols

    var grad = Matrix[dtype](N, dim)

    # for _ in range(num_iterations):
    #     grad.zeros()
    #     compute_gradient[dtype](grad, X, D, rt)
    #     for r in range(X.rows):
    #         for c in range(X.cols):
    #             X[r, c] -= learning_rate * grad[r, c]

    
    # ## Using extended matrix methods
    # for _ in range(num_iterations):
    #     grad.zeros()
    #     compute_gradient[dtype](grad, X, D, rt)
    #     X -= learning_rate * grad


    ## Parallel gradient computation
    for _ in range(num_iterations):
        grad.zeros()
        compute_gradient[dtype, nelts](grad, X, D, rt)
        for r in range(X.rows):
            for c in range(X.cols):
                X[r, c] -= learning_rate * grad[r, c]



### Vector & Parallel

fn compute_gradient[dtype: DType, nelts: Int](inout grad: Matrix[dtype], X: Matrix[dtype], D: Matrix[dtype], rt: Runtime):
    let N = X.rows
    let dim = X.cols

    @parameter
    fn calc_row(i: Int):
        var squared_distance: SIMD[dtype, 1] = 0

        for j in range(N):
            squared_distance = 0
            for d in range(dim):
                squared_distance += (X[i, d] - X[j, d])**2

            for d in range(dim):
                grad[i, d] += 4 * (squared_distance - D[i, j] ** 2) * (X[i, d] - X[j, d])

            # @parameter
            # fn grad_vector[nelts: Int](d: Int):
            #     grad.store[nelts](
            #         i, d, grad.load[nelts](i, d) + 4 * (squared_distance - D[i, j] ** 2) * (X.load[nelts](i, d) - X.load[nelts](j, d))
            #     )
                
            # vectorize[nelts, grad_vector](dim)

    parallelize[calc_row](rt, N)

