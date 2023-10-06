from algorithm import vectorize, parallelize, vectorize_unroll
from python.python import Python

from mojo.gradi.matrix import Matrix


fn loss[dtype: DType](X: Matrix[dtype], D: Matrix[dtype]) -> SIMD[dtype, 1]:
    var total_loss: SIMD[dtype, 1] = 0
    var squared_distance: SIMD[dtype, 1] = 0
    
    for i in range(X.rows):
        for j in range(X.rows):
            squared_distance = 0
            for d in range(X.cols):
                squared_distance += (X[i, d] - X[j, d])**2
            
            total_loss += (squared_distance - D[i, j]**2)**2

    return total_loss


fn compute_gradient[dtype: DType](inout grad: Matrix[dtype], X: Matrix[dtype], D: Matrix[dtype]):
    var squared_distance: SIMD[dtype, 1]
    
    for i in range(X.rows):
        for j in range(X.rows):
            squared_distance = 0
            for d in range(X.cols):
                squared_distance += (X[i, d] - X[j, d])**2

            for d in range(X.cols):
                grad[i, d] += 4 * (squared_distance - D[i, j] ** 2) * (X[i, d] - X[j, d])


fn gradient_descent[dtype: DType, nelts: Int](
        inout X: Matrix[dtype], 
        D: Matrix[dtype],
        learning_rate: SIMD[dtype, 1], 
        num_iterations: Int
    ):

    var grad = Matrix[dtype](X.rows, X.cols)

    for _ in range(num_iterations):
        grad.zeros()
        compute_gradient[dtype](grad, X, D)
        for r in range(X.rows):
            for c in range(X.cols):
                X[r, c] -= learning_rate * grad[r, c]

    
    # ## Using extended matrix methods
    # for _ in range(num_iterations):
    #     grad.zeros()
    #     compute_gradient[dtype](grad, X, D)
    #     X -= learning_rate * grad


    # ## Parallel gradient computation
    # for _ in range(num_iterations):
    #     grad.zeros()
    #     compute_gradient[dtype, nelts](grad, X, D)
    #     for r in range(X.rows):
    #         for c in range(X.cols):
    #             X[r, c] -= learning_rate * grad[r, c]



### Vector & Parallel

fn compute_gradient[dtype: DType, nelts: Int](inout grad: Matrix[dtype], X: Matrix[dtype], D: Matrix[dtype]):
    
    @parameter
    fn calc_row(i: Int):
        var squared_distance: SIMD[dtype, 1] = 0

        for j in range(X.rows):
            squared_distance = 0
            for d in range(X.cols):
                squared_distance += (X[i, d] - X[j, d])**2

            for d in range(X.cols):
                grad[i, d] += 4 * (squared_distance - D[i, j] ** 2) * (X[i, d] - X[j, d])

            # @parameter
            # fn grad_vector[nelts: Int](d: Int):
            #     grad.store[nelts](
            #         i, d, grad.load[nelts](i, d) + 4 * (squared_distance - D[i, j] ** 2) * (X.load[nelts](i, d) - X.load[nelts](j, d))
            #     )
                
            # vectorize[nelts, grad_vector](X.cols)

    parallelize[calc_row](X.rows, X.rows)

