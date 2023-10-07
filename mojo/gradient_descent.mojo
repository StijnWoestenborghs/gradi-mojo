from algorithm import vectorize, parallelize, vectorize_unroll
from python.python import Python
from tensor import Tensor
from utils.index import Index

from mojo.tensorutils import zero



fn loss[dtype: DType](X: Tensor[dtype], D: Tensor[dtype]) -> SIMD[dtype, 1]:
    var total_loss: SIMD[dtype, 1] = 0
    var squared_distance: SIMD[dtype, 1] = 0
    let shape = X.shape()

    for i in range(shape[0]):
        for j in range(shape[0]):
            squared_distance = 0
            for d in range(shape[1]):
                squared_distance += (X[i, d] - X[j, d])**2
            
            total_loss += (squared_distance - D[i, j]**2)**2

    return total_loss


fn compute_gradient[dtype: DType](inout grad: Tensor[dtype], X: Tensor[dtype], D: Tensor[dtype]):
    var squared_distance: SIMD[dtype, 1]
    let shape = X.shape()

    for i in range(shape[0]):
        for j in range(shape[0]):
            squared_distance = 0
            for d in range(shape[1]):
                squared_distance += (X[i, d] - X[j, d])**2

            for d in range(shape[1]):
                grad[Index(i, d)] += 4 * (squared_distance - D[i, j] ** 2) * (X[i, d] - X[j, d])


fn gradient_descent[dtype: DType, nelts: Int](
        inout X: Tensor[dtype], 
        D: Tensor[dtype],
        learning_rate: SIMD[dtype, 1], 
        num_iterations: Int
    ):

    let shape = X.shape()
    var grad = Tensor[dtype](shape[0], shape[1])

    for _ in range(num_iterations):
        zero(grad)
        compute_gradient[dtype](grad, X, D)
        for r in range(shape[0]):
            for c in range(shape[1]):
                X[Index(r, c)] -= learning_rate * grad[r, c]

    
    # ## Using extended matrix methods
    # for _ in range(num_iterations):
    #     zero(grad)
    #     compute_gradient[dtype](grad, X, D)
    #     X -= learning_rate * grad


    # ## Parallel gradient computation
    # for _ in range(num_iterations):
    #     zero(grad)
    #     compute_gradient[dtype, nelts](grad, X, D)
    #     for r in range(shape[0]):
    #         for c in range(shape[1]):
    #             X[Index(r, c)] -= learning_rate * grad[r, c]



### Vector & Parallel

fn compute_gradient[dtype: DType, nelts: Int](inout grad: Tensor[dtype], X: Tensor[dtype], D: Tensor[dtype]):
    let shape = X.shape()

    @parameter
    fn calc_row(i: Int):
        var squared_distance: SIMD[dtype, 1] = 0

        for j in range(shape[0]):
            squared_distance = 0
            for d in range(shape[1]):
                squared_distance += (X[i, d] - X[j, d])**2

            for d in range(shape[1]):
                grad[Index(i, d)] += 4 * (squared_distance - D[i, j] ** 2) * (X[i, d] - X[j, d])

            # @parameter
            # fn grad_vector[nelts: Int](d: Int):
            #     grad.store[nelts](
            #         i, d, grad.load[nelts](i, d) + 4 * (squared_distance - D[i, j] ** 2) * (X.load[nelts](i, d) - X.load[nelts](j, d))
            #     )
                
            # vectorize[nelts, grad_vector](X.cols)

    parallelize[calc_row](shape[0], shape[0])

