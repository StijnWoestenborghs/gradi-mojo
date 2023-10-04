from memory import memset_zero
from random import rand
from algorithm import vectorize, parallelize, vectorize_unroll
from runtime.llcl import Runtime
from python.python import Python


alias dtype = DType.float32
alias type = Float32           # equals to SIMD[DType.float32, 1]


struct Matrix:
    var data: DTypePointer[dtype]
    var rows: Int
    var cols: Int

    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[dtype].alloc(rows * cols)
        rand(self.data, rows * cols)
        self.rows = rows
        self.cols = cols

    fn __del__(owned self):
        self.data.free()

    fn zero(inout self):
        memset_zero(self.data, self.rows * self.cols)

    @always_inline
    fn __getitem__(self, y: Int, x: Int) -> type:
        return self.load[1](y, x)

    @always_inline
    fn __setitem__(self, y: Int, x: Int, val: type):
        return self.store[1](y, x, val)

    @always_inline
    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[dtype, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    @always_inline
    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[dtype, nelts]):
        return self.data.simd_store[nelts](y * self.cols + x, val)

    fn __str__(inout self) -> String:
        """
        Until mojo has traits, there isn't a clean implementation of __str__ and __repr__ that are usable for a polymorphic implementation of print.
        https://github.com/modularml/mojo/discussions/325 --> use print(X.__str__()) instead
        TODO: implement as joint function ones lists are supported
        """
        var row_str: String
        var matrix_str: String = "["
        
        for y in range(self.rows):
            row_str = "[ " + String(self.__getitem__(y, 0))
            for x in range(1, self.cols):
                row_str += ", " + String(self.__getitem__(y, x))
            row_str += "]\n "
            matrix_str += row_str
        matrix_str = matrix_str[:-2] + "]"

        return matrix_str


# The SIMD vector width of your machine
alias nelts = simdwidthof[dtype]()  

# Parallelized + Vectorized
fn matmul(C: Matrix, A: Matrix, B: Matrix, rt: Runtime):
    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[nelts, dot](C.cols)

    parallelize[calc_row](rt, C.rows)


fn loss(X: Matrix, D: Matrix) -> type:
    let N = X.rows
    let dim = X.cols
    var squared_distance: type = 0
    var total_loss: type = 0

    for i in range(N):
        for j in range(N):

            squared_distance = 0
            for d in range(dim):
                squared_distance += (X[i, d] - X[j, d])**2

            total_loss += (squared_distance - D[i, j]**2)**2
    
    return total_loss


fn compute_gradient(inout grad: Matrix, X: Matrix, D: Matrix, _rt: Runtime):
    let N = X.rows
    let dim = X.cols
    var squared_distance: type = 0
    grad.zero()

    for i in range(N):
        for j in range(N):
            squared_distance = 0
            for d in range(dim):
                squared_distance += (X[i, d] - X[j, d])**2

            for d in range(dim):
                grad[i, d] += 4 * (squared_distance - D[i, j] ** 2) * (X[i, d] - X[j, d])


fn gradient_descent(inout X: Matrix, D: Matrix, rt: Runtime, learning_rate: type = 0.0001, num_iterations: Int = 1000):
    let N = X.rows
    let dim = X.cols
    var grad = Matrix(N, dim)

    for _ in range(num_iterations):
        compute_gradient(grad, X, D, rt)
        for r in range(X.rows):
            for c in range(X.cols):
                X[r, c] -= learning_rate * grad[r, c]


### Vector & Parallel

fn compute_gradient_vp(inout grad: Matrix, X: Matrix, D: Matrix, rt: Runtime):
    let N = X.rows
    let dim = X.cols
    var squared_distance: type = 0
    grad.zero()

    @parameter
    fn calc_row(i: Int):
        for j in range(N):
            squared_distance = 0
            for d in range(dim):
                squared_distance += (X[i, d] - X[j, d])**2

            @parameter
            fn grad_vector[nelts: Int](d: Int):
                grad.store[nelts](
                    i, d, grad.load[nelts](i, d) + 4 * (squared_distance - D[i, j] ** 2) * (X.load[nelts](i, d) - X.load[nelts](j, d))
                )
                
            vectorize[nelts, grad_vector](dim)

    parallelize[calc_row](rt, N)


fn gradient_descent_vp(inout X: Matrix, D: Matrix, rt: Runtime, learning_rate: type = 0.0001, num_iterations: Int = 1000):
    let N = X.rows
    let dim = X.cols
    var grad = Matrix(N, dim)

    for _ in range(num_iterations):
        # compute_gradient(grad, X, D, rt)
        compute_gradient_vp(grad, X, D, rt)
        
        @parameter
        fn calc_row(r: Int):
    
            @parameter
            fn grad_vector[nelts: Int](c: Int):
                X.store[nelts](
                    r, c, X.load[nelts](r, c) - learning_rate * grad.load[nelts](r, c)
                )

            vectorize[nelts, grad_vector](X.cols)

        parallelize[calc_row](rt, X.rows)
