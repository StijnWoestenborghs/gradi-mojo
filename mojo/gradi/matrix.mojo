from memory import memset_zero
from random import rand


struct Matrix[dtype: DType, n_rows: Int, n_cols: Int]:
    var data: DTypePointer[dtype]
    var rows: Int
    var cols: Int

    fn __init__(inout self):
        self.data = DTypePointer[dtype].alloc(n_rows * n_cols)
        self.rows = n_rows
        self.cols = n_cols
        
    fn __del__(owned self):
        self.data.free()

    fn zero(inout self):
        memset_zero(self.data, n_rows * n_cols)

    @always_inline
    fn __getitem__(inout self, y: Int, x: Int) -> SIMD[dtype, 1]:
        return self.load[1](y, x)

    @always_inline
    fn __setitem__(inout self, y: Int, x: Int, val: SIMD[dtype, 1]):
        return self.store[1](y, x, val)

    @always_inline
    fn __copyinit__(inout self, other: Matrix[dtype, n_rows, n_cols]):
        self.data = DTypePointer[dtype].alloc(n_rows * n_cols)
        for y in range(n_rows):
            for x in range(n_cols):
                self.data.simd_store[1](y * n_cols + x, other.data.simd_load[1](y * n_cols + x))
        self.rows = other.rows
        self.cols = other.cols

    @always_inline
    fn load[nelts: Int](inout self, y: Int, x: Int) -> SIMD[dtype, nelts]:
        return self.data.simd_load[nelts](y * n_cols + x)

    @always_inline
    fn store[nelts: Int](inout self, y: Int, x: Int, val: SIMD[dtype, nelts]):
        return self.data.simd_store[nelts](y * n_cols + x, val)

    @always_inline
    fn __str__(inout self) -> String:
        """
        Until mojo has traits, there isn't a clean implementation of __str__ and __repr__ that are usable for a polymorphic implementation of print.
        https://github.com/modularml/mojo/discussions/325 --> use print(X.__str__()) instead
        """
        var row_str: String
        var matrix_str: String = "["
        
        for y in range(n_rows):
            row_str = "[ " + String(self[y, 0])
            for x in range(1, n_cols):
                row_str += ", " + String(self[y, x])
            row_str += "]\n "
            matrix_str += row_str
        matrix_str = matrix_str[:-2] + "]"

        return matrix_str

    fn T(owned self) -> Matrix[dtype, n_cols, n_rows]:
        var transposed = Matrix[dtype, n_cols, n_rows]()
        for y in range(n_cols):
            for x in range(n_rows):
                transposed[y, x] = self[x, y]

        return transposed









# # Parallelized + Vectorized
# fn matmul(C: Matrix, A: Matrix, B: Matrix, rt: Runtime):
#     @parameter
#     fn calc_row(m: Int):
#         for k in range(A.cols):

#             @parameter
#             fn dot[nelts: Int](n: Int):
#                 C.store[nelts](
#                     m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
#                 )

#             vectorize[nelts, dot](C.cols)

#     parallelize[calc_row](rt, C.rows)