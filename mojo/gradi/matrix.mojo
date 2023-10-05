from algorithm import vectorize, parallelize, vectorize_unroll
from memory import memset_zero
from random import rand
from runtime.llcl import Runtime



struct Matrix[dtype: DType]:
    var data: DTypePointer[dtype]
    var rows: Int
    var cols: Int

    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[dtype].alloc(rows * cols)
        self.rows = rows
        self.cols = cols
        
    fn __del__(owned self):
        self.data.free()

    fn zero(inout self):
        memset_zero(self.data, self.rows * self.cols)

    @always_inline
    fn __getitem__(inout self, y: Int, x: Int) -> SIMD[dtype, 1]:
        return self.load[1](y, x)

    @always_inline
    fn __setitem__(inout self, y: Int, x: Int, val: SIMD[dtype, 1]):
        return self.store[1](y, x, val)

    @always_inline
    fn __copyinit__(inout self, other: Matrix[dtype]):
        self.data = DTypePointer[dtype].alloc(other.rows * other.cols)
        for y in range(other.rows):
            for x in range(other.cols):
                self.data.simd_store[1](y * other.cols + x, other.data.simd_load[1](y * other.cols + x))
        self.rows = other.rows
        self.cols = other.cols

    @always_inline
    fn load[nelts: Int](inout self, y: Int, x: Int) -> SIMD[dtype, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    @always_inline
    fn store[nelts: Int](inout self, y: Int, x: Int, val: SIMD[dtype, nelts]):
        return self.data.simd_store[nelts](y * self.cols + x, val)

    @always_inline
    fn __str__(inout self) -> String:
        """
        Until mojo has traits, there isn't a clean implementation of __str__ and __repr__ that are usable for a polymorphic implementation of print.
        https://github.com/modularml/mojo/discussions/325 --> use print(X.__str__()) instead
        """
        var row_str: String
        var matrix_str: String = "["
        
        for y in range(self.rows):
            row_str = "[ " + String(self[y, 0])
            for x in range(1, self.cols):
                row_str += ", " + String(self[y, x])
            row_str += "]\n "
            matrix_str += row_str
        matrix_str = matrix_str[:-2] + "]"

        return matrix_str

    fn T(inout self) -> Matrix[dtype]:
        var transposed = Matrix[dtype](self.cols, self.rows)
        for y in range(self.cols):
            for x in range(self.rows):
                transposed[y, x] = self[x, y]

        return transposed
    
    fn dot[nelts: Int](inout self, other: Matrix[dtype], rt: Runtime) -> Matrix[dtype]:
        var C = Matrix[dtype](self.rows, other.cols)
        C.zero()

        @parameter
        fn calc_row(m: Int):
            for k in range(self.cols):

                @parameter
                fn dot[nelts: Int](n: Int):
                    C.store[nelts](
                        m, n, C.load[nelts](m, n) + self[m, k] * other.data.simd_load[nelts](k * other.cols + n)
                    )

                vectorize[nelts, dot](C.cols)

        parallelize[calc_row](rt, C.rows)
        
        return C
