from python.python import Python
from algorithm import vectorize, parallelize, vectorize_unroll
from memory import memset_zero, memset
from random import rand



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

    fn zeros(inout self):
        memset_zero(self.data, self.rows * self.cols)

    fn ones(inout self):
        # memset(self.data, 1, self.rows * self.cols)    # v0.4.0: memset only takes in SIMD[ui8, 1] for now
        for y in range(self.rows):
            for x in range(self.cols):
                self[y, x] = 1
        
    fn rand(inout self):
        rand(self.data, self.rows*self.cols)

    @always_inline
    fn __getitem__(self, y: Int, x: Int) -> SIMD[dtype, 1]:
        return self.load[1](y, x)

    @always_inline
    fn __setitem__(inout self, y: Int, x: Int, val: SIMD[dtype, 1]):
        self.store[1](y, x, val)

    @always_inline
    fn __copyinit__(inout self, other: Matrix[dtype]):
        self.data = DTypePointer[dtype].alloc(other.rows * other.cols)
        for y in range(other.rows):
            for x in range(other.cols):
                self.data.simd_store[1](y * other.cols + x, other.data.simd_load[1](y * other.cols + x))
        self.rows = other.rows
        self.cols = other.cols

    @always_inline
    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[dtype, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    @always_inline
    fn store[nelts: Int](inout self, y: Int, x: Int, val: SIMD[dtype, nelts]):
        return self.data.simd_store[nelts](y * self.cols + x, val)

    @always_inline
    fn __str__(self) -> String:
        """
        v0.4.0
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

    fn T(self) -> Matrix[dtype]:
        var transposed = Matrix[dtype](self.cols, self.rows)
        for y in range(self.cols):
            for x in range(self.rows):
                transposed[y, x] = self[x, y]

        return transposed
    
    fn dot[nelts: Int](inout self, other: Matrix[dtype]) -> Matrix[dtype]:
        var C = Matrix[dtype](self.rows, other.cols)
        C.zeros()

        @parameter
        fn calc_row(m: Int):
            for k in range(self.cols):

                @parameter
                fn dot[nelts: Int](n: Int):
                    C.store[nelts](
                        m, n, C.load[nelts](m, n) + self[m, k] * other.load[nelts](k, n)
                    )

                vectorize[nelts, dot](C.cols)

        parallelize[calc_row](C.rows, C.rows)
        
        return C

    @always_inline
    fn __mul__(self, scalar: SIMD[dtype, 1]) -> Matrix[dtype]:
        var res = Matrix[dtype](self.rows, self.cols)
        for y in range(self.rows):
            for x in range(self.cols):
                res[y, x] = scalar * self.data.simd_load[1](y * self.cols + x)
        return res

    @always_inline
    fn __rmul__(self, scalar: SIMD[dtype, 1]) -> Matrix[dtype]:
        return self.__mul__(scalar)

    @always_inline
    fn __add__(self, other: Matrix[dtype]) -> Matrix[dtype]:
        var res = Matrix[dtype](self.rows, self.cols)
        for y in range(self.rows):
            for x in range(self.cols):
                res[y, x] = self.data.simd_load[1](y * self.cols + x) + other.data.simd_load[1](y * self.cols + x)
        return res

    @always_inline
    fn __iadd__(inout self, owned other: Matrix[dtype]):
        for y in range(self.rows):
            for x in range(self.cols):
                self[y, x] +=  other[y, x]

    @always_inline
    fn __isub__(inout self, owned other: Matrix[dtype]):
        for y in range(self.rows):
            for x in range(self.cols):
                self[y, x] -=  other[y, x]

    def to_python(self) -> PythonObject:
        try:
            np = Python.import_module("numpy")
            pymatrix = np.zeros((self.rows, self.cols), np.float64)
            for y in range(self.rows):
                for x in range(self.cols):
                    pymatrix.itemset((y, x), self[y, x])
            return pymatrix
        except:
            raise Error("Failed to convert Matrix to PythonObject")