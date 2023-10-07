from random import rand
from algorithm import vectorize
from tensor import Tensor, TensorShape
from memory import memset_zero, memset


@always_inline
fn zero[dtype: DType](inout t: Tensor[dtype]):
    memset_zero[dtype](t.data(), t.num_elements())

fn ones[dtype: DType](inout t: Tensor[dtype]):
    fill[dtype](t, 1)

fn fill[dtype: DType](inout t: Tensor[dtype], val: SIMD[dtype, 1]):
    alias nelts: Int = simdwidthof[dtype]()
    @parameter
    fn fill_vec[nelts: Int](idx: Int):
        t.simd_store[nelts](idx, t.simd_load[nelts](idx).splat(val))
    vectorize[nelts, fill_vec](t.num_elements())

@always_inline
fn randt[dtype: DType](inout t: Tensor[dtype]):
    rand(t.data(), t.num_elements())

# fn repeat_tab(scalar: Int) -> String:
#     var result: String = ""
#     for i in range(scalar):
#         result += "\t"
#     return result

# fn find_first(s: String, char: String) ->  Int:
#     for i in range(len(s)):
#         if s[i] == char:
#             return i
#     return -1

# fn round4[dtype: DType](num: SIMD[dtype, 1]) -> String:
#     var s: String = String(num)
#     let pos: Int = find_first(s, ".")
#     if pos != -1:
#         return  s[:pos+5]
#     else:
#         return s





fn tprint[dtype: DType](t: Tensor[dtype], indent: Int = 0):
    let n: Int = t.num_elements()
    let shape = t.shape()
    var s: String

    if t.rank() == 0:
        s = String(t[0])
        print(s)
    elif t.rank() == 1:
        s = "[" + String(t[0])
        for i in range(1, shape[0]):
            s += "\t" + String(t[i])
        s += "]"
        print(s)
    #TODO: Implement recursive from here
    # else:
    #     print(repeat_tab(indent), "[")
    #     for i in range(shape[0]):
    #         ## TODO: select sub tensor of lower rank
    #         # tprint[dtype](sub_tensor, indent + 1)
            
    #     print(repeat_tab(indent), "]")
    
    elif t.rank() == 2:
        var srow: String
        
        s = "["
        for i in range(shape[0]):
            srow = "[" + String(t[i, 0])
            for j in range(1, shape[1]):
                srow += "\t" + String(t[i, j])
            srow += "]\n "
            s += srow
        s = s[:-2] + "]"
        print(s)

    elif t.rank() == 3:
        var smat: String
        var srow: String

        s = "[\n"
        for i in range(shape[0]):
            smat = "    ["
            for j in range(shape[1]):
                srow = "[" + String(t[i, j, 0])
                for k in range(1, shape[2]):
                    srow += "\t" + String(t[i, j, k])
                srow += "]\n     "
                smat += srow
            smat = smat[:-6] + "]"
            s += smat + "\n\n"
        s = s[:-1] + "]"
        print(s)
            
    print_no_newline("Tensor shape:", t.shape().__str__(), ", ")
    print_no_newline("Tensor rank:", t.rank(), ", ")
    print_no_newline("DType:", t.type().__str__(), "\n\n")
