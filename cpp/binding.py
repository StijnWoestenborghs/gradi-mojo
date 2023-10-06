import os
import json
import ctypes
from ctypes import *
import numpy as np


libc = CDLL("cpp/build/lib/gradient_descent.so")
run_binding_external = libc.run_binding_external
run_binding_external.argtypes = [c_char_p, c_int, POINTER(c_char_p), POINTER(c_int)]
run_binding_external.restype = c_int

delete_c_return = libc.delete_c_return
delete_c_return.argtypes = [c_char_p]
delete_c_return.restype = None


def run_binding(input_string):
    s_return = c_char_p()
    length_return = c_int()
    code = run_binding_external(input_string, len(input_string), pointer(s_return), pointer(length_return))
    if code == 0:
        result = string_at(s_return, length_return)
        delete_c_return(s_return)
        return result
    else:
        raise Exception('Error in binding')


def gradient_descent_cpp(X, D, learning_rate, num_iterations):
    # Serialize input
    # Flatten input matrices to 1D arrays
    input_json = json.dumps(
        {
            "N": int(X.shape[0]),
            "dim": int(X.shape[1]),
            "X": X.ravel().tolist(),
            "D": D.ravel().tolist(),
            "learning_rate": float(learning_rate),
            "num_iterations": int(num_iterations)
        }
    )

    # call python-cpp binding
    result = run_binding(input_json.encode('utf-8'))

    # Deserialize output
    output_json = json.loads(result.decode('utf-8'))
    X_out = output_json["X"]
    X_out = np.reshape(np.array(X_out), (X.shape[1], X.shape[0])).T

    return X_out
