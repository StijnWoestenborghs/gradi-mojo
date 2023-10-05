# STOLEN CODE: <O.O> 
#
# [SOURCE]
# https://github.com/Jmkernes/Automatic-Differentiation/blob/main/AutomaticDifferentiation.ipynb
# https://towardsdatascience.com/build-your-own-automatic-differentiation-program-6ecd585eec2a

import numpy as np
from autodiff import *
from timeit import timeit
import jax
import jax.numpy as jnp


if __name__ == "__main__":
    ### TEST AUTODIFF
    key = jax.random.PRNGKey(0)
    X = jax.random.uniform(key, (10000, 3))
    def autodiff_original(X):
        with Graph() as g:
            x = Variable(X[:,0], name='x')
            y = Variable(X[:,1], name='y')
            c = Constant(X[:,2], name='c')
            z = (x*y+c)*c + x

            order = topological_sort(head_node=z, graph=g)
            res = forward_pass(order)
            grads = backward_pass(order)
            return grads
    grads = autodiff_original(X)
    
    def autodiff_jax():
        def z(x,y,c):
            return (x*y+c)*c + x
        grad_func = jax.grad(z, argnums=(0,1,2))
        return jax.vmap(grad_func, in_axes=(0, 0, 0))
    grad_func = autodiff_jax()
    grad_val = grad_func(X[:,0], X[:,1], X[:,2])

    def autodiff_jax_jit():
        def z(x,y,c):
            return (x*y+c)*c + x
        grad_func = jax.grad(z, argnums=(0,1,2))
        return jax.jit(jax.vmap(grad_func, in_axes=(0, 0, 0)))
    grad_jit = autodiff_jax_jit()
    grad_val_jit = grad_jit(X[:,0], X[:,1], X[:,2])

    ### COMPARE
    def benchmark_function(f, *args, name, iterations=1000):
        secs = timeit(lambda: f(*args), number=iterations)
        print(f"Total time for {iterations} iterations for method {name}: {secs}")
        return secs
    secs_orig = benchmark_function(autodiff_original, X, name="autodiff_original")
    #secs_jax = benchmark_function(grad_func, val1, val2, val3, name="autodiff_jax")
    secs_jaxjit = benchmark_function(grad_jit, X[:,0], X[:,1], X[:,2], name="autodiff_jax_jit")

    print(f"Gain of {secs_orig/secs_jaxjit} for autodiff_jax_jit")
    