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
from casadi import *

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
    
    
    def grad_casadi_slow():  # was slow
        X = SX.sym('X', 10000,3)
        x = X[:,0]
        y = X[:,1]
        c = X[:,2]
        z = (x*y+c)*c + x
        idxs = np.arange(X.shape[0])
        jac = jacobian(z, X)
        djdx = vertcat(*[jac[idx,idx] for idx in idxs])
        djdy = vertcat(*[jac[idx,idx+10000] for idx in idxs])
        djdc = vertcat(*[jac[idx,idx+20000] for idx in idxs])
        jacf = Function('gradf', [X], [horzcat(djdx, djdy, djdc)])
        return jacf
    
    def grad_casadi():      # also slow
        X = SX.sym('X', 10000,3)
        x = X[:,0]
        y = X[:,1]
        c = X[:,2]
        z = (x*y+c)*c + x
        gradients = []
        for i in range(10000):
            z += x[i]*y[i]
            grad_z_i = jacobian(z[i], X[i])
            gradients.append(grad_z_i)
        print("start compile")
        jacf = Function('gradf', [X], [vertcat(*gradients)])
        print("end compile")
        return jacf
    grad_func_casadi = grad_casadi()
    grad_val_casadi = grad_func_casadi(np.array(X))
    
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
    secs_casadi = benchmark_function(grad_func_casadi, np.array(X), name="casadi")

    print(f"Gain of {secs_orig/secs_jaxjit} for autodiff_jax_jit")
    