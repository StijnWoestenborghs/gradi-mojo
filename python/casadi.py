from casadi import *

def casadi_solve(X_init, D):
    N = D.shape[0]
    opti = Opti()
    X = opti.variable(N, 2)
    loss = 0
    for i in range(N):
        for j in range(N):
            dx = X[i,0] - X[j,0]
            dy = X[i,1] - X[j,1]
            squared_distance = dx**2 + dy**2
            loss += (squared_distance - D[i, j]**2)**2
    opti.minimize(loss)
    
    opti.solver('ipopt')
    opti.set_initial(X, X_init)
    p_opts = {}
    s_opts = {"max_iter": 100}
    
    opti.solver("ipopt",p_opts,
                    s_opts)
    sol = opti.solve()
    return sol.value(X)
