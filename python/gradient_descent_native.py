import random


class PyMatrix:
    def __init__(self, value, rows, cols):
        self.value = value
        self.rows = rows
        self.cols = cols
        self.shape = (rows, cols)

    def __getitem__(self, idxs):
        if isinstance(idxs, tuple):
            return self.value[idxs[0]][idxs[1]]
        elif isinstance(idxs, int):
            return PyMatrix([self.value[idxs]], 1, self.cols)

    def __setitem__(self, idxs, value):
        self.value[idxs[0]][idxs[1]] = value

    def __add__(self, other):              
        return PyMatrix(
            [[self[i, j] + other[i, j] for j in range(self.cols)] for i in range(self.rows)], 
            self.rows, 
            self.cols
        )

    def __sub__(self, other):
        return PyMatrix(
            [[self[i, j] - other[i, j] for j in range(self.cols)] for i in range(self.rows)], 
            self.rows, 
            self.cols
        )
    
    def __str__(self):
        rows_str = ['[' + ', '.join(map(str, row)) + ']' for row in self.value]
        matrix_str = '[' + ',\n '.join(rows_str) + ']'
        return matrix_str
    
    def T(self):
        return PyMatrix(
            [[self[i, j] for i in range(self.rows)] for j in range(self.cols)],
            self.cols,
            self.rows
        )
    
    def copy(self):
        return PyMatrix(
            [[self[i, j] for j in range(self.cols)] for i in range(self.rows)], 
            self.rows, 
            self.cols
        )


def matmul_native(A, B):
    """
    The problem here is that you need to initialize a result matrix of zeros which cost resources.
    Numpy (optimized C in the backend) handles this by allocating memory directly which is way more efficient.
    Note that the matmul example in mojo did exclude this from teh benchmark
    """
    C = PyMatrix([[0.0] * B.cols for _ in range(A.rows)], A.rows, B.cols)

    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]

    return C


def loss_native(X, D):
    N = X.rows
    total_loss = 0
   
    for i in range(N):
        for j in range(N):
            difference = X[i] - X[j]
            squared_distance = matmul_native(difference, difference.T())[0, 0]
            total_loss += (squared_distance - D[i, j]**2)**2

    return total_loss


def compute_gradient_native(X, D):
    N = X.rows
    dim = X.cols
    grad = PyMatrix([[0.0] * dim for _ in range(N)], N, dim)

    for i in range(N):
        for j in range(N):
            difference = X[i] - X[j]
            squared_distance = matmul_native(difference, difference.T())[0, 0]
            for d in range(dim):
                grad[i, d] += 4 * (squared_distance - D[i, j] ** 2) * difference[0, d]           
                
    return grad


def gradient_descent_native(D, learning_rate=0.0001, num_iterations=1000):
    dim = 2
    N = D.rows
    X = PyMatrix([[random.random() for _ in range(dim)] for _ in range(N)], N, dim)

    for i in range(num_iterations):
        grad = compute_gradient_native(X, D)
        for r in range(X.rows):
            for c in range(X.cols):
                X[r, c] -= learning_rate * grad[r, c]

    return X


def gradient_descent_native_cache(D, learning_rate=0.0001, num_iterations=1000):
    dim = 2
    N = D.rows
    X = PyMatrix([[random.random() for _ in range(dim)] for _ in range(N)], N, dim)

    positions_over_time = []
    loss_over_time = []

    for i in range(num_iterations):
        positions_over_time.append(X.copy().value)
        loss_over_time.append(loss_native(X, D))

        grad = compute_gradient_native(X, D)
        for r in range(X.rows):
            for c in range(X.cols):
                X[r, c] -= learning_rate * grad[r, c]
   
    positions_over_time.append(X.copy().value)
    loss_over_time.append(loss_native(X, D))
    
    return positions_over_time, loss_over_time