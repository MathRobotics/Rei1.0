import liteopt
import numpy as np

target = np.array([1.0, -2.0])

def residual(x):
    return x - target

def jacobian(x):
    return np.eye(2)

x_star, cost, iters, r_norm, dx_norm, ok = liteopt.gn(residual, jacobian, x0=np.zeros((2,)), max_iters=100, lambda_=1e-3, verbose=True)
print(ok, x_star, cost)