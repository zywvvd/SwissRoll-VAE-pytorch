import numpy as np

class MCMCSampler(object):
    def __init__(self, func, x0, burning_iters=500):
        self._func = func
        self._x0 = x0
        self._burning_iters = burning_iters

    def sample(self, num=1000):
        # burn the first lot of samples
        x = self._x0
        for _ in range(self._burning_iters):
            x = self._mcmc_iter(x)

        samples = list()
        for _ in range(num):
            x = self._mcmc_iter(x)
            samples.append(x)

        return samples

    def _mcmc_iter(self, x_curr):
        # kernel distribution
        x_next = np.random.rand() + x_curr
        y_curr = self._func(x_curr)
        y_next = self._func(x_next)

        if y_curr == 0:
            return x_next
        else:
            alpha = y_next / y_curr
            if np.random.rand() <= alpha:
                return x_next
            else:
                return x_curr
        