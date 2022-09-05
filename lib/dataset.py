from sklearn.datasets import make_swiss_roll
import numpy as np
import matplotlib.pyplot as plt

class DatasetBase(object):
    def batch(self, size=320):
        for idx in range(0, len(self.samples), size):
            yield self.samples[idx:idx+size]

    def __len__(self):
        return len(self.samples)

    def show(self):
        nSamples = max(len(self), 500)
        Xs, Ys = self.samples[:nSamples,0], self.samples[:nSamples,1]
        plt.figure(123, figsize=(8,8))
        plt.clf()
        plt.scatter(Xs, Ys)
        plt.title(type(self).__name__)
        plt.pause(.1)


class DatasetSwissRoll(DatasetBase):
    def __init__(self, size=1024):
        swiss_roll_samples, _ = make_swiss_roll(size, noise=1, random_state=123)
        self.samples = swiss_roll_samples[:,[0,2]]


class DatasetUniform(DatasetBase):
    def __init__(self, size=1024):
        self.samples = np.random.rand(size, 2)


class DatasetGaussian(DatasetBase):
    def __init__(self, size=1024):
        self.samples = np.random.randn(size, 2) * 5