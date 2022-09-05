import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix
from .utils import GFG

class GeneratorBase(object):
    def __init__(self):
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def minimum_dist_pairing(self, samples, anchor_samples):
        Xs = samples.detach().numpy()
        Ys = anchor_samples.detach().numpy()
        dist_mat = distance_matrix(Xs, Ys)
        import ipdb; ipdb.set_trace()
        g = GFG(-dist_mat)
        return g.compute_row_indices()

    def loss(self, gt_samples):
        nSamples = len(gt_samples)
        gt_samples = torch.tensor(gt_samples, dtype=torch.float32)
        pred_samples = self.forward(nSamples)
        shuffled_ids = self.minimum_dist_pairing(pred_samples, gt_samples)
        shuffled_pred_samples = pred_samples[shuffled_ids, :]
        loss = self.mse_loss(shuffled_pred_samples, gt_samples)
        return loss

    def show(self):
        nSamples = 500
        pred_samples = self.forward(nSamples)
        pred_samples = pred_samples.detach().numpy()
        Xs, Ys = pred_samples[:,0], pred_samples[:,1]
        plt.figure(figsize=(8,8))
        plt.scatter(Xs, Ys)
        plt.title('swiss roll')
        plt.show()


class MLPGenerator(torch.nn.Module, GeneratorBase):
    def __init__(self):
        super().__init__()
        self.generator = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.Sigmoid(),
            torch.nn.Linear(32, 2),
        )

    def forward(self, nSamples):
        standard_gauss_samples = np.random.randn(nSamples, 2)
        noise_seed = torch.tensor(standard_gauss_samples, dtype=torch.float32)
        pred_samples = self.generator(noise_seed)
        return pred_samples
