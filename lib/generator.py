import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix
from .utils import hungarian_match

class BestMatchGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def minimum_dist_pairing(self, samples, anchor_samples):
        Xs = samples.detach().numpy()
        Ys = anchor_samples.detach().numpy()
        dist_mat = distance_matrix(Xs, Ys)
        indices = hungarian_match(dist_mat)
        return indices

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
        plt.figure(111, figsize=(8,8))
        plt.clf()
        plt.scatter(Xs, Ys)
        plt.title(type(self).__name__)
        plt.pause(.1)


class MLPGenerator(BestMatchGenerator):
    def __init__(self):
        super().__init__()
        self.noise_dim = 2
        hidden_dim = 512
        self.generator = torch.nn.Sequential(
            torch.nn.Linear(self.noise_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 2),
        )

    def forward(self, nSamples):
        standard_gauss_samples = np.random.randn(nSamples, self.noise_dim)
        noise_seed = torch.tensor(standard_gauss_samples, dtype=torch.float32)
        pred_samples = self.generator(noise_seed)
        return pred_samples
