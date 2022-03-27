import torch
import numpy as np
from tqdm import tqdm
from mcmc import MCMCSampler
import matplotlib.pyplot as plt


class Exp(torch.nn.Module):
    def forward(self, x):
        return torch.exp(x)


class GibbsDistModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f = torch.nn.Sequential(
            torch.nn.Linear(4, 1),
            Exp()
        )
        self._mcmc_sampler = MCMCSampler(self.func, 0)

    def forward(self, x):
        feature = torch.stack([x, x**2, x**3, x**4], dim=1)
        return self.f(feature)

    def func(self, x):
        x = torch.tensor(x, dtype=torch.float32).reshape([-1])
        y = self.forward(x)
        return float(y.detach().numpy())

    def draw(self):
        with torch.no_grad():
            x = torch.linspace(-5, 5, 500)
            f = self.forward(x)
            f = f.detach().numpy()
            x = x.detach().numpy()

        plt.figure(1, figsize=(12, 8))
        plt.clf()
        plt.plot(x, f)
        plt.xlim([-5, 5])
        plt.title('min/max E(x): {:.3f}/{:.3f}'.format(f.min(), f.max()))
        plt.pause(.5)



class Dataset:
    def __init__(self, size=1024):
        self.samples = 3 * np.random.randn(size).astype('float32') + 1.5

    def batch(self, size=320):
        for idx in range(0, len(self.samples), size):
            yield self.samples[idx:idx+size]

    def __len__(self):
        return len(self.samples)


#
# Start training
#
data = Dataset()
model = GibbsDistModel()

lr = 1e-3

for iepoch in tqdm(range(1000)):

    # visualize results
    model.draw()

    Xs = model._mcmc_sampler.sample()

    ## MCMC to simulate the expectation loss
    expect_param_grads = dict()
    for x in Xs:
        x = torch.tensor(x, dtype=torch.float32).reshape([-1])
        model.zero_grad()
        f = model(x)
        f.backward()
        with torch.no_grad():
            for param in model.parameters():
                grad = param.grad.clone()
                gdf = (grad / f).reshape_as(param)
                if grad not in expect_param_grads:
                    expect_param_grads[param]  = gdf
                else:
                    expect_param_grads[param] += gdf
    
    with torch.no_grad():
        for param in expect_param_grads:
            expect_param_grads[param] /= len(Xs)


    param_grads = list()
    batch_size = 1
    for batch in data.batch(batch_size):
        model.zero_grad()
        x = torch.tensor(batch)
        f = model(x)
        f.backward()

        with torch.no_grad():
            for param in model.parameters():
                grad = param.grad.clone()
                expect_grad = expect_param_grads[param]
                total_grad = (lr * (grad - expect_grad) / batch_size)
                param -= total_grad
