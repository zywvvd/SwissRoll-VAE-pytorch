import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class GibbsDistModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.energy_func = torch.nn.Sequential(
            torch.nn.Linear(3, 1),
            torch.nn.ReLU()
        )

    def forward(self, x):
        feature = torch.stack([x, x**2, x**3], dim=1)
        energy = self.energy_func(feature)
        return energy

    def draw(self):
        x = torch.linspace(-3, 3, 300)
        E = self.forward(x)
        E = E.detach().numpy()
        x = x.detach().numpy()
        fx = np.exp(-E)

        plt.figure(1)
        plt.clf()
        plt.plot(x, fx)
        plt.title('min/max f(x): {:.3f}/{:.3f}'.format(fx.min(), fx.max()))
        plt.pause(.5)



class Dataset:
    def __init__(self, size=5000):
        self.samples = np.random.randn(size).astype('float32')

    def batch(self, size=320):
        for idx in range(0, len(self.samples), size):
            yield self.samples[idx:idx+size]


#
# Start training
#
data_main = Dataset()
data_vice = Dataset()
model = GibbsDistModel()
optim = torch.optim.SGD(model.parameters(), lr=1e-3)

for iepoch in tqdm(range(100)):

    # visualize results
    model.draw()

    optim.zero_grad()
    for batch_main in data_main.batch(1):
        x_main = torch.tensor(batch_main)
        energy_main = model(x_main)
        grad_energy_main = energy_main.backward()

        batch_vice_iter = data_vice.batch(5000)
        batch_vice = next(batch_vice_iter)
        x_vice = torch.tensor(batch_vice)
        energy_vice = model(x_vice)
        energy_vice = -torch.mean(energy_vice)
        grad_energy_vice = energy_vice.backward()

    optim.step()
