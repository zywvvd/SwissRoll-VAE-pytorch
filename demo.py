import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class GibbsDistModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.energy_func = torch.nn.Sequential(
            torch.nn.Linear(5, 1),
            torch.nn.ReLU()
        )

    def forward(self, x):
        feature = torch.stack([x, x**2, x**3, x**4, x**5], dim=1)
        energy = self.energy_func(feature)
        return energy

    def draw(self):
        x = torch.linspace(-5, 5, 500)
        E = self.forward(x)
        E = E.detach().numpy()
        x = x.detach().numpy()
        fx = np.exp(-E)

        plt.figure(1, figsize=(12, 8))
        plt.clf()
        plt.plot(x, fx)
        plt.xlim([-5, 5])
        plt.title('min/max E(x): {:.3f}/{:.3f}'.format(E.min(), E.max()))
        plt.pause(.5)



class Dataset:
    def __init__(self, size=10000):
        self.samples = np.random.randn(size).astype('float32')

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

lr = 1e-1

for iepoch in tqdm(range(100)):

    # visualize results
    model.draw()

    model.zero_grad()
    batch_iter = data.batch(len(data))
    batch = next(batch_iter)
    x = torch.tensor(batch)
    energy = model(x)
    energy = torch.mean(energy)
    energy.backward()

    param_grads = list()
    for param in model.parameters():
        param_grads.append(param.grad.clone())


    model.zero_grad()
    batch_iter = data.batch(len(data))
    batch = next(batch_iter)
    x = torch.tensor(batch)
    energy = model(x)
    energy = torch.mean(energy)
    energy.backward()

    with torch.no_grad():
        for param, vice_grad in zip(model.parameters(), param_grads):
            print(param.grad, vice_grad)
            param -= lr * (param.grad - vice_grad)
