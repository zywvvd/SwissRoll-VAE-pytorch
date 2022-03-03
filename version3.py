import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class Abs(torch.nn.Module):
    def forward(self, x):
        return torch.abs(x)

class GibbsDistModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.energy_func = torch.nn.Sequential(
            torch.nn.Linear(4, 1),
            Abs()
        )

    def forward(self, x):
        feature = torch.stack([x, x**2, x**3, x**4], dim=1)
        energy = self.energy_func(feature)
        return energy

    def draw(self):
        with torch.no_grad():
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
    def __init__(self, size=100):
        self.samples = 3 * np.random.randn(size).astype('float32')

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

lr = 1e-2

for iepoch in tqdm(range(1000)):

    # visualize results
    model.draw()

    model.zero_grad()
    batch_iter = data.batch(len(data))
    batch = next(batch_iter)
    x = torch.tensor(batch)
    energy = model(x)
    energy = torch.mean(energy)
    energy.backward()

    vice_param_grads = list()
    for param in model.parameters():
        vice_param_grads.append(param.grad.clone())


    param_grads = list()
    for batch in data.batch(1):
        model.zero_grad()
        x = torch.tensor(batch)
        energy = model(x)
        energy.backward()

        with torch.no_grad():
            for param, vice_grad in zip(model.parameters(), vice_param_grads):
                grad = param.grad.clone() - vice_grad
                param_grads.append(grad)

    with torch.no_grad():
        for param, grad in zip(model.parameters(), param_grads):
            param -= (lr * grad / len(data))
