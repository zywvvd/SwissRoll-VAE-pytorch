import torch
from tqdm import tqdm

from lib.dataset import DatasetSwissRoll as Dataset
from lib.generator import MLPGenerator as Generator

data = Dataset()
model = Generator()
optim = torch.optim.SGD(model.parameters(), lr=1e-1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=2, gamma=.5)

batch_size = 32

for iepoch in range(500):

    # visualize results
    model.show()

    qbar = tqdm(total=len(data)//batch_size)
    for gt_samples in data.batch(batch_size):
        optim.zero_grad()

        loss = model.loss(gt_samples)
        loss.backward()

        scheduler.step()
        optim.step()

        qbar.update(1)
        qbar.set_description(desc='Loss: {:.2f}'.format(loss.detach().numpy()))
