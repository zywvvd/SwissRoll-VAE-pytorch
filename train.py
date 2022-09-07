import torch
from tqdm import tqdm

from lib.vvd_data import DatasetSwissRoll as Dataset
# from lib.dataset import DatasetUniform as Dataset
# from lib.dataset import DatasetGaussian as Dataset
# from lib.generator import MLPGenerator as Generator
from lib.vvd_gen import VVD_VAE

data = Dataset()
# data.show()

model = VVD_VAE()
model.to('cuda:0')

optim = torch.optim.Adam(model.parameters(), lr=2e-2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=500, gamma=.5)

batch_size = 2048

for iepoch in range(800):
    # visualize prediction and ground truth
    # data.show()
    # model.show()
    
    # training
    optim.zero_grad()
    gt_samples = data.gen_data_xy(batch_size)
    tensor_gt_samples = torch.Tensor(gt_samples).to('cuda:0')
    forward_res = model(tensor_gt_samples)

    loss = model.loss_function(forward_res, 1)
    print(f"step {iepoch}: loss: {loss['loss']}, Reconstruction_Loss: {loss['Reconstruction_Loss']}, KLD:{loss['KLD']}. ")
    loss['loss'].backward()

    scheduler.step()
    optim.step()
    model.show()

data.show()

