import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from lib import DatasetSwissRoll as Dataset
from lib import SimpleVAE
from lib import get_device_str
from lib import show


def temp_show(ax, gt_samples, data, model, batch_size, fix=False):
    cur_ax = ax[0]
    cur_ax.clear()
    show(gt_samples, name='GT samples', ax=cur_ax)
    cur_ax = ax[1]
    cur_ax.clear()
    show(data, name='Sampled Z', ax=cur_ax)
    cur_ax = ax[2]
    cur_ax.clear()
    model.show(batch_size, ax=cur_ax)
    if fix:
        plt.show()
    else:
        plt.pause(0.002)


if __name__ == '__main__':
    data = Dataset()
    # visualize ground truth data
    data.show(fix=True)

    model = SimpleVAE(latent_dim=2)
    model.to(get_device_str())

    optim = torch.optim.Adam(model.parameters(), lr=2e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=1000, gamma=.7)

    # training
    batch_size = 512
    iter_num = 10000
    fig, ax = plt.subplots(1, 3, figsize=[15, 5])
    qbar = tqdm(total=iter_num)

    for iter in range(iter_num):

        optim.zero_grad()
        gt_samples = data.gen_data_xy(batch_size)
        tensor_gt_samples = torch.Tensor(gt_samples).to('cuda:0')
        forward_res = model(tensor_gt_samples)

        loss = model.loss_function(forward_res, 1)
        loss['loss'].backward()

        scheduler.step()
        optim.step()
        # model.show()
        if iter % 200 == 0:
            temp_show(ax, gt_samples, forward_res[4].cpu().detach().numpy(), model, batch_size)

        qbar.update(1)
        qbar.set_description(desc=f"step: {iter}, lr: {format(optim.param_groups[0]['lr'], '.2e')}, loss: {format(loss['loss'], '.3f')}, Reconstruction_Loss: {format(loss['Reconstruction_Loss'], '.3f')}, KLD: {format(loss['KLD'], '.3f')}.")

    plt.close()
    model.show(fix=True)
    pass