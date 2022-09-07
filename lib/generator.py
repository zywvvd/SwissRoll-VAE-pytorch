import torch
from .base import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
from .utils import show


class SimpleVAE(BaseVAE):
    def __init__(self, in_channels: int=2, latent_dim: int=2, hidden_dims: List = None) -> None:
        super(SimpleVAE, self).__init__()

        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [128, 128]

        ori_in_channels = in_channels

        # Build Encoder
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []
        de_hidden_dims = [hidden_dims[-1]] + hidden_dims

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])
        hidden_dims.reverse()

        for i in range(len(de_hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(de_hidden_dims[i], de_hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                            nn.Linear(de_hidden_dims[-1], ori_in_channels))

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x in_channels]
        :return: (Tensor) List of latent codes [N x latent_dim]
        """
        result = self.encoder(input)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes onto the data space.
        
        :param z: (Tensor) [N x latent_dim]
        :return: (Tensor) [N x in_channels]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [N x latent_dim]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [N x latent_dim]
        :return: (Tensor) [N x latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var, z]

    def loss_function(self, forward_res, kld_weight) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        """
        recons = forward_res[0]
        input = forward_res[1]
        mu = forward_res[2]
        log_var = forward_res[3]

        recons_loss =F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        data space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """

        z = torch.randn(num_samples,
                        self.latent_dim)
        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def show(self, sample_num = 1024, fix=False, ax=None):
        """
        Show model sample scatter
        Args:
            sample_num (int, optional): sample num. Defaults to 1024.
        """
        current_device = self.fc_mu.weight.device
        z = self.sample(sample_num, current_device)
        gen_samples = z.cpu().detach().numpy()
        show(gen_samples, 'Generated', fix=fix, ax=ax)
        pass
