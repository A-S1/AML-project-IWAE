import torch.nn as nn
import numpy as np
import torch.distributions as dists
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from livelossplot import PlotLosses


class Binarized_MNIST(datasets.MNIST):
    def __init__(self, root, train, transform=None, target_transform=None, download=False):
        super(Binarized_MNIST, self).__init__(root, train, transform, target_transform, download)

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        return dists.Bernoulli(img).sample().type(torch.float32)


MNIST_SIZE = 28
HIDDEN_DIM = 400
LATENT_DIM = 50


class VAE(nn.Module):
    def __init__(self, k):
        super(VAE, self).__init__()
        self.k = k
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(MNIST_SIZE**2, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, 2 * LATENT_DIM))
        self.decoder = nn.Sequential(nn.Linear(LATENT_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, MNIST_SIZE**2), nn.Sigmoid())
        return

    def compute_loss(self, x, k=None, mode = "standard"):
        if mode == "standard":
            if not k:
                k = self.k
            [x_tilde, z, mu_z, log_var_z] = self.forward(x, k)
            # upsample x
            x_s = x.unsqueeze(1).repeat(1, k, 1, 1, 1)
            # compute negative log-likelihood
            NLL = -dists.Bernoulli(x_tilde).log_prob(x_s).sum(axis=(2, 3, 4)).mean()
            # copmute kl divergence
            KL_Div = -0.5 * (1 + log_var_z - mu_z.pow(2) - log_var_z.exp()).sum(1).mean()
            # compute loss
            loss = NLL + KL_Div
            return loss
        elif mode == "iwae":
            """computes the IWAE loss, which is the negative ELBO with importance
            weights computed by the IWAE method, used for second stage testing
            """
            return self.compute_marginal_log_likelihood(x, k)[0]
    
    def forward(self, x, k=None):
        """feed image (x) through VAE

        Args:
            x (torch tensor): input [batch, img_channels, img_dim, img_dim]

        Returns:
            x_tilde (torch tensor): [batch, k, img_channels, img_dim, img_dim]
            z (torch tensor): latent space samples [batch, k, LATENT_DIM]
            mu_z (torch tensor): mean latent space [batch, LATENT_DIM]
            log_var_z (torch tensor): log var latent space [batch, LATENT_DIM]
        """
        if not k:
            k = self.k
        z, mu_z, log_var_z = self.encode(x, k)
        x_tilde = self.decode(z, k)
        return [x_tilde, z, mu_z, log_var_z]

    def encode(self, x, k):
        """computes the approximated posterior distribution parameters and
        samples from this distribution

        Args:
            x (torch tensor): input [batch, img_channels, img_dim, img_dim]

        Returns:
            z (torch tensor): latent space samples [batch, k, LATENT_DIM]
            mu_E (torch tensor): mean latent space [batch, LATENT_DIM]
            log_var_E (torch tensor): log var latent space [batch, LATENT_DIM]
        """
        # get encoder distribution parameters
        out_encoder = self.encoder(x)
        mu_E, log_var_E = torch.chunk(out_encoder, 2, dim=1)
        # increase shape for sampling [batch, samples, latent_dim]
        mu_E_ups = mu_E.unsqueeze(1).repeat(1, k, 1)
        log_var_E_ups = log_var_E.unsqueeze(1).repeat(1, k, 1)
        # sample noise variable for each batch and sample
        epsilon = torch.randn_like(log_var_E_ups)
        # get latent variable by reparametrization trick
        z = mu_E_ups + torch.exp(0.5 * log_var_E_ups) * epsilon
        return z, mu_E, log_var_E

    def decode(self, z, k):
        """computes the Bernoulli mean of p(x|z)
        note that linear automatically parallelizes computation

        Args:
            z (torch tensor): latent space samples [batch, k, LATENT_DIM]

        Returns:
            x_tilde (torch tensor): [batch, k, img_channels, img_dim, img_dim]
        """
        # get decoder distribution parameters
        x_tilde = self.decoder(z)  # [batch*samples, MNIST_SIZE**2]
        # reshape into [batch, samples, 1, MNIST_SIZE, MNIST_SIZE] (input shape)
        x_tilde = x_tilde.view(-1, k, 1, MNIST_SIZE, MNIST_SIZE)
        return x_tilde

    def create_latent_traversal(self, image_batch, n_pert, pert_min_max=2, n_latents=5):
        device = image_batch.device
        # initialize images of latent traversal
        images = torch.zeros(n_latents, n_pert, *image_batch.shape[1::])
        # select the latent_dims with lowest variance (most informative)
        [x_tilde, z, mu_z, log_var_z] = self.forward(image_batch)
        i_lats = log_var_z.mean(axis=0).sort()[1][:n_latents]
        # sweep for latent traversal
        sweep = np.linspace(-pert_min_max, pert_min_max, n_pert)
        # take first image and encode
        [z, mu_E, log_var_E] = self.encode(image_batch[0:1], k=1)
        for latent_dim, i_lat in enumerate(i_lats):
            for pertubation_dim, z_replaced in enumerate(sweep):
                z_new = z.detach().clone()
                z_new[0][0][i_lat] = z_replaced

                img_rec = self.decode(z_new.to(device), k=1).squeeze(0)
                img_rec = img_rec[0].clamp(0, 1).cpu()

                images[latent_dim][pertubation_dim] = img_rec
        return images

    def compute_marginal_log_likelihood(self, x, k=None):
        """computes the marginal log-likelihood in which the sampling
        distribution is exchanged to q_{\phi} (z|x),
        this function can also be used for the IWAE loss computation

        Args:
            x (torch tensor): images [batch, img_channels, img_dim, img_dim]

        Returns:
            log_marginal_likelihood (torch tensor): scalar
            log_w (torch tensor): unnormalized log importance weights [batch, k]
        """
        if not k:
            k = self.k
        [x_tilde, z, mu_z, log_var_z] = self.forward(x, k)
        # upsample mu_z, std_z, x_s
        mu_z_s = mu_z.unsqueeze(1).repeat(1, k, 1)
        std_z_s = (0.5 * log_var_z).exp().unsqueeze(1).repeat(1, k, 1)
        x_s = x.unsqueeze(1).repeat(1, k, 1, 1, 1)
        # compute logarithmic unnormalized importance weights [batch, k]
        log_p_x_g_z = dists.Bernoulli(x_tilde).log_prob(x_s).sum(axis=(2, 3, 4))
        log_prior_z = dists.Normal(0, 1).log_prob(z).sum(2)
        log_q_z_g_x = dists.Normal(mu_z_s, std_z_s).log_prob(z).sum(2)
        log_w = log_p_x_g_z + log_prior_z - log_q_z_g_x
        # compute marginal log-likelihood
        log_marginal_likelihood = (torch.logsumexp(log_w, 1) - np.log(k)).mean()
        return log_marginal_likelihood, log_w


class IWAE(VAE):
    def __init__(self, k):
        super(IWAE, self).__init__(k)
        return

    def compute_loss(self, x, k=None, mode="fast"):
        if not k:
            k = self.k
        # compute unnormalized importance weights in log_units
        log_likelihood, log_w = self.compute_marginal_log_likelihood(x, k)
        # loss computation (several ways possible)
        if mode == "original":
            ####################### ORIGINAL IMPLEMENTAION #######################
            # numerical stability (found in original implementation)
            log_w_minus_max = log_w - log_w.max(1, keepdim=True)[0]
            # compute normalized importance weights (no gradient)
            w = log_w_minus_max.exp()
            w_tilde = (w / w.sum(axis=1, keepdim=True)).detach()
            # compute loss (negative IWAE objective)
            loss = -(w_tilde * log_w).sum(1).mean()
        elif mode == "normalized weights":
            ######################## LOG-NORMALIZED TRICK ########################
            # copmute normalized importance weights (no gradient)
            log_w_tilde = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
            w_tilde = log_w_tilde.exp().detach()
            # compute loss (negative IWAE objective)
            loss = -(w_tilde * log_w).sum(1).mean()
        elif mode == "fast":
            ########################## SIMPLE AND FAST ###########################
            loss = -log_likelihood
        return loss


BATCH_SIZE = 1000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6


def train(dataset, vae_model, iwae_model, num_epochs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))

    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    vae_model.to(device)
    iwae_model.to(device)

    optimizer_vae = torch.optim.Adam(vae_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer_iwae = torch.optim.Adam(iwae_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    losses_plot = PlotLosses(groups={"Loss": ["VAE (ELBO)", "IWAE (NLL)"]})
    for epoch in range(1, num_epochs + 1):
        avg_NLL_VAE, avg_NLL_IWAE = 0, 0
        for x in data_loader:
            x = x.to(device)
            # IWAE update
            optimizer_iwae.zero_grad()
            loss = iwae_model.compute_loss(x)
            loss.backward()
            optimizer_iwae.step()
            avg_NLL_IWAE += loss.item() / len(data_loader)

            # VAE update
            optimizer_vae.zero_grad()
            loss = vae_model.compute_loss(x)
            loss.backward()
            optimizer_vae.step()

            avg_NLL_VAE += loss.item() / len(data_loader)
        # plot current losses
        losses_plot.update({"VAE (ELBO)": avg_NLL_VAE, "IWAE (NLL)": avg_NLL_IWAE}, current_step=epoch)
        losses_plot.send()
    trained_vae, trained_iwae = vae_model, iwae_model
    return trained_vae, trained_iwae


if __name__ == "__main__":
    train_ds = Binarized_MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    num_epochs = 50
    list_of_ks = [1, 10]
    for k in list_of_ks:
        vae_model = VAE(k)
        iwae_model = IWAE(k)
        trained_vae, trained_iwae = train(train_ds, vae_model, iwae_model, num_epochs)
        torch.save(trained_vae, f"./results/trained_vae_{k}.pth")
        torch.save(trained_iwae, f"./results/trained_iwae_{k}.pth")
