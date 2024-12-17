import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

import torch.distributions as dists
import torch
from torchvision import datasets, transforms


def reparametrize(mean, logvar, key):
    # Use the reparameterization trick to sample
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(key, mean.shape)
    return mean + eps * std  # shape same as mean


class GaussianLayer(nn.Module):
    output_features: int

    @nn.compact
    def __call__(self, x):
        hidden1 = nn.tanh(nn.Dense(self.output_features)(x))
        hidden2 = nn.tanh(nn.Dense(self.output_features)(hidden1))
        mean = nn.Dense(self.output_features)(hidden2)
        logvar = nn.Dense(self.output_features)(hidden2)
        return mean, logvar


class OutputLayer(nn.Module):
    hidden_features: int
    output_features: int

    @nn.compact
    def __call__(self, x):
        hidden1 = nn.tanh(nn.Dense(self.hidden_features)(x))
        hidden2 = nn.tanh(nn.Dense(self.hidden_features)(hidden1))
        mean = nn.sigmoid(nn.Dense(self.output_features)(hidden2))
        return mean


class Decoder(nn.Module):
    output_features: int
    hidden_features: int

    @nn.compact
    def __call__(self, h2, key):
        k1, k2 = jax.random.split(key)
        # Gaussian layer
        mean1, logvar1 = GaussianLayer(self.hidden_features)(h2)
        h1 = reparametrize(mean1, logvar1, k1, 1)

        # Output layer
        mean2 = OutputLayer(self.hidden_features, self.output_features)(h1)

        return h1, (mean1, logvar1), mean2


class Encoder(nn.Module):
    hidden_features: int
    latent_features: int

    @nn.compact
    def __call__(self, x, key):
        k1, k2 = jax.random.split(key)
        # First Gaussian layer
        mean1, logvar1 = GaussianLayer(self.hidden_features)(x)
        # here we will sample k times from the distribution
        h1 = reparametrize(mean1, logvar1, k1)

        # Second Gaussian layer
        mean2, logvar2 = GaussianLayer(self.latent_features)(h1)
        # for all the k h1, we will sample one h2 each
        h2 = reparametrize(mean2, logvar2, k2)

        return h1, h2, (mean1, logvar1), (mean2, logvar2)


class IWAE(nn.Module):
    input_features: int
    hidden_features: int
    latent_features: int

    def setup(self):
        self.encoder = Encoder(self.hidden_features, self.latent_features)
        self.decoder = Decoder(self.input_features, self.hidden_features)

    def __call__(self, x, key, k):
        for i in range(k):
            q_h1, q_h2, (q_mean1, q_logvar1), (q_mean2, q_logvar2) = self.encoder(x, key)
            p_h1, (p_mean1, p_logvar1), reconstructed_x = self.decoder(q_h2, key)

            if i == 0:
                q_h1s = q_h1
                q_h2s = q_h2
                q_mean1s = q_mean1
                q_logvar1s = q_logvar1
                q_mean2s = q_mean2
                q_logvar2s = q_logvar2
                p_h1s = p_h1
                p_mean1s = p_mean1
                p_logvar1s = p_logvar1
                reconstructed_xs = reconstructed_x
            else:
                q_h1s = jnp.concatenate((q_h1s, q_h1), axis=0)
                q_h2s = jnp.concatenate((q_h2s, q_h2), axis=0)
                q_mean1s = jnp.concatenate((q_mean1s, q_mean1), axis=0)
                q_logvar1s = jnp.concatenate((q_logvar1s, q_logvar1), axis=0)
                q_mean2s = jnp.concatenate((q_mean2s, q_mean2), axis=0)
                q_logvar2s = jnp.concatenate((q_logvar2s, q_logvar2), axis=0)
                p_h1s = jnp.concatenate((p_h1s, p_h1), axis=0)
                p_mean1s = jnp.concatenate((p_mean1s, p_mean1), axis=0)
                p_logvar1s = jnp.concatenate((p_logvar1s, p_logvar1), axis=0)
                reconstructed_xs = jnp.concatenate((reconstructed_xs, reconstructed_x), axis=0)

        return q_h1s, q_h2s, q_mean1s, q_logvar1s, q_mean2s, q_logvar2s, p_h1s, p_mean1s, p_logvar1s, reconstructed_xs

    def compute_marginal_ll(self, x, key, k):
        [q_h1, q_h2, q_mean1, q_logvar1, q_mean2, q_logvar2, p_h1, p_mean1, p_logvar1, reconstructed_x] = self.__call__(x, key, k)

        log_p_h_2 = log_normal(q_h2, 0, 1)
        log_p_h_1_given_h_2 = log_normal(p_h1, p_mean1, p_logvar1)
        log_p_x_given_h_1 = log_bernoulli(x, reconstructed_x)

        log_q_h_1_given_x = log_normal(q_h1, q_mean1, q_logvar1)
        log_q_h_2_given_h_1 = log_normal(q_h2, q_mean2, q_logvar2)

        log_ws = log_p_h_2 + log_p_h_1_given_h_2 + log_p_x_given_h_1 - log_q_h_1_given_x - log_q_h_2_given_h_1

        log_marginal_likelihood = jax.scipy.special.logsumexp(log_ws, axis=-1) - jnp.log(k)

        return log_marginal_likelihood, log_ws

    def compute_loss(self, x, key, k):
        log_marginal_likelihood, log_ws = self.compute_marginal_ll(x, key, k)
        iwae_loss = -jnp.mean(log_marginal_likelihood)
        return iwae_loss


# Helper functions for log probabilities
def log_normal(x, mean, logvar):
    return -0.5 * (logvar + ((x - mean) ** 2) / jnp.exp(logvar) + jnp.log(2 * jnp.pi))


def log_bernoulli(x, p):
    return x * jnp.log(p + 1e-8) + (1 - x) * jnp.log(1 - p + 1e-8)
