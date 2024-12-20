import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import tensorflow_datasets as tfds
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf


@jax.jit
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
        h1 = reparametrize(mean1, logvar1, k1)

        # Output layer
        mean2 = OutputLayer(self.hidden_features, self.output_features)(h1)

        return h1, mean1, logvar1, mean2


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

        return h1, h2, mean1, logvar1, mean2, logvar2


class IWAE(nn.Module):
    input_features: int
    hidden_features: int
    latent_features: int

    def setup(self):
        self.encoder = Encoder(self.hidden_features, self.latent_features)
        self.decoder = Decoder(self.input_features, self.hidden_features)

    def __call__(self, x, key, k):
        # Split the key into k different keys for each importance sample
        keys = jax.random.split(key, k)

        # Vectorize the encoder and decoder over the k samples
        vmapped_encoder = jax.vmap(self.encoder, in_axes=(None, 0))
        vmapped_decoder = jax.vmap(self.decoder, in_axes=(None, 0))

        # Apply the encoder and decoder
        encoder_outputs = vmapped_encoder(x, keys)  # Outputs have shape (k, batch_size, ...)
        decoder_outputs = vmapped_decoder(encoder_outputs[1], keys)  # Assuming q_h2 is at index 1

        # Extract and concatenate outputs across the k samples
        q_h1s, q_h2s, q_mean1s, q_logvar1s, q_mean2s, q_logvar2s = encoder_outputs
        p_h1s, p_mean1s, p_logvar1s, reconstructed_xs = decoder_outputs

        # Reshape the outputs to combine the k samples with the batch dimension
        def flatten(k_array):
            return k_array.reshape(-1, k_array.shape[-1])

        q_h1s = flatten(q_h1s)
        q_h2s = flatten(q_h2s)
        q_mean1s = flatten(q_mean1s)
        q_logvar1s = flatten(q_logvar1s)
        q_mean2s = flatten(q_mean2s)
        q_logvar2s = flatten(q_logvar2s)
        p_h1s = flatten(p_h1s)
        p_mean1s = flatten(p_mean1s)
        p_logvar1s = flatten(p_logvar1s)
        reconstructed_xs = flatten(reconstructed_xs)

        return q_h1s, q_h2s, q_mean1s, q_logvar1s, q_mean2s, q_logvar2s, p_h1s, p_mean1s, p_logvar1s, reconstructed_xs


# Helper functions for log probabilities
@jax.jit
def log_normal(x, mean, logvar):
    return -0.5 * (logvar + ((x - mean) ** 2) / jnp.exp(logvar) + jnp.log(2 * jnp.pi))


@jax.jit
def log_bernoulli(x, p):
    return x * jnp.log(p + 1e-8) + (1 - x) * jnp.log(1 - p + 1e-8)


@jax.jit
def compute_marginal_ll(q_h1, q_h2, q_mean1, q_logvar1, q_mean2, q_logvar2, p_h1, p_mean1, p_logvar1, reconstructed_x, x, k):
    log_p_h_2 = log_normal(q_h2, 0, 1)
    log_p_h_1_given_h_2 = log_normal(p_h1, p_mean1, p_logvar1)
    log_p_x_given_h_1 = log_bernoulli(x, reconstructed_x)

    log_q_h_1_given_x = log_normal(q_h1, q_mean1, q_logvar1)
    log_q_h_2_given_h_1 = log_normal(q_h2, q_mean2, q_logvar2)

    log_ws = log_p_h_2 + log_p_h_1_given_h_2 + log_p_x_given_h_1 - log_q_h_1_given_x - log_q_h_2_given_h_1

    log_marginal_likelihood = jax.scipy.special.logsumexp(log_ws, axis=0) - jnp.log(k)

    return log_marginal_likelihood, log_ws


@partial(jax.jit, static_argnums=(3,))
def train_step(state, batch, rng, k):
    def loss_fn(params):
        [q_h1, q_h2, q_mean1, q_logvar1, q_mean2, q_logvar2, p_h1, p_mean1, p_logvar1, reconstructed_x] = state.apply_fn({"params": params}, batch, rng, k)
        log_marginal_likelihood, _ = compute_marginal_ll(q_h1, q_h2, q_mean1, q_logvar1, q_mean2, q_logvar2, p_h1, p_mean1, p_logvar1, reconstructed_x, batch, k)
        iwae_loss = -jnp.mean(log_marginal_likelihood)
        return iwae_loss

    grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

    return state


@partial(jax.jit, static_argnums=(3,))
def eval_step(params, batch, rng, k):
    [q_h1, q_h2, q_mean1, q_logvar1, q_mean2, q_logvar2, p_h1, p_mean1, p_logvar1, reconstructed_x] = state.apply_fn({"params": params}, batch, rng, k)
    log_marginal_likelihood, _ = compute_marginal_ll(q_h1, q_h2, q_mean1, q_logvar1, q_mean2, q_logvar2, p_h1, p_mean1, p_logvar1, reconstructed_x, batch, k)
    iwae_loss = -jnp.mean(log_marginal_likelihood)
    return iwae_loss


def train(train_ds, test_ds, state, num_epochs, k):
    train_losses = []
    test_losses = []

    rng = jax.random.PRNGKey(0)

    for epoch in range(1, num_epochs + 1):
        # Training
        epoch_loss = 0
        for batch_images, _ in train_ds:
            batch = jnp.array(batch_images.numpy())
            rng, step_rng = jax.random.split(rng)
            state = train_step(state, batch, step_rng, k)
            loss = eval_step(state.params, batch, step_rng, k)
            epoch_loss += loss

        avg_train_loss = epoch_loss / len(train_ds)
        train_losses.append(avg_train_loss)

        # Evaluation
        epoch_val_loss = 0
        for batch_images, batch_labels in test_ds:
            batch = jnp.array(batch_images.numpy())
            rng, step_rng = jax.random.split(rng)
            loss = eval_step(state.params, batch, step_rng, k)
            epoch_val_loss += loss

        avg_val_loss = epoch_val_loss / len(test_ds)
        test_losses.append(avg_val_loss)

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_val_loss:.4f}")

    return state, train_losses, test_losses


def preprocess(image, label):
    # Normalize the image to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    # Flatten the image
    image = tf.reshape(image, [-1])
    return image, label


def load_dataset(split, batch_size, shuffle=True):
    ds = tfds.load("mnist", split=split, as_supervised=True)
    if shuffle:
        ds = ds.shuffle(buffer_size=1024)
    ds = ds.batch(batch_size)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# Define hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
NUM_EPOCHS = 50
K = 5  # Number of importance samples

# Initialize the model
input_features = 784  # Example for MNIST
hidden_features = 400
latent_features = 20

model = IWAE(input_features, hidden_features, latent_features)

# Initialize parameters
rng = jax.random.PRNGKey(0)
dummy_input = jnp.ones((BATCH_SIZE, input_features))
params = model.init(rng, dummy_input, rng, K)["params"]

# Define the optimizer
optimizer = optax.adam(learning_rate=LEARNING_RATE)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

train_ds = load_dataset("train", BATCH_SIZE, shuffle=True)
test_ds = load_dataset("test", BATCH_SIZE, shuffle=False)

state, train_losses, test_losses = train(train_ds, test_ds, state, NUM_EPOCHS, K)


plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Train Loss")
plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss over Epochs")
plt.legend()
plt.show()
