import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np


def kernel_init(*args):
    return np.sqrt(2) * jax.nn.initializers.lecun_normal()(*args)


class MLPModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        input_x = x
        x = nn.Dense(256, use_bias=True, kernel_init=kernel_init)(x)
        x = nn.relu(x)
        x = nn.Dense(256, use_bias=True, kernel_init=kernel_init)(x)
        x = nn.relu(x)
        x = nn.Dense(256, use_bias=True, kernel_init=kernel_init)(x)
        x = nn.relu(x)
        x = jnp.concatenate([x, input_x[:, -20:]], axis=1)
        x = nn.Dense(1, kernel_init=kernel_init)(x)
        return x[:, 0]
