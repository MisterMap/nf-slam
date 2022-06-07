import functools

import jax
import jax.numpy as jnp
import numpy as np

from nf_slam.space_hashing_mapping.map_model import MapModel
from nf_slam.space_hashing_mapping.mlp_model import MLPModel


@jax.jit
def hashfunction(x, t):
    pi1 = 1
    pi2 = 19349663
    # pi3 = 83492791
    result = jnp.bitwise_xor(x[:, 0] * pi1, x[:, 1] * pi2)
    return result % t


@jax.jit
def step_function(x):
    return x * x * (3 - 2 * x) * np.sqrt(4 / 3)


@jax.jit
def bilinear_interpolation(deltas, values0, values1, values2, values3):
    f1 = values0 * step_function(1 - deltas[:, 0])[:, None] + values1 * step_function(deltas)[:, 0][:, None]
    f2 = values2 * step_function(1 - deltas[:, 0])[:, None] + values3 * step_function(deltas)[:, 0][:, None]
    return f1 * step_function(1 - deltas[:, 1])[:, None] + f2 * step_function(deltas)[:, 1][:, None]


@functools.partial(jax.jit, static_argnums=[4])
def calculate_layer_embeddings(hashtable, points, resolution, origin, t, rotation):
    points = ((points - origin) / resolution)
    x = jnp.cos(rotation) * points[:, 0] - jnp.sin(rotation) * points[:, 1]
    y = jnp.sin(rotation) * points[:, 0] + jnp.cos(rotation) * points[:, 1]
    points = jnp.stack([x, y], axis=1)
    cells = jnp.array(points // 1, jnp.int32)
    deltas = points % 1
    values0 = hashtable[hashfunction(cells, t)]
    values1 = hashtable[hashfunction(cells + jnp.array([1, 0]), t)]
    values2 = hashtable[hashfunction(cells + jnp.array([0, 1]), t)]
    values3 = hashtable[hashfunction(cells + jnp.array([1, 1]), t)]
    return bilinear_interpolation(deltas, values0, values1, values2, values3)


def calculate_densities(points: jnp.array, map_model: MapModel, model: MLPModel, layer_count):
    embeddings = []
    for i in range(layer_count):
        embedding = calculate_layer_embeddings(map_model.hashtable[i],
                                               points,
                                               map_model.resolutions[i],
                                               map_model.origins[i],
                                               map_model.hashtable.shape[1],
                                               map_model.rotations[i])
        embeddings.append(embedding)
    embedding = jnp.concatenate(embeddings, axis=1)
    return jax.nn.softplus(model.apply(map_model.variables, embedding))
