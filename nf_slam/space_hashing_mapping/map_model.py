from dataclasses import dataclass, field

import flax.core
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np


@jdc.pytree_dataclass
class MapModel:
    hashtable: jnp.array
    variables: flax.core.frozen_dict.FrozenDict
    resolutions: jnp.array
    origins: jnp.array
    rotations: jnp.array


@dataclass(unsafe_hash=True)
class MapModelConfig(object):
    minimal_depth: float
    maximal_depth: float
    bins_count: int
    density_scale: float
    variance_weight: float
    F: int
    L: int
    T: int
    max_log_resolution: float
    min_log_resolution: float
    depth_delta: float = field(init=False)
    huber_delta: float

    def __post_init__(self):
        self.depth_delta = (self.maximal_depth - self.minimal_depth) / self.bins_count


# noinspection PyArgumentList
def init_map_model(model, config):
    hashtable: jnp.array = jax.random.normal(jax.random.PRNGKey(1), (config.L, config.T, config.F))
    resolutions = 2 ** jnp.linspace(config.min_log_resolution, config.max_log_resolution, config.L)
    origins = jnp.zeros(2) + jax.random.normal(jax.random.PRNGKey(2), (config.L, 2))
    rotations = jax.random.uniform(jax.random.PRNGKey(10), (config.L,)) * 2 * np.pi
    batch = jnp.ones([10, config.F * config.L])
    variables = model.init(jax.random.PRNGKey(1), batch)
    return MapModel(hashtable=hashtable, variables=variables, resolutions=resolutions,
                    origins=origins, rotations=rotations)
