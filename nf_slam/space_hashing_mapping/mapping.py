import dataclasses
import functools
import time
from typing import List

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import tqdm
from flax.optim import OptimizerState
from flax.optim.adam import Adam

from nf_slam.laser_data import LaserData
from nf_slam.space_hashing_mapping.jax_math import calculate_densities
from nf_slam.space_hashing_mapping.map_model import MapModel, MapModelConfig, init_map_model
from nf_slam.space_hashing_mapping.mlp_model import MLPModel


@jdc.pytree_dataclass
class ScanData(object):
    depths: jnp.array
    angles: jnp.array

    @classmethod
    def from_laser_data(cls, laser_data):
        mask = laser_data.ranges < laser_data.parameters.maximal_distance
        # noinspection PyArgumentList
        return cls(angles=jnp.array(laser_data.angles[mask]), depths=jnp.array(laser_data.ranges[mask]))

    def get_random_subset(self, point_count):
        indices = np.arange(self.depths.shape[0])
        indices = jnp.array(np.random.choice(indices, point_count))
        # noinspection PyArgumentList
        return self.__class__(angles=self.angles[indices], depths=self.depths[indices])


@jdc.pytree_dataclass
class LearningData(object):
    uniform: jnp.array


@jdc.pytree_dataclass
class Position(object):
    x: jnp.array
    y: jnp.array
    angle: jnp.array


@jax.jit
def calculate_weights(densities, depth_deltas):
    mis_probability = jnp.exp(-densities * depth_deltas)
    hit_probability = 1 - mis_probability
    mis_probability = jnp.concatenate([jnp.ones(1), mis_probability])
    hit_probability = jnp.concatenate([hit_probability, jnp.ones(1)])
    cumulative_product = jnp.cumprod(mis_probability)
    weights = cumulative_product * hit_probability
    return weights


@functools.partial(jax.jit, static_argnums=1)
def sample_depth_bins(learning_data, parameters):
    depths = jnp.linspace(parameters.minimal_depth, parameters.maximal_depth, parameters.bins_count + 1)[:-1]
    depths = depths + parameters.depth_delta * learning_data.uniform
    return jnp.concatenate([
        jnp.full([depths.shape[0], 1], parameters.minimal_depth),
        depths,
        jnp.full([depths.shape[0], 1], parameters.maximal_depth)], axis=-1)


@functools.partial(jax.jit, static_argnums=1)
def sample_depth_bins_exp(learning_data, parameters):
    weights = jnp.exp(jnp.linspace(0, 1, parameters.bins_count) * 0.017 * parameters.bins_count)
    deltas = weights / jnp.sum(weights) * (parameters.maximal_depth - parameters.minimal_depth)
    depths = jnp.cumsum(deltas) + parameters.minimal_depth
    sampled_depths = depths[None] - learning_data.uniform * deltas[None]
    return jnp.concatenate([
        jnp.full([sampled_depths.shape[0], 1], parameters.minimal_depth),
        sampled_depths,
        jnp.full([sampled_depths.shape[0], 1], parameters.maximal_depth)], axis=-1)


@jax.jit
def transform_points(points, position):
    x = position[..., 0, None]
    y = position[..., 1, None]
    angle = position[..., 2, None]
    transformed_x = x + points[..., 0] * jnp.cos(angle) - points[..., 1] * jnp.sin(angle)
    transformed_y = y + points[..., 0] * jnp.sin(angle) + points[..., 1] * jnp.cos(angle)
    return jnp.stack([transformed_x, transformed_y], axis=-1)


@jax.jit
def calculate_points(depths, scan_data: ScanData):
    x = depths * jnp.cos(scan_data.angles[..., None])
    y = depths * jnp.sin(scan_data.angles[..., None])
    return jnp.stack([x, y], axis=-1)


@functools.partial(jax.jit, static_argnums=[4, 5])
def depth_prediction_loss_function(map_model: MapModel, position: jnp.array, scan_data: ScanData,
                                   learning_data: LearningData, config: MapModelConfig, model: MLPModel):
    depth_bins = sample_depth_bins(learning_data, config)
    depths = (depth_bins[..., 1:] + depth_bins[..., :-1]) / 2
    depth_deltas = (depth_bins[..., 1:] - depth_bins[..., :-1]) / 2
    points = calculate_points(depths, scan_data)
    points = transform_points(points, position).reshape(-1, 2)
    densities = config.density_scale * calculate_densities(points, map_model, model, config.L).reshape(
        depths.shape[:2])
    weights = jax.vmap(calculate_weights)(densities, depth_deltas)
    extended_depths = jnp.concatenate([depths, jnp.full([depths.shape[0], 1], config.maximal_depth)], axis=-1)
    predicted_depths = jnp.sum(weights * extended_depths, axis=-1)
    predicted_variance = jnp.sum(weights * (extended_depths - predicted_depths[..., None]) ** 2, axis=-1)
    return jnp.mean((scan_data.depths - predicted_depths) ** 2 / jax.lax.stop_gradient(predicted_variance + 1e-1) +
                    config.variance_weight * predicted_variance)


@functools.partial(jax.jit, static_argnums=[4, 5])
def predict_depths(map_model: MapModel, position: jnp.array, scan_data: ScanData,
                   learning_data: LearningData, config: MapModelConfig, model: MLPModel):
    depth_bins = sample_depth_bins(learning_data, config)
    depths = (depth_bins[..., 1:] + depth_bins[..., :-1]) / 2
    depth_deltas = (depth_bins[..., 1:] - depth_bins[..., :-1]) / 2
    points = calculate_points(depths, scan_data)
    points = transform_points(points, position).reshape(-1, 2)
    densities = config.density_scale * calculate_densities(points, map_model, model, config.L).reshape(
        depths.shape[:2])
    weights = jax.vmap(calculate_weights)(densities, depth_deltas)
    extended_depths = jnp.concatenate([depths, jnp.full([depths.shape[0], 1], config.maximal_depth)], axis=-1)
    return jnp.sum(weights * extended_depths, axis=-1)


@dataclasses.dataclass
class OptimizerConfig:
    learning_rate: float
    beta1: float
    beta2: float
    eps: float
    weight_decay: float


@dataclasses.dataclass
class LearningConfig:
    iterations: int  # 100
    variable_optimizer_config: OptimizerConfig
    hashtable_optimizer_config: OptimizerConfig


@dataclasses.dataclass
class BuildMapResult:
    loss_history: List[float]
    map_model: MapModel


@dataclasses.dataclass
class BuildMapState:
    iteration: int
    variable_state: OptimizerState
    hashtable_state: OptimizerState


class MapBuilder:
    def __init__(self, learning_config: LearningConfig, map_model_config: MapModelConfig, mlp_model: MLPModel):
        self.state = None
        self.grad_function = None
        self._variable_optimizer = Adam(**dataclasses.asdict(learning_config.variable_optimizer_config))
        self._hashtable_optimizer = Adam(**dataclasses.asdict(learning_config.hashtable_optimizer_config))
        self._learning_config = learning_config
        self._map_model_config = map_model_config
        self._mlp_model = mlp_model

    def setup(self, scan_data: ScanData, map_model: MapModel, position: jnp.array):
        start_time = time.time()
        # noinspection PyArgumentList
        learning_data = LearningData(uniform=jax.random.uniform(
            jax.random.PRNGKey(0),
            (len(scan_data.depths), self._map_model_config.bins_count)))
        grad_function = jax.jit(jax.value_and_grad(depth_prediction_loss_function), static_argnums=[4, 5])

        # noinspection PyUnresolvedReferences
        self.grad_function = grad_function.lower(map_model, position, scan_data, learning_data, self._map_model_config,
                                                 self._mlp_model).compile()
        time_delta = time.time() - start_time
        print(f"Compilation take {time_delta} s")

        self.state = BuildMapState(
            iteration=0,
            variable_state=self._variable_optimizer.init_state(map_model.variables),
            hashtable_state=self._hashtable_optimizer.init_state(map_model.hashtable)
        )

    def step(self, result: BuildMapResult, position: jnp.array, scan_data: ScanData):
        map_model = result.map_model
        # noinspection PyArgumentList
        learning_data = LearningData(uniform=jax.random.uniform(
            jax.random.PRNGKey(self.state.iteration),
            (len(scan_data.depths), self._map_model_config.bins_count)))
        loss, grad = self.grad_function(map_model, position, scan_data, learning_data)
        loss_history = result.loss_history + [loss]
        variables, variable_state = self._variable_optimizer.apply_gradient(
            self._variable_optimizer.hyper_params,
            map_model.variables,
            self.state.variable_state, grad.variables)
        hashtable, hashtable_state = self._hashtable_optimizer.apply_gradient(
            self._hashtable_optimizer.hyper_params,
            map_model.hashtable,
            self.state.hashtable_state, grad.hashtable)
        map_model = MapModel(hashtable=hashtable, variables=variables, resolutions=map_model.resolutions,
                             origins=map_model.origins, rotations=map_model.rotations)
        self.state = BuildMapState(self.state.iteration + 1, variable_state, hashtable_state)
        return BuildMapResult(loss_history, map_model)

    # noinspection PyUnresolvedReferences
    def build_map(self, laser_data: LaserData, position: jnp.array):
        scan_data = ScanData.from_laser_data(laser_data)
        map_model = init_map_model(self._mlp_model, self._map_model_config)
        result = BuildMapResult([], map_model)
        self.setup(scan_data, map_model, position)
        for i in tqdm.tqdm(range(self._learning_config.iterations)):
            result = self.step(result, position, scan_data)
        return result
