import dataclasses
import functools
from typing import List

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import tqdm
from flax.optim.adam import Adam

from nf_slam.laser_data import LaserData
from nf_slam.space_hashing_mapping.jax_math import calculate_densities
from nf_slam.space_hashing_mapping.map_model import MapModel, ModelConfig, init_map_model
from nf_slam.space_hashing_mapping.mlp_model import MLPModel


@jdc.pytree_dataclass
class ScanData(object):
    depths: jnp.array
    angles: jnp.array

    # noinspection PyArgumentList
    @classmethod
    def from_laser_data(cls, laser_data):
        mask = laser_data.ranges < laser_data.parameters.maximal_distance
        return cls(angles=jnp.array(laser_data.angles[mask]), depths=jnp.array(laser_data.ranges[mask]))


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
                                   learning_data: LearningData, config: ModelConfig, model: MLPModel):
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
                   learning_data: LearningData, config: ModelConfig, model: MLPModel):
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
class ConstructMapResult:
    loss_history: List[float]
    map_model: MapModel


# noinspection PyArgumentList
def construct_map_from_one_scan(config: ModelConfig, learning_config: LearningConfig, laser_data: LaserData,
                                model: MLPModel, map_position):
    loss_function = depth_prediction_loss_function
    grad_function = jax.jit(jax.grad(loss_function), static_argnums=[4, 5])
    scan_data = ScanData.from_laser_data(laser_data)
    map_model = init_map_model(model, config)
    variable_optimizer = Adam(**dataclasses.asdict(learning_config.variable_optimizer_config))
    variable_state = variable_optimizer.init_state(map_model.variables)
    hashtable_optimizer = Adam(**dataclasses.asdict(learning_config.hashtable_optimizer_config))
    hashtable_state = hashtable_optimizer.init_state(map_model.hashtable)

    loss_history = []
    for i in tqdm.tqdm(range(learning_config.iterations)):
        learning_data = LearningData(uniform=jax.random.uniform(
            jax.random.PRNGKey(i),
            (len(scan_data.depths), config.bins_count)))
        grad = grad_function(map_model, map_position, scan_data, learning_data, config, model)
        loss = loss_function(map_model, map_position, scan_data, learning_data, config, model)
        loss_history.append(loss)
        variables, variable_state = hashtable_optimizer.apply_gradient(
            hashtable_optimizer.hyper_params,
            map_model.variables,
            variable_state, grad.variables)
        hashtable, hashtable_state = hashtable_optimizer.apply_gradient(
            hashtable_optimizer.hyper_params,
            map_model.hashtable,
            hashtable_state, grad.hashtable)
        map_model = MapModel(hashtable=hashtable, variables=variables, resolutions=map_model.resolutions,
                             origins=map_model.origins, rotations=map_model.rotations)
    return ConstructMapResult(loss_history, map_model)
