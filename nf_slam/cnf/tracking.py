import dataclasses
import functools
from typing import Optional

import jax
import jax.numpy as jnp

from nf_slam.cnf.mapping import calculate_densities
from nf_slam.space_hashing_mapping.map_model import MapModel
from nf_slam.space_hashing_mapping.mapping import ScanData
from nf_slam.cnf.mapping import transform_points, calculate_points
from nf_slam.space_hashing_mapping.mlp_model import MLPModel
from nf_slam.tracking.batch_tracking import ScanDataBatch


@dataclasses.dataclass(unsafe_hash=True)
class TrackingConfig:
    huber_scale: float


@dataclasses.dataclass
class OptimizePositionConfig:
    learning_rate: float
    iterations: int  # 100
    init_hessian: jnp.array  # jnp.diag(jnp.array([2000, 2000, 200]))
    maximal_clip_norm: float  # 30
    beta1: float  # 0.7
    beta2: float  # 0.4
    hessian_adder: jnp.array  # jnp.diag(jnp.array([20, 20, 2]))
    tracking_config: TrackingConfig


@functools.partial(jax.jit, static_argnums=[1, ])
def huber(input_array, delta):
    mask = jnp.abs(input_array) < delta
    result = jnp.where(mask, input_array, jnp.sign(input_array) * (delta * (2 * jnp.abs(input_array) - delta)) ** 0.5)
    return result


@functools.partial(jax.jit, static_argnums=[3, 4])
def calculate_deltas(map_model: MapModel, position: jnp.array, scan_data: ScanData, model: MLPModel,
                     tracking_config: TrackingConfig):
    sampling_depths = scan_data.depths[..., None]
    points = calculate_points(sampling_depths, scan_data)
    points = transform_points(points, position).reshape(-1, 2)
    densities = calculate_densities(points, map_model, model)
    deltas = huber(densities, tracking_config.huber_scale)
    deltas = deltas / jnp.sqrt(scan_data.depths.shape[0])
    return deltas


@functools.partial(jax.jit, static_argnums=[3, 4])
def tracking_loss(map_model: MapModel, position: jnp.array, scan_data: ScanData, model: MLPModel,
                  tracking_config: TrackingConfig):
    return (calculate_deltas(map_model, position, scan_data, model, tracking_config) ** 2).sum()


@dataclasses.dataclass
class BatchPositionOptimizerConfig(OptimizePositionConfig):
    batch_size: int


@dataclasses.dataclass
class OptimizePositionState:
    iteration: int
    previous_hessian: jnp.array
    previous_grad: jnp.array


@functools.partial(jax.jit, static_argnums=[3, 4])
def batched_calculate_depth_deltas(map_model, positions, scan_data: ScanData, tracking_config, model, indices):
    positions = positions.reshape(-1, 3)[indices]
    return calculate_deltas(map_model, positions, scan_data, model, tracking_config)


@functools.partial(jax.jit, static_argnums=[3, 4])
def batch_loss_function(map_model, positions, scan_data, tracking_config, model, indices):
    positions = positions.reshape(-1, 3)[indices]
    return tracking_loss(map_model, positions, scan_data, model, tracking_config)


class BatchPositionOptimizer:
    def __init__(self, config: BatchPositionOptimizerConfig, model: MLPModel):
        self._tracking_config = config
        self.state: Optional[OptimizePositionState] = None
        self.jacobian_function = None
        self._hessian_adder = None
        self._model = model
        self.loss_history = []

    def setup(self):
        self._hessian_adder = self.make_block_matrix(self._tracking_config.hessian_adder)
        self.state = OptimizePositionState(iteration=0,
                                           previous_hessian=self.make_block_matrix(self._tracking_config.init_hessian),
                                           previous_grad=jnp.zeros(3 * self._tracking_config.batch_size))
        self.jacobian_function = jax.jit(jax.jacfwd(batched_calculate_depth_deltas, argnums=1), static_argnums=[3, 4])

    def step(self, optimized_positions: jnp.array, map_model, scan_data_batch: ScanDataBatch):
        jacobian = self.jacobian_function(map_model,
                                          optimized_positions,
                                          scan_data_batch.scan_data,
                                          self._tracking_config.tracking_config,
                                          self._model,
                                          scan_data_batch.indices)
        depth_deltas = batched_calculate_depth_deltas(map_model,
                                                      optimized_positions,
                                                      scan_data_batch.scan_data,
                                                      self._tracking_config.tracking_config,
                                                      self._model,
                                                      scan_data_batch.indices)
        grad = 2 * jnp.sum(jacobian * depth_deltas[:, None], axis=0)
        hessian = 2 * jacobian.T @ jacobian
        grad = self._tracking_config.beta2 * self.state.previous_grad + (1 - self._tracking_config.beta2) * grad
        hessian = self._tracking_config.beta1 * self.state.previous_hessian + (
                1 - self._tracking_config.beta1) * hessian
        previous_hessian = hessian
        hessian = hessian + self._hessian_adder
        delta = -(jnp.linalg.inv(hessian) @ grad) * self._tracking_config.learning_rate
        optimized_positions = optimized_positions + delta
        previous_grad = grad + hessian @ delta
        self.state = OptimizePositionState(self.state.iteration + 1, previous_hessian, previous_grad)
        loss = batch_loss_function(map_model, optimized_positions, scan_data_batch.scan_data,
                                   self._tracking_config.tracking_config, self._model, scan_data_batch.indices)
        self.loss_history.append(loss)
        return optimized_positions

    def make_block_matrix(self, matrix):
        n = matrix.shape[0]
        result = jnp.zeros((n * self._tracking_config.batch_size, n * self._tracking_config.batch_size))
        for i in range(self._tracking_config.batch_size):
            result = result.at[i * n: i * n + n, i * n: i * n + n].set(matrix)
        return result
