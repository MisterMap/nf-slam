import dataclasses
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from nf_slam.position_2d import Position2D
from nf_slam.space_hashing_mapping.map_model import MapModelConfig
from nf_slam.space_hashing_mapping.mapping import ScanData, LearningData
from nf_slam.space_hashing_mapping.mlp_model import MLPModel
from nf_slam.tracking.tracking import OptimizePositionConfig, OptimizePositionState, calculate_depth_deltas, \
    loss_function_without_normalization


def batched_calculate_depth_deltas(map_model, positions, scan_data: ScanData, learning_data, config, model, indices):
    positions = positions.reshape(-1, 3)[indices]
    return calculate_depth_deltas(map_model, positions, scan_data, learning_data, config, model)


def batch_loss_function(map_model, positions, scan_data, learning_data, config, model, indices):
    positions = positions.reshape(-1, 3)[indices]
    return loss_function_without_normalization(map_model, positions, scan_data, learning_data, config, model)


@dataclasses.dataclass
class ScanDataBatch:
    indices: jnp.array
    scan_data: ScanData

    @classmethod
    def from_data_list(cls, data_list, points_per_scan):
        depths = []
        angles = []
        local_indices = []
        for i, laser_data in enumerate(data_list):
            mask = laser_data.ranges < 10
            indices = np.arange(mask.shape[0])[mask]
            indices = np.random.choice(indices, points_per_scan)
            depths.extend(list(laser_data.ranges[indices]))
            angles.extend(list(laser_data.angles[indices]))
            local_indices.extend([i] * points_per_scan)
        return cls(
            indices=jnp.array(local_indices),
            scan_data=ScanData(
                depths=jnp.array(depths),
                angles=jnp.array(angles)
            )
        )


@dataclasses.dataclass
class BatchPositionOptimizerConfig(OptimizePositionConfig):
    batch_size: int


class BatchPositionOptimizer:
    def __init__(self, config: BatchPositionOptimizerConfig, map_model_config: MapModelConfig, model: MLPModel):
        self._config = config
        self._map_model_config = map_model_config
        self.state: Optional[OptimizePositionState] = None
        self.jacobian_function = None
        self._hessian_adder = None
        self._model = model
        self.loss_history = []
        self.position_history = []

    def setup(self):
        self._hessian_adder = self.make_block_matrix(self._config.hessian_adder)
        self.state = OptimizePositionState(iteration=0,
                                           previous_hessian=self.make_block_matrix(self._config.init_hessian),
                                           previous_grad=jnp.zeros(3 * self._config.batch_size))
        self.jacobian_function = jax.jit(jax.jacfwd(batched_calculate_depth_deltas, argnums=1), static_argnums=[4, 5])

    def step(self, optimized_positions: jnp.array, map_model, scan_data_batch: ScanDataBatch):
        learning_data = LearningData(uniform=jax.random.uniform(
            jax.random.PRNGKey(self.state.iteration),
            (len(scan_data_batch.scan_data.depths), self._map_model_config.bins_count)))
        jacobian = self.jacobian_function(map_model,
                                          optimized_positions,
                                          scan_data_batch.scan_data,
                                          learning_data,
                                          self._map_model_config,
                                          self._model,
                                          scan_data_batch.indices)
        jacobian_norm = jnp.linalg.norm(jacobian, axis=1) + 1e-4
        clipped_norm = jnp.clip(jacobian_norm, 0, self._config.maximal_clip_norm)
        jacobian = jacobian / jacobian_norm[:, None] * clipped_norm[:, None]
        # jacobian = jnp.clip(jacobian, -self._config.maximal_clip_norm, self._config.maximal_clip_norm)
        depth_deltas = batched_calculate_depth_deltas(map_model,
                                                      optimized_positions,
                                                      scan_data_batch.scan_data,
                                                      learning_data,
                                                      self._map_model_config,
                                                      self._model,
                                                      scan_data_batch.indices)
        grad = 2 * jnp.sum(jacobian * depth_deltas[:, None], axis=0)
        hessian = 2 * jacobian.T @ jacobian
        grad = self._config.beta2 * self.state.previous_grad + (1 - self._config.beta2) * grad
        hessian = self._config.beta1 * self.state.previous_hessian + (1 - self._config.beta1) * hessian
        previous_hessian = hessian
        hessian = hessian + self._hessian_adder
        delta = -(jnp.linalg.inv(hessian) @ grad) * self._config.learning_rate
        optimized_positions = optimized_positions + delta
        previous_grad = grad + hessian @ delta
        # previous_grad = grad
        self.state = OptimizePositionState(self.state.iteration + 1, previous_hessian, previous_grad)
        loss = batch_loss_function(map_model, optimized_positions, scan_data_batch.scan_data,
                                   learning_data,
                                   self._map_model_config, self._model, scan_data_batch.indices)
        self.loss_history.append(loss)
        self.position_history.append(Position2D.from_vec(np.array(optimized_positions)))
        return optimized_positions

    def make_block_matrix(self, matrix):
        n = matrix.shape[0]
        result = jnp.zeros((n * self._config.batch_size, n * self._config.batch_size))
        for i in range(self._config.batch_size):
            result = result.at[i * n: i * n + n, i * n: i * n + n].set(matrix)
        return result
