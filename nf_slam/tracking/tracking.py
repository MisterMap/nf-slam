import dataclasses
import functools
from typing import List, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from nf_slam.laser_data import LaserData
from nf_slam.position_2d import Position2D
from nf_slam.space_hashing_mapping.map_model import MapModel, MapModelConfig
from nf_slam.space_hashing_mapping.mapping import predict_depths, ScanData, LearningData, predict_depths_and_variances
from nf_slam.space_hashing_mapping.mlp_model import MLPModel


@functools.partial(jax.jit, static_argnums=[1, ])
def huber(input_array, delta):
    mask = jnp.abs(input_array) < delta
    result = jnp.where(mask, input_array, jnp.sign(input_array) * (delta * (2 * jnp.abs(input_array) - delta)) ** 0.5)
    return result


@functools.partial(jax.jit, static_argnums=[4, 5])
def loss_function_without_normalization(map_model: MapModel, position: jnp.array, scan_data: ScanData,
                                        learning_data: LearningData, config: MapModelConfig, model: MLPModel):
    predicted_depths = predict_depths(map_model, position, scan_data, learning_data, config, model)
    return jnp.sqrt(jnp.sum((huber(scan_data.depths - predicted_depths, 2)) ** 2) / predicted_depths.shape[0])


@dataclasses.dataclass
class OptimizePositionConfig:
    learning_rate: float
    iterations: int  # 100
    init_hessian: jnp.array  # jnp.diag(jnp.array([2000, 2000, 200]))
    maximal_clip_norm: float  # 30
    beta1: float  # 0.7
    beta2: float  # 0.4
    hessian_adder: jnp.array  # jnp.diag(jnp.array([20, 20, 2]))


@dataclasses.dataclass
class OptimizePositionState:
    iteration: int
    previous_hessian: jnp.array
    previous_grad: jnp.array


@dataclasses.dataclass
class OptimizePositionData:
    map_model: MapModel
    config: OptimizePositionConfig
    scan_data: ScanData
    map_model_config: MapModelConfig
    jacobian_function: Callable


def calculate_depth_deltas(map_model, position, scan_data, learning_data, config: MapModelConfig, model):
    depths, variances = predict_depths_and_variances(map_model, position, scan_data, learning_data, config, model)
    scale = jax.lax.stop_gradient(variances + 1e-1) ** 0.5
    deltas = (depths - scan_data.depths) / scale
    return huber(deltas, config.huber_delta)


class PositionOptimizer:
    def __init__(self, config: OptimizePositionConfig, map_model_config: MapModelConfig, model: MLPModel):
        self._config = config
        self._map_model_config = map_model_config
        self.state: Optional[OptimizePositionState] = None
        self.jacobian_function = None
        self._model = model
        self.loss_history = []
        self.position_history = []

    def setup(self):
        self.state = OptimizePositionState(iteration=0, previous_hessian=self._config.init_hessian,
                                           previous_grad=jnp.zeros(3))
        self.jacobian_function = jax.jit(jax.jacfwd(calculate_depth_deltas, argnums=1), static_argnums=[4, 5])

    def step(self, optimized_position: jnp.array, map_model, scan_data):
        learning_data = LearningData(uniform=jax.random.uniform(
            jax.random.PRNGKey(self.state.iteration),
            (len(scan_data.depths), self._map_model_config.bins_count)))
        jacobian = self.jacobian_function(map_model, optimized_position, scan_data,
                                          learning_data,
                                          self._map_model_config, self._model)
        jacobian_norm = jnp.linalg.norm(jacobian, axis=1) + 1e-4
        clipped_norm = jnp.clip(jacobian_norm, 0, self._config.maximal_clip_norm)
        jacobian = jacobian / jacobian_norm[:, None] * clipped_norm[:, None]
        depth_deltas = calculate_depth_deltas(map_model, optimized_position, scan_data,
                                              learning_data, self._map_model_config, self._model)
        grad = 2 * jnp.sum(jacobian * depth_deltas[:, None], axis=0)
        hessian = 2 * jacobian.T @ jacobian
        grad = self._config.beta2 * self.state.previous_grad + (1 - self._config.beta2) * grad
        hessian = self._config.beta1 * self.state.previous_hessian + (1 - self._config.beta1) * hessian
        previous_hessian = hessian
        hessian = hessian + self._config.hessian_adder
        delta = -(jnp.linalg.inv(hessian) @ grad) * self._config.learning_rate
        optimized_position = optimized_position + delta
        previous_grad = grad + hessian @ delta
        self.state = OptimizePositionState(self.state.iteration + 1, previous_hessian, previous_grad)
        loss = loss_function_without_normalization(map_model, optimized_position, scan_data, learning_data,
                                                   self._map_model_config, self._model)
        self.loss_history.append(loss)
        self.position_history.append(Position2D.from_vec(np.array(optimized_position)))
        return optimized_position

    def find_position(self, laser_data: LaserData, map_model: MapModel, init_position: jnp.array):
        self.setup()
        scan_data = ScanData.from_laser_data(laser_data)
        # scan_data = ScanData(depths=scan_data.depths[-30:], angles=scan_data.angles[-30:])
        learning_data = LearningData(uniform=jax.random.uniform(
            jax.random.PRNGKey(0),
            (len(scan_data.depths), self._map_model_config.bins_count)))
        loss = loss_function_without_normalization(map_model, init_position, scan_data, learning_data,
                                                   self._map_model_config,
                                                   self._model)
        position = init_position
        for _ in tqdm.tqdm(range(self._config.iterations)):
            position = self.step(position, map_model, scan_data)
        return position
