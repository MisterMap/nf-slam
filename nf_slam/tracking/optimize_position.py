import dataclasses
import functools
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

from nf_slam.laser_data import LaserData
from nf_slam.position_2d import Position2D
from nf_slam.space_hashing_mapping.map_model import MapModel, ModelConfig
from nf_slam.space_hashing_mapping.mapping import predict_depths, ScanData, LearningData
from nf_slam.space_hashing_mapping.mlp_model import MLPModel


@functools.partial(jax.jit, static_argnums=[4, 5])
def loss_function_without_normalization(map_model: MapModel, position: jnp.array, scan_data: ScanData,
                                        learning_data: LearningData, config: ModelConfig, model: MLPModel):
    predicted_depths = predict_depths(map_model, position, scan_data, learning_data, config, model)
    return jnp.sum(((scan_data.depths - predicted_depths) ** 2))


@dataclasses.dataclass
class OptimizePositionConfig:
    iterations: int  # 100
    init_hessian: jnp.array  # jnp.diag(jnp.array([2000, 2000, 200]))
    maximal_clip_norm: float  # 30
    beta1: float  # 0.7
    beta2: float  # 0.4
    hessian_adder: jnp.array  # jnp.diag(jnp.array([20, 20, 2]))


@dataclasses.dataclass
class OptimizePositionResult:
    loss_history: List[float]
    position_history: List[Position2D]
    optimized_position: jnp.array


def optimize_position(laser_data: LaserData, map_model: MapModel, init_position: jnp.array, model_config: ModelConfig,
                      config: OptimizePositionConfig, model: MLPModel):
    optimized_position = init_position
    jacobian_function = jax.jit(jax.jacfwd(predict_depths, argnums=1), static_argnums=[4, 5])
    scan_data = ScanData.from_laser_data(laser_data)
    learning_data = LearningData(uniform=jax.random.uniform(
        jax.random.PRNGKey(0),
        (len(scan_data.depths), model_config.bins_count)))
    loss_history = [
        loss_function_without_normalization(map_model, optimized_position, scan_data, learning_data, model_config,
                                            model)]
    position_history = [Position2D.from_vec(np.array(optimized_position))]
    previous_hessian = config.init_hessian
    previous_grad = jnp.zeros(3)
    for i in range(config.iterations):
        learning_data = LearningData(uniform=jax.random.uniform(
            jax.random.PRNGKey(0),
            (len(scan_data.depths), model_config.bins_count)))
        jacobian = jacobian_function(map_model, optimized_position, scan_data, learning_data, config, model)
        jacobian_norm = jnp.linalg.norm(jacobian, axis=1)
        clipped_norm = jnp.clip(jacobian_norm, 0, config.maximal_clip_norm)
        jacobian = jacobian / jacobian_norm[:, None] * clipped_norm[:, None]
        predicted_depths = predict_depths(map_model, optimized_position, scan_data, learning_data, model_config, model)
        grad = 2 * jnp.sum(jacobian * (predicted_depths - scan_data.depths)[:, None], axis=0)
        hessian = 2 * jacobian.T @ jacobian + config.hessian_adder
        grad = config.beta2 * previous_grad + (1 - config.beta2) * grad
        hessian = config.beta1 * previous_hessian + (1 - config.beta1) * hessian
        previous_grad = grad
        previous_hessian = hessian
        delta = -(jnp.linalg.inv(hessian) @ grad)
        optimized_position = optimized_position + delta
        loss = loss_function_without_normalization(map_model, optimized_position, scan_data, learning_data,
                                                   model_config, model)
        loss_history.append(loss)
        position_history.append(Position2D.from_vec(np.array(optimized_position)))
    return OptimizePositionResult(loss_history, position_history, optimized_position)
