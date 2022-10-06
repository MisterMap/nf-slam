import dataclasses
from typing import Union

import jax.numpy as jnp
import numpy as np

from nf_slam.cnf.mapping import CNFMapBuilder, CNFMapModelConfig
from nf_slam.cnf.tracking import CNFPositionOptimizer
from nf_slam.laser_data import LaserData
from nf_slam.position_2d import Position2D
from nf_slam.space_hashing_mapping.map_model import init_map_model
from nf_slam.space_hashing_mapping.mapping import ScanData
from nf_slam.space_hashing_mapping.mlp_model import MLPModel, NormMLPModel


@dataclasses.dataclass
class CNFOdometryConfig:
    tracking_iterations: int
    mapping_iterations: int
    batch_mapping_iterations: int
    mapping_batch_count: int
    point_count: int


def position2d_from_jax(jax_position):
    return Position2D.from_vec(np.array(jax_position))


@dataclasses.dataclass
class DataPointBatch(object):
    depths: jnp.array
    angles: jnp.array
    x: jnp.array
    y: jnp.array
    angle: jnp.array

    @property
    def scan_data(self):
        return ScanData(depths=self.depths, angles=self.angles)

    @property
    def position(self):
        return jnp.stack([self.x, self.y, self.angle], axis=1)


def get_random_data_point_batch(laser_data_list, scan_count, points_per_scan, positions):
    laser_data_indices = np.random.choice(np.arange(len(laser_data_list)), scan_count - 1)
    laser_data_indices = np.append(laser_data_indices, len(laser_data_list) - 1)
    depths = []
    angles = []
    x = []
    y = []
    angle = []
    for i in range(scan_count):
        laser_data = laser_data_list[laser_data_indices[i]]
        position = positions[laser_data_indices[i]]
        mask = laser_data.ranges < 10
        indices = np.arange(mask.shape[0])[mask]
        indices = np.random.choice(indices, points_per_scan)
        depths.extend(list(laser_data.ranges[indices]))
        angles.extend(list(laser_data.angles[indices]))
        x.extend(list(np.full(indices.shape[0], position.x)))
        y.extend(list(np.full(indices.shape[0], position.y)))
        angle.extend(list(np.full(indices.shape[0], position.rotation)))
    return DataPointBatch(
        depths=jnp.array(depths),
        angles=jnp.array(angles),
        x=jnp.array(x),
        y=jnp.array(y),
        angle=jnp.array(angle)
    )


class CNFOdometry:
    def __init__(self, parameters: CNFOdometryConfig, batch_map_builder: CNFMapBuilder,
                 position_optimizer: CNFPositionOptimizer, map_builder: CNFMapBuilder,
                 mlp_model: Union[MLPModel, NormMLPModel], map_model_config: CNFMapModelConfig):
        self._batch_map_builder = batch_map_builder
        self._map_builder = map_builder
        self._position_optimizer = position_optimizer
        self._parameters = parameters
        self._mlp_model = mlp_model
        self._map_config = map_model_config
        self.map_model = None
        self.tracked_position = None
        self.processed_laser_data_list = []
        self.reconstructed_odometry_positions = []
        self.previous_wheel_odometry = None
        self.iteration = 0

    def setup(self):
        self._position_optimizer.setup()
        self.map_model = init_map_model(self._mlp_model, self._map_config)
        self._map_builder.setup(self.map_model)
        self._batch_map_builder.setup(self.map_model)
        self.tracked_position = jnp.zeros(3)
        self.previous_wheel_odometry = Position2D.from_vec(np.zeros(3))
        self.iteration = 0

    def step(self, laser_data: LaserData):
        scan_data = ScanData.from_laser_data(laser_data)
        self._add_odometry(laser_data.odometry_position)
        self._optimize_position(scan_data)
        self._optimize_map(scan_data)
        self.processed_laser_data_list.append(laser_data)
        self.reconstructed_odometry_positions.append(position2d_from_jax(self.tracked_position))
        self._batch_optimize_map()
        self.iteration += 1

    def _add_odometry(self, wheel_odometry):
        current_position = position2d_from_jax(self.tracked_position)
        current_position = current_position * self.previous_wheel_odometry.inv() * wheel_odometry
        self.previous_wheel_odometry = wheel_odometry
        self.tracked_position = jnp.array(current_position.as_vec())

    def _optimize_map(self, scan_data):
        for i in range(self._parameters.mapping_iterations):
            self.map_model = self._map_builder.step(
                self.map_model, self.tracked_position, scan_data.get_random_subset(self._parameters.point_count))

    def _optimize_position(self, scan_data):
        if self.iteration == 0:
            return
        for i in range(self._parameters.tracking_iterations):
            self.tracked_position = self._position_optimizer.step(
                self.tracked_position, self.map_model, scan_data.get_random_subset(self._parameters.point_count))

    def _batch_optimize_map(self):
        for i in range(self._parameters.batch_mapping_iterations):
            batch = get_random_data_point_batch(
                self.processed_laser_data_list, self._parameters.mapping_batch_count, self._parameters.point_count,
                self.reconstructed_odometry_positions)
            self.map_model = self._batch_map_builder.step(self.map_model, batch.position, batch.scan_data)

    @property
    def current_position(self):
        return position2d_from_jax(self.tracked_position)
