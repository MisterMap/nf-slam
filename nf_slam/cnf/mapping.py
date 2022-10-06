import dataclasses
import functools
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import optax
import tqdm
from flax.optim.adam import Adam

from nf_slam.laser_data import LaserData
from nf_slam.space_hashing_mapping.jax_math import calculate_layer_embeddings
from nf_slam.space_hashing_mapping.map_model import MapModel, init_map_model
from nf_slam.space_hashing_mapping.mapping import ScanData, LearningConfig, BuildMapState
from nf_slam.space_hashing_mapping.mapping import transform_points, calculate_points
from nf_slam.space_hashing_mapping.mlp_model import MLPModel


@dataclass(unsafe_hash=True)
class RectangleBoundary:
    min_x: float
    max_x: float
    min_y: float
    max_y: float

    @classmethod
    def from_vec(cls, vec):
        return cls(*vec)

    def apply(self, points):
        x = points[:, 0] * (self.max_x - self.min_x) + self.min_x
        y = points[:, 1] * (self.max_y - self.min_y) + self.min_y
        return jnp.stack([x, y], axis=1)


@jdc.pytree_dataclass
class LearningData:
    normal: jnp.array
    random_points: jnp.array

    @classmethod
    def from_config(cls, scan_data, config, key, boundary=None):
        if boundary is None:
            boundary = config.random_point_boundary
        random_points = jax.random.uniform(jax.random.PRNGKey(key), (config.random_point_count, 2))
        # noinspection PyArgumentList
        return cls(
            normal=jax.random.normal(jax.random.PRNGKey(key),
                                     (len(scan_data.depths), config.sampling_depth_count)),
            random_points=boundary.apply(random_points))


@dataclass(unsafe_hash=True)
class CNFMapModelConfig(object):
    minimal_depth: float
    maximal_depth: float
    F: int
    L: int
    T: int
    max_log_resolution: float
    min_log_resolution: float


@dataclass(unsafe_hash=True)
class CNFMapBuildingConfig:
    sampling_depth_delta: float
    sampling_depth_count: int
    classification_loss_weight: float
    point_loss_weight: float
    random_point_boundary: RectangleBoundary
    random_point_loss_weight: float
    random_point_count: int


def calculate_densities(points: jnp.array, map_model: MapModel, model: MLPModel):
    embedding = calculate_layer_embeddings(map_model.hashtable,
                                           points,
                                           map_model.resolutions,
                                           map_model.origins,
                                           map_model.hashtable.shape[1],
                                           map_model.hashtable.shape[0],
                                           map_model.rotations)
    embedding = jnp.transpose(embedding, (1, 0, 2)).reshape(embedding.shape[1], -1)
    return model.apply(map_model.variables, embedding)


@jax.jit
def transform_points(points, position):
    x = position[..., 0, None]
    y = position[..., 1, None]
    angle = position[..., 2, None]
    transformed_x = x + points[..., 0] * jnp.cos(angle) - points[..., 1] * jnp.sin(angle)
    transformed_y = y + points[..., 0] * jnp.sin(angle) + points[..., 1] * jnp.cos(angle)
    return jnp.stack([transformed_x, transformed_y], axis=-1)


@functools.partial(jax.jit, static_argnums=[4, 5, 6])
def mapping_classification_loss(map_model: MapModel, position: jnp.array, scan_data: ScanData,
                                learning_data: LearningData, map_model_config: CNFMapModelConfig, model: MLPModel,
                                config: CNFMapBuildingConfig):
    sampling_depth_deltas = jnp.where(learning_data.normal > 0, config.sampling_depth_delta,
                                      scan_data.depths[..., None] / 2)
    sampling_depths = learning_data.normal * sampling_depth_deltas + scan_data.depths[..., None]
    sampling_depths = jnp.clip(sampling_depths, 0, None)
    points = calculate_points(sampling_depths, scan_data)
    points = transform_points(points, position).reshape(-1, 2)
    labels = (jnp.sign(learning_data.normal.reshape(-1)) + 1) / 2.
    densities = calculate_densities(points, map_model, model)
    # labels = jnp.where(labels < 0.5, labels, 0.75 * jax.nn.sigmoid(densities) + 0.25)
    return optax.sigmoid_binary_cross_entropy(densities, labels).mean()


@functools.partial(jax.jit, static_argnums=[4, 5, 6])
def mapping_point_loss(map_model: MapModel, position: jnp.array, scan_data: ScanData,
                       learning_data: LearningData, map_model_config: CNFMapModelConfig, model: MLPModel,
                       config: CNFMapBuildingConfig):
    sampling_depths = scan_data.depths[..., None]
    points = calculate_points(sampling_depths, scan_data)
    points = transform_points(points, position).reshape(-1, 2)
    densities = calculate_densities(points, map_model, model)
    return (densities ** 2).mean()


@functools.partial(jax.jit, static_argnums=[4, 5, 6])
def random_mapping_point_loss(map_model: MapModel, position: jnp.array, scan_data: ScanData,
                              learning_data: LearningData, map_model_config: CNFMapModelConfig, model: MLPModel,
                              config: CNFMapBuildingConfig):
    points = learning_data.random_points
    densities = calculate_densities(points, map_model, model)
    return (densities ** 2).mean()


@functools.partial(jax.jit, static_argnums=[4, 5, 6])
def mapping_loss(map_model: MapModel, position: jnp.array, scan_data: ScanData,
                 learning_data: LearningData, map_model_config: CNFMapModelConfig, model: MLPModel,
                 config: CNFMapBuildingConfig):
    return config.point_loss_weight * mapping_point_loss(map_model, position, scan_data,
                                                         learning_data, map_model_config, model, config) + \
           config.classification_loss_weight * mapping_classification_loss(map_model, position, scan_data,
                                                                           learning_data, map_model_config, model,
                                                                           config) + \
           config.random_point_loss_weight * random_mapping_point_loss(map_model, position, scan_data,
                                                                         learning_data, map_model_config, model, config)


class CNFMapBuilder:
    def __init__(self, learning_config: LearningConfig, map_building_config: CNFMapBuildingConfig,
                 map_model_config: CNFMapModelConfig, mlp_model: MLPModel):
        self.state: Optional[BuildMapState] = None
        self.grad_function = None
        self._variable_optimizer = Adam(**dataclasses.asdict(learning_config.variable_optimizer_config))
        self._hashtable_optimizer = Adam(**dataclasses.asdict(learning_config.hashtable_optimizer_config))
        self._learning_config = learning_config
        self._map_building_config = map_building_config
        self._map_model_config = map_model_config
        self._mlp_model = mlp_model
        self.loss_history = []

    def setup(self, map_model: MapModel):
        self.grad_function = jax.jit(jax.value_and_grad(mapping_loss), static_argnums=[4, 5, 6])
        self.state = BuildMapState(
            iteration=0,
            variable_state=self._variable_optimizer.init_state(map_model.variables),
            hashtable_state=self._hashtable_optimizer.init_state(map_model.hashtable)
        )

    def step(self, map_model: MapModel, position: jnp.array, scan_data: ScanData):
        borders = self._get_borders(scan_data, position)
        learning_data = LearningData.from_config(scan_data, self._map_building_config, self.state.iteration, borders)
        loss, grad = self.grad_function(map_model, position, scan_data, learning_data, self._map_model_config,
                                        self._mlp_model,
                                        self._map_building_config)
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
        self.loss_history.append(loss)
        return map_model

    def build_map(self, laser_data: LaserData, position: jnp.array):
        scan_data = ScanData.from_laser_data(laser_data)
        map_model = init_map_model(self._mlp_model, self._map_model_config)
        self.setup(map_model)
        for i in tqdm.tqdm(range(self._learning_config.iterations)):
            map_model = self.step(map_model, position, scan_data)
        return map_model

    @staticmethod
    def _get_borders(scan_data, position):
        points = calculate_points(scan_data.depths, scan_data)
        points = transform_points(points, position).reshape(-1, 2)
        return RectangleBoundary(jnp.max(points[:, 0]), jnp.min(points[:, 0]), jnp.max(points[:, 1]),
                                 jnp.min(points[:, 1]))
