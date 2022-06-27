import jax.numpy as jnp
import pytest

from nf_slam.lidar_dataset_loader import LidarDatasetLoaderConfig, LidarDatasetLoader
from nf_slam.space_hashing_mapping.map_model import MapModelConfig, init_map_model
from nf_slam.space_hashing_mapping.mapping import MapBuilder, LearningConfig, OptimizerConfig, BuildMapResult, ScanData
from nf_slam.space_hashing_mapping.mlp_model import MLPModel
from nf_slam.tracking.tracking import OptimizePositionConfig, PositionOptimizer, OptimizePositionResult


@pytest.fixture
def datafile():
    return "/home/mikhail/Downloads/intel.gfs(4).log"


@pytest.fixture
def dataset_loader():
    lidar_dataset_loader_config = LidarDatasetLoaderConfig(
        maximal_token_length=180,
        minimal_angle=90.,
        maximal_angle=90.,
        maximal_distance=10.
    )
    return LidarDatasetLoader(lidar_dataset_loader_config)


@pytest.fixture
def laser_data_list(dataset_loader, datafile):
    return dataset_loader.load(datafile)


@pytest.fixture
def laser_data(laser_data_list):
    return laser_data_list[0]


@pytest.fixture
def mlp_model():
    return MLPModel()


@pytest.fixture
def map_model_config():
    return MapModelConfig(
        minimal_depth=0.05,
        maximal_depth=10,
        bins_count=60,
        density_scale=0.05,
        variance_weight=0.1,
        F=32,
        L=4,
        T=2048,
        min_log_resolution=-4,
        max_log_resolution=1.
    )


@pytest.fixture
def input_x():
    return jnp.array([1, 2, 3])


@pytest.fixture
def mlp_model_init_batch(map_model_config):
    return jnp.ones([10, map_model_config.F * map_model_config.L])


@pytest.fixture
def map_model(mlp_model, map_model_config):
    return init_map_model(mlp_model, map_model_config)


@pytest.fixture
def init_position():
    return jnp.array([0., 0, 0])


@pytest.fixture
def position_optimization_config():
    return OptimizePositionConfig(
        iterations=100,
        init_hessian=jnp.diag(jnp.array([2000, 2000, 200])),
        maximal_clip_norm=30,
        beta1=0.7,
        beta2=0.4,
        hessian_adder=jnp.diag(jnp.array([20, 20, 2]))
    )


@pytest.fixture
def learning_config():
    return LearningConfig(
        iterations=100,
        variable_optimizer_config=OptimizerConfig(
            learning_rate=2e-2,
            beta1=0.9,
            beta2=0.99,
            eps=1e-15,
            weight_decay=1e-6,
        ),
        hashtable_optimizer_config=OptimizerConfig(
            learning_rate=2e-2,
            beta1=0.9,
            beta2=0.99,
            eps=1e-15,
            weight_decay=0,
        )
    )


@pytest.fixture
def map_builder(learning_config, map_model_config, mlp_model):
    return MapBuilder(learning_config, map_model_config, mlp_model)


@pytest.fixture
def position_optimizer(position_optimization_config, map_model_config, mlp_model):
    return PositionOptimizer(position_optimization_config, map_model_config, mlp_model)


@pytest.fixture
def build_map_result(map_model):
    return BuildMapResult([], map_model)


@pytest.fixture
def scan_data(laser_data):
    return ScanData.from_laser_data(laser_data)


@pytest.fixture
def optimize_position_result(init_position):
    return OptimizePositionResult([], [], init_position)
