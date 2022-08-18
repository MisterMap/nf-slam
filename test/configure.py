import jax.numpy as jnp
import numpy as np
import pytest

from nf_slam.cnf.mapping import CNFMapBuilder, CNFMapBuildingConfig, RectangleBoundary, CNFMapModelConfig
from nf_slam.cnf.odometry import CNFOdometry, CNFOdometryConfig
from nf_slam.cnf.tracking import CNFPositionOptimizer, CNFPositionOptimizerConfig, TrackingConfig
from nf_slam.lidar_dataset_loader import LidarDatasetLoaderConfig, LidarDatasetLoader
from nf_slam.space_hashing_mapping.map_model import MapModelConfig, init_map_model
from nf_slam.space_hashing_mapping.mapping import MapBuilder, LearningConfig, OptimizerConfig, ScanData
from nf_slam.space_hashing_mapping.mlp_model import MLPModel
from nf_slam.tracking.batch_tracking import BatchPositionOptimizerConfig, BatchPositionOptimizer, ScanDataBatch
from nf_slam.tracking.tracking import OptimizePositionConfig, PositionOptimizer
from nf_slam.utils.universal_factory import UniversalFactory


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
        max_log_resolution=1.,
        huber_delta=2.
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
        hessian_adder=jnp.diag(jnp.array([20, 20, 2])),
        learning_rate=0.9,
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
def cnf_position_optimizer(position_optimization_config, map_model_config, mlp_model):
    return PositionOptimizer(position_optimization_config, map_model_config, mlp_model)


@pytest.fixture
def scan_data(laser_data):
    return ScanData.from_laser_data(laser_data)


@pytest.fixture
def batch_position_optimization_config():
    return BatchPositionOptimizerConfig(
        iterations=100,
        init_hessian=jnp.diag(jnp.array([2000, 2000, 200])),
        maximal_clip_norm=30,
        beta1=0.7,
        beta2=0.4,
        hessian_adder=jnp.diag(jnp.array([20, 20, 2])),
        batch_size=5,
        learning_rate=0.9
    )


@pytest.fixture
def batch_position_optimizer(batch_position_optimization_config, map_model_config, mlp_model):
    return BatchPositionOptimizer(batch_position_optimization_config, map_model_config, mlp_model)


@pytest.fixture
def scan_data_batch(laser_data_list):
    return ScanDataBatch.from_data_list(laser_data_list[:5], 100)


@pytest.fixture
def batch_init_position(laser_data_list):
    return jnp.array([x.odometry_position.as_vec() for x in laser_data_list[:5]]).reshape(-1)


@pytest.fixture
def cnf_odometry_config():
    return CNFOdometryConfig(
        tracking_iterations=2,
        mapping_iterations=2,
        batch_mapping_iterations=2,
        mapping_batch_count=5,
        point_count=200
    )


@pytest.fixture
def bounds():
    return np.array([0, 10, 0, 10])


@pytest.fixture
def cnf_map_building_config(bounds):
    return CNFMapBuildingConfig(
        sampling_depth_delta=3.,
        sampling_depth_count=200,
        point_loss_weight=1.,
        classification_loss_weight=1.,
        random_point_boundary=RectangleBoundary(*bounds),
        random_point_loss_weight=100.,
        random_point_count=100,
    )


@pytest.fixture
def cnf_map_model_config():
    return CNFMapModelConfig(
        minimal_depth=0.05,
        maximal_depth=10,
        F=4,
        L=16,
        T=4096,
        min_log_resolution=-0.5,
        max_log_resolution=2.,
    )


@pytest.fixture
def cnf_batch_map_builder(learning_config, cnf_map_building_config, cnf_map_model_config, mlp_model):
    return CNFMapBuilder(learning_config, cnf_map_building_config, cnf_map_model_config, mlp_model)


@pytest.fixture
def cnf_map_builder(learning_config, cnf_map_building_config, cnf_map_model_config, mlp_model):
    return CNFMapBuilder(learning_config, cnf_map_building_config, cnf_map_model_config, mlp_model)


@pytest.fixture
def cnf_position_optimizer_config():
    return CNFPositionOptimizerConfig(
        learning_rate=0.9,
        iterations=100,
        init_hessian=jnp.diag(jnp.array([200, 200, 200])),
        maximal_clip_norm=100,
        beta1=0.5,
        beta2=0.3,
        hessian_adder=jnp.diag(jnp.array([200, 200, 200])) * 0.1,
        tracking_config=TrackingConfig(huber_scale=0.3),
    )


@pytest.fixture
def cnf_position_optimizer(cnf_position_optimizer_config, mlp_model):
    return CNFPositionOptimizer(cnf_position_optimizer_config, mlp_model)


@pytest.fixture
def cnf_odometry(cnf_odometry_config, cnf_batch_map_builder, cnf_position_optimizer, cnf_map_builder,
                 mlp_model, cnf_map_model_config):
    return CNFOdometry(cnf_odometry_config, cnf_batch_map_builder, cnf_position_optimizer, cnf_map_builder,
                       mlp_model, cnf_map_model_config)


@pytest.fixture
def universal_factory():
    return UniversalFactory([CNFOdometry, MLPModel, CNFMapModelConfig, LearningConfig, CNFMapBuildingConfig])


@pytest.fixture
def cnf_odometry_full_config_dict(bounds):
    result = {
        'mlp_model': {
            "type": "MLPModel",
        },
        'map_model_config': {
            'type': "CNFMapModelConfig",
            'minimal_depth': 0.05,
            'maximal_depth': 10,
            'F': 4,
            'L': 16,
            'T': 4096,
            'min_log_resolution': -0.5,
            'max_log_resolution': 2.,
        },
        'learning_config': {
            "type": "LearningConfig",
            'iterations': 100,
            'variable_optimizer_config': {
                'learning_rate': 2e-2,
                'beta1': 0.9,
                'beta2': 0.99,
                'eps': 1e-15,
                'weight_decay': 1e-6
            },
            'hashtable_optimizer_config': {
                'learning_rate': 2e-2,
                'beta1': 0.9,
                'beta2': 0.99,
                'eps': 1e-15,
                'weight_decay': 0
            }
        },
        'map_building_config': {
            "type": "CNFMapBuildingConfig",
            'sampling_depth_delta': 3.,
            'sampling_depth_count': 200,
            'point_loss_weight': 1.,
            'classification_loss_weight': 1.,
            'random_point_loss_weight': 100.,
            'random_point_count': 100,
            "random_point_boundary": RectangleBoundary(*bounds)
        },
        'cnf_odometry': {
            "type": "CNFOdometry",
            "parameters": {
                'tracking_iterations': 2,
                'mapping_iterations': 2,
                'batch_mapping_iterations': 2,
                'mapping_batch_count': 5,
                'point_count': 200
            },
            "batch_map_builder": {},
            "position_optimizer": {
                "config": {
                    'learning_rate': 0.9,
                    'iterations': 100,
                    'init_hessian': jnp.diag(jnp.array([200, 200, 200])),
                    'maximal_clip_norm': 100,
                    'beta1': 0.5,
                    'beta2': 0.3,
                    'hessian_adder': jnp.diag(jnp.array([200, 200, 200])) * 0.1,
                    'tracking_config': {'huber_scale': 0.3}
                }
            },
            "map_builder": {}
        }
    }
    return result
