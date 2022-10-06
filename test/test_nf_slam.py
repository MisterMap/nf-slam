import jax

from configure import *
from nf_slam.position_2d import Position2D
from nf_slam.space_hashing_mapping.jax_math import step_function


def test_dataset_loader(dataset_loader, datafile):
    laser_data_list = dataset_loader.load(datafile)
    assert len(laser_data_list) == 910


def test_laser_data(laser_data):
    assert len(laser_data.angles) == 180


def test_step_function(input_x: jnp.array):
    output = step_function(input_x)
    assert output.shape[0] == 3


def test_init_map_model(mlp_model, map_model_config):
    init_map_model(mlp_model, map_model_config)


def test_init_model(mlp_model, mlp_model_init_batch):
    variables = mlp_model.init(jax.random.PRNGKey(1), jax.block_until_ready(mlp_model_init_batch))


def test_map_builder_step(map_builder: MapBuilder, map_model, init_position, scan_data):
    map_builder.setup(scan_data, map_model, init_position)
    result = map_builder.step(map_model, init_position, scan_data)


def test_map_builder_build_map(map_builder, laser_data, init_position):
    map_builder.build_map(laser_data, init_position)


def test_position_optimizer_step(cnf_position_optimizer, init_position, map_model, scan_data):
    cnf_position_optimizer.setup()
    cnf_position_optimizer.step(init_position, map_model, scan_data)


def test_position_optimizer_optimize_position(cnf_position_optimizer, laser_data, map_model, init_position):
    result = cnf_position_optimizer.find_position(laser_data, map_model, init_position)
    assert len(cnf_position_optimizer.position_history) == 100
    assert type(cnf_position_optimizer.position_history[0]) == Position2D


def test_batch_position_optimizer_config(batch_position_optimizer: BatchPositionOptimizer, batch_init_position,
                                         map_model, scan_data_batch):
    batch_position_optimizer.setup()
    result = batch_position_optimizer.step(batch_init_position, map_model, scan_data_batch)


def test_cnf_odometry_setup(cnf_odometry):
    cnf_odometry.setup()


def test_cnf_odometry_step(cnf_odometry, laser_data):
    cnf_odometry.setup()
    cnf_odometry.step(laser_data)


def test_cnf_odometry_second_step(cnf_odometry, laser_data):
    cnf_odometry.setup()
    cnf_odometry.step(laser_data)
    cnf_odometry.step(laser_data)


def test_universal_factory(universal_factory, cnf_odometry_full_config_dict):
    order = ["mlp_model", "map_model_config", "map_model_config", "learning_config", "map_building_config",
             "cnf_odometry"]
    odometry = universal_factory.iterative_make(order, cnf_odometry_full_config_dict)
    odometry.setup()
