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
    print(type(mlp_model))
    init_map_model(mlp_model, map_model_config)


def test_init_model(mlp_model, mlp_model_init_batch):
    variables = mlp_model.init(jax.random.PRNGKey(1), jax.block_until_ready(mlp_model_init_batch))


def test_map_builder_step(map_builder: MapBuilder, build_map_result, init_position, scan_data, map_model):
    map_builder.setup(scan_data, map_model, init_position)
    result = map_builder.step(build_map_result, init_position, scan_data)


def test_map_builder_build_map(map_builder, laser_data, init_position):
    map_builder.build_map(laser_data, init_position)


def test_position_optimizer_step(position_optimizer, optimize_position_result, map_model, scan_data):
    position_optimizer.setup()
    position_optimizer.step(optimize_position_result, map_model, scan_data)


def test_position_optimizer_optimize_position(position_optimizer, laser_data, map_model, init_position):
    result = position_optimizer.find_position(laser_data, map_model, init_position)
    assert len(result.position_history) == 101
    assert type(result.position_history[0]) == Position2D
