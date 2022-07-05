import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from nf_slam.laser_data import LaserData
from nf_slam.position_2d import Position2D
from nf_slam.space_hashing_mapping.jax_math import calculate_densities
from nf_slam.space_hashing_mapping.map_model import MapModel, MapModelConfig
from nf_slam.space_hashing_mapping.mapping import ScanData, LearningData, predict_depths
from nf_slam.space_hashing_mapping.mlp_model import MLPModel


def show_points(laser_data_list, c="yellow", s=0.3):
    all_points = []
    for laser_data in laser_data_list:
        all_points.append(laser_data.as_points_in_odometry_frame())
    points = np.concatenate(all_points, axis=0)
    plt.scatter(points[:, 0], points[:, 1], s=s, c=c)
    plt.gca().set_aspect("equal")
    return points


def plot_model_heatmap(map_model: MapModel, bounds, model, grid_shape=(200, 200), angle=0, vmin=None, vmax=None):
    grid_x, grid_y = jnp.meshgrid(jnp.linspace(bounds[0], bounds[1], grid_shape[0]),
                                  jnp.linspace(bounds[2], bounds[3], grid_shape[1]))
    grid_angle = jnp.ones_like(grid_x) * angle
    grid = jnp.stack([grid_x, grid_y, grid_angle], axis=2).reshape(-1, 3)
    obstacle_probabilities = calculate_densities(grid, map_model, model, map_model.hashtable.shape[0])
    obstacle_probabilities = np.array(obstacle_probabilities).reshape(*grid_shape)
    grid = grid.reshape(grid_shape[0], grid_shape[1], 3)
    plt.gca().pcolormesh(grid[:, :, 0], grid[:, :, 1], obstacle_probabilities, cmap='RdBu', shading='auto',
                         vmin=vmin, vmax=vmax)
    plt.gca().set_aspect('equal')


def plot_nf_with_scans(laser_data_list, map_model, mlp_model):
    points = show_points(laser_data_list)
    bounds = (np.min(points[:, 0]) - 1, np.max(points[:, 0]) + 1, np.min(points[:, 1]) - 1, np.max(points[:, 1]) + 1)
    plot_model_heatmap(map_model, bounds, mlp_model)


def plot_positions(position_history):
    positions = Position2D.from_array(position_history).as_vec()
    plt.plot(positions[:, 0], positions[:, 1])
    plt.quiver(positions[:, 0], positions[:, 1], np.cos(positions[:, 2]), np.sin(positions[:, 2]), scale=25)


def plot_position(positions):
    plt.quiver(positions[None, 0], positions[None, 1], np.cos(positions[None, 2]),
               np.sin(positions[None, 2]), scale=10, color="red")


def plot_optimization_result(laser_data: LaserData, model_config: MapModelConfig, map_model: MapModel, model: MLPModel,
                             optimized_position, position_history):
    scan_data = ScanData.from_laser_data(laser_data)
    plt.figure(dpi=200)
    learning_data = LearningData(uniform=jnp.ones((len(scan_data.depths), model_config.bins_count)) * 0.5)
    predicted_depths = predict_depths(map_model, optimized_position, scan_data, learning_data, model_config, model)
    new_points = np.stack([np.cos(scan_data.angles) * predicted_depths, np.sin(scan_data.angles) * predicted_depths],
                          axis=1)
    new_points = Position2D.from_vec(np.array(optimized_position)).apply(new_points)

    ground_truth_points = np.stack(
        [np.cos(scan_data.angles) * scan_data.depths, np.sin(scan_data.angles) * scan_data.depths],
        axis=1)
    ground_truth_points = Position2D.from_vec(np.array(optimized_position)).apply(ground_truth_points)
    show_points([laser_data], s=3, c="orange")
    plt.scatter(new_points[:, 0], new_points[:, 1], s=3)
    plt.scatter(ground_truth_points[:, 0], ground_truth_points[:, 1], s=3)
    plot_positions(position_history)
    plot_position(laser_data.odometry_position.as_vec())
    plt.gca().set_aspect('equal')


def plot_reconstructed_result(laser_data, model_config, mlp_model, map_model, s=3, c="blue", position=None):
    scan_data = ScanData.from_laser_data(laser_data)
    if position is None:
        position = laser_data.odometry_position
    learning_data = LearningData(uniform=jnp.ones((len(scan_data.depths), model_config.bins_count)) * 0.5)
    predicted_depths = predict_depths(map_model, jnp.array(position.as_vec()),
                                      scan_data, learning_data, model_config, mlp_model)
    points = np.stack([np.cos(scan_data.angles) * predicted_depths, np.sin(scan_data.angles) * predicted_depths],
                      axis=1)
    points = position.apply(points)
    plt.scatter(points[:, 0], points[:, 1], s=s, c=c)
