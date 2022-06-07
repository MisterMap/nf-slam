import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt

from nf_slam.space_hashing_mapping.jax_math import calculate_densities
from nf_slam.space_hashing_mapping.map_model import MapModel


def show_points(laser_data_list, c="yellow", s=0.3):
    all_points = []
    for laser_data in laser_data_list:
        all_points.append(laser_data.as_points_in_odometry_frame())
    points = np.concatenate(all_points, axis=0)
    plt.scatter(points[:, 0], points[:, 1], s=s, c=c)
    plt.gca().set_aspect("equal")
    return points


def plot_model_heatmap(map_model: MapModel, bounds, model):
    grid_shape = (200, 200)
    grid_x, grid_y = jnp.meshgrid(jnp.linspace(bounds[0], bounds[1], grid_shape[0]),
                                  jnp.linspace(bounds[2], bounds[3], grid_shape[1]))
    grid = jnp.stack([grid_x, grid_y], axis=2).reshape(-1, 2)
    obstacle_probabilities = calculate_densities(grid, map_model, model, map_model.hashtable.shape[0])
    obstacle_probabilities = np.array(obstacle_probabilities).reshape(*grid_shape)
    grid = grid.reshape(grid_shape[0], grid_shape[1], 2)
    plt.gca().pcolormesh(grid[:, :, 0], grid[:, :, 1], obstacle_probabilities, cmap='RdBu', shading='auto',
                         vmin=None, vmax=None)
    plt.gca().set_aspect('equal')


def plot_nf_with_scans(laser_data_list, map_model, mlp_model):
    points = show_points(laser_data_list)
    bounds = (np.min(points[:, 0]) - 1, np.max(points[:, 0]) + 1, np.min(points[:, 1]) - 1, np.max(points[:, 1]) + 1)
    plot_model_heatmap(map_model, bounds, mlp_model)
