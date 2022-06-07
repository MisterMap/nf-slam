from dataclasses import dataclass

import numpy as np

from nf_slam.position_2d import Position2D


@dataclass(frozen=True)
class LaserDataConfig:
    maximal_distance: float


@dataclass(frozen=True)
class LaserData(object):
    ranges: np.array
    angles: np.array
    timestamp: float
    odometry_position: Position2D
    parameters: LaserDataConfig

    def as_points(self):
        mask = self.ranges < self.parameters.maximal_distance
        x = self.ranges[mask] * np.cos(self.angles[mask])
        y = self.ranges[mask] * np.sin(self.angles[mask])
        return np.stack([x, y], axis=1)

    def as_points_in_odometry_frame(self):
        points = self.as_points()
        return self.odometry_position.apply(points)
