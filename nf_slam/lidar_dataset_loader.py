import dataclasses

import numpy as np

from nf_slam.laser_data import LaserData, LaserDataConfig
from nf_slam.position_2d import Position2D


@dataclasses.dataclass
class LidarDatasetLoaderConfig:
    maximal_token_length: int  # 180
    minimal_angle: float  # 90.
    maximal_angle: float  # 90.
    maximal_distance: float


class LidarDatasetLoader:
    def __init__(self, parameters: LidarDatasetLoaderConfig):
        self._parameters = parameters

    def load(self, datafile):
        result = []
        with open(datafile, "r") as fd:
            for line in fd.readlines():
                line = line.strip()
                tokens = line.split(' ')
                if len(tokens) <= self._parameters.maximal_token_length:
                    continue
                num_scans = int(tokens[1])
                ranges = np.array([float(r) for r in tokens[2:(num_scans + 2)]])
                angles = np.linspace(-self._parameters.minimal_angle / 180.0 * np.pi,
                                     self._parameters.maximal_angle / 180.0 * np.pi, num_scans + 1)[:-1]
                timestamp = float(tokens[(num_scans + 8)])
                odom_x, odom_y, odom_theta = [float(r) for r in tokens[(num_scans + 2):(num_scans + 5)]]
                position = Position2D(odom_x, odom_y, odom_theta)
                result.append(LaserData(ranges=ranges, angles=angles, timestamp=timestamp, odometry_position=position,
                                        parameters=LaserDataConfig(self._parameters.maximal_distance)))
        return result
