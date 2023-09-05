"""
Copyright (C) 2023 GeoTECH Group <geotech@uvigo.gal>
Copyright (C) 2023 Daniel Lamas Novoa <daniel.lamas.novoa@uvigo.gal>
This file is part of the program Automated instance and semantic 
segmentation of point clouds of large metallic truss bridges with modelling
purposes.
The program is free software: you can redistribute it and/or modify it 
under the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or any later version.
The program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for 
more details.
You should have received a copy of the GNU General Public License along 
with the program in COPYING. If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from os import listdir
import laspy


def clouds_in_range(cloud_path: str, trajectory, max_distance: float, 
                    verbose: bool = False):
    """
    Function to calculate for all clouds in cloud_path, which trajectory points are related to that cloud
    In other words, which trajectory points are inside of its limits XY, or inside in X and closer than max_distance
    to its Y limits, or vise versa, or closer than max_distance to its min_x and min_y or max_y or vise versa.

    :param cloud_path: folder with the point clouds.
    :param trajectory: trajectory points with XYZ in the first 3 columns.
    :param max_distance: max distance between a trajectory point and a point cloud considered.
    :return: boolean numpy matrix nº trajectory points x nº clouds. True if that points is closer than max_distance to the
    cloud. Points clouds are sorted alphabetically.
    """

    point_cloud_names = listdir(cloud_path)
    point_cloud_names.sort() # list of cloud's names sorted
    output = np.zeros((trajectory.shape[0], len(point_cloud_names)), dtype=bool)  # boolean matrix nºpoints traj x nº clouds

    # For all clouds
    for i in range(len(point_cloud_names)):

        point_cloud = laspy.read(cloud_path + '/' + point_cloud_names[i])  # Loading just the header

        # traj point inside point cloud limits or
        # inside in X and closer than max_distance to Y limit (either max or min) or
        # inside in Y and closer than max_distance to X limit (either max or min) or
        # Closer than max_distance to any X limits and to any Y limits (close to point cloud squares)
        in_range = np.logical_or.reduce((
            np.logical_and(np.all(point_cloud.header.min[0:2] < trajectory[:, 0:2], axis=1),
                           np.all(point_cloud.header.max[0:2] > trajectory[:, 0:2],
                                  axis=1)),
            np.logical_and(np.logical_and(point_cloud.header.min[0] < trajectory[:, 0],
                                          point_cloud.header.max[0] > trajectory[:, 0]),
                           np.logical_or(np.abs(
                               point_cloud.header.min[1] - trajectory[:, 1]) < max_distance,
                                         np.abs(point_cloud.header.max[1] - trajectory[:,
                                                                            1]) < max_distance)),
            np.logical_and(np.logical_and(point_cloud.header.min[1] < trajectory[:, 1],
                                          point_cloud.header.max[1] > trajectory[:, 1]),
                           np.logical_or(np.abs(
                               point_cloud.header.min[0] - trajectory[:, 0]) < max_distance,
                                         np.abs(point_cloud.header.max[0] - trajectory[:,
                                                                            0]) < max_distance)),
            np.logical_and(np.logical_or(
                np.abs(point_cloud.header.min[0] - trajectory[:, 0]) < max_distance,
                np.abs(point_cloud.header.max[0] - trajectory[:, 0]) < max_distance),
                np.logical_or(np.abs(
                    point_cloud.header.min[1] - trajectory[:, 1]) < max_distance,
                              np.abs(point_cloud.header.max[1] - trajectory[:,
                                                                 1]) < max_distance))))

        output[in_range, i] = True  # in the column of this point cloud, true in the points of the traj in range
        
        if verbose: print(point_cloud_names[i] + " analysed")
        
    return output