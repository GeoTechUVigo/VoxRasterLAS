"""
Copyright (C) 2023 GeoTECH Group <geotech@uvigo.gal>
Copyright (C) 2023 Daniel Lamas Novoa <daniel.lamas.novoa@uvigo.gal>
This file is part of VoxRasterLAS.
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


import laspy
from VoxRasterLAS import Voxels
from VoxRasterLAS import Raster
import numpy as np

cloud_path = "data/BaileyTruss_000.las"

# Read point cloud
las = laspy.read(cloud_path)

# Raster
rt = Raster(las, grid=0.01, min_dimensions=['z'], max_dimensions=['z'], numba=True)

# Voxels
vx = Voxels(las, grid=[0.02,0.2,0.2], mean=['xyz'], centroid=['a', 'b', 'c'], var=['z'], random=['x','y', 'z'], random_suffix='_random', var_suffix='_var', neighbours=True, numba=True)

# Neighbours
nb = vx.neighbours

# parent idx
parent_idx = vx.get_parent_idx([4,5,6,7,3,100])

# select
idx = vx.las.z <np.mean(vx.las.z)

# selected
vx_selected = vx[idx]
print(len(vx))
print(len(vx_selected))

#parent point cloud selected
las_select = las[vx.get_parent_idx(idx)]

# Randomly downsampling xyz
vx = Voxels(las, grid=[0.02,0.2,0.2], random=['xyz'])

# Method
from VoxRasterLAS.segmentation import clouds_in_range
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