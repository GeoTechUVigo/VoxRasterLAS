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


from numba import cuda

# Disable numba warninngs
import logging
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.ERROR) # only show error


# Numba functions to work in the GPU
@cuda.jit
def mean(input, index, n_points, output):
    x, y = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)
    
    for j in range(y, input.shape[1], stride_y):
        for i in range(x, input.shape[0],stride_x):
            cuda.atomic.add(output, (index[i], j), input[i, j] / n_points[index[i]])


@cuda.jit
def mean_sum(input, n_points, output):
    x, y = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)
    
    for j in range(y, input.shape[1], stride_y):
        for i in range(x, input.shape[0],stride_x):
            output[i, j] = input[i, j] / n_points[i]


@cuda.jit
def sum(input, index, output):
    x, y = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)
    
    for j in range(y, input.shape[1], stride_y):
        for i in range(x, input.shape[0],stride_x):
            cuda.atomic.add(output, (index[i], j), input[i, j])


@cuda.jit
def max(input, index, output):
    x, y = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)
    
    for j in range(y, input.shape[1], stride_y):
        for i in range(x, input.shape[0],stride_x):
            cuda.atomic.max(output, (index[i], j), input[i, j])

@cuda.jit
def min(input, index, output):
    x, y = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)
    
    for j in range(y, input.shape[1], stride_y):
        for i in range(x, input.shape[0],stride_x):
            cuda.atomic.min(output, (index[i], j), input[i, j])


@cuda.jit
def squared_dist_sum(input, index, mean, output):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    for i in range(idx, input.shape[0], stride):
        cuda.atomic.add(output, (index[i], 0), (input[i, 0] - mean[index[i], 0])**2)


@cuda.jit
def covar_dist_sum(input_1, input_2, index, mean_1, mean_2, output):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    for i in range(idx, input_1.shape[0], stride):
        cuda.atomic.add(output, (index[i], 0), (input_1[i, 0] - mean_1[index[i], 0]) * (input_2[i, 0] - mean_2[index[i], 1]))


@cuda.jit
def sum_class(input, index, output):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    for i in range(idx, input.shape[0], stride):
        cuda.atomic.add(output, (index[i], input[i, 0]), 1)


@cuda.jit
def mode_column(input, n_times, output):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(idx, input.shape[0],stride):
        for j in range(input.shape[1]):
            if input[i, j] > n_times[i]:               
                n_times[i] = input[i,j]
                output[i,0] = j