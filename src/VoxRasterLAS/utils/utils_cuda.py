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
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(idx, input.shape[0],stride):
        cuda.atomic.add(output, index[i], input[i] / n_points[index[i]])


@cuda.jit
def max(input, index, output):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(idx, input.shape[0],stride):
        cuda.atomic.max(output, index[i], input[i])

@cuda.jit
def min(input, index, output):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)  
    for i in range(idx, input.shape[0],stride):
        cuda.atomic.min(output, index[i], input[i])


@cuda.jit
def var(input, index, mean, n_points, output):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)  
    for i in range(idx, input.shape[0], stride):
        cuda.atomic.add(output, index[i], ((input[i] - mean[index[i]])**2) / n_points[index[i]])


@cuda.jit
def cov(input_1, input_2, index, mean_1, mean_2, n_points, output):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(idx, input_1.shape[0], stride):
        cuda.atomic.add(output, index[i], ((input_1[i] - mean_1[index[i]]) * (input_2[i] - mean_2[index[i]])) / n_points[index[i]])


@cuda.jit
def sum_class(input, index, output):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(idx, input.shape[0], stride):
        cuda.atomic.add(output, (index[i], input[i]), 1)


@cuda.jit
def mode_column(input, n_times, output):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(idx, input.shape[0],stride):
        for j in range(input.shape[1]):
            if input[i, j] > n_times[i]:               
                n_times[i] = input[i,j]
                output[i] = j