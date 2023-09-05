from numba import cuda

# Disable numba warninngs
import logging
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.ERROR) # only show error


# Numba functions to work in the GPU
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