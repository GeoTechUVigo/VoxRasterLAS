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


import numpy as np
from copy import deepcopy
from numba import cuda
from .utils import utils_cuda
from ismember import ismember
import laspy
from .utils import utils
import numbers


class Raster(object):
    def __init__(self, point_cloud: laspy.lasdata.LasData, pixel_size: float, occupation: bool=False, mean_dimensions: list=[], mode_dimensions: list=[], var_dimensions: list=[], 
                 max_dimensions: list=[], min_dimensions: list=[], mean_suffix: str='', mode_suffix: str='', var_suffix: str='_var', max_suffix: str='_max', min_suffix: str='_min',
                 adjust_pixel_size: bool=False, raster_size: np.array = None, centre: np.array = None, down_mode: bool=False,
                 numba=True, blocks = 128, threads_per_block = 64):
        """
        Raster constructor. This is a class to work with rasterised 2D point clouds in XY plane.

        A Raster object is generated from the laspy.lasdata.LasData object, specifying a pixel_size and which las dimensions you need to calculate and how.
        The dimension computed with variance and max are set with the name $DIMENSION_var and $DIMENSION_max.

        :param point_cloud: point cloud to be vonxelised.
        :param pixel_size: size of pixel. It can numeric or 3x1 array.
        :param occupation: if True, it returns a boolean raster with True in the occupated pixels. Saved in property occupation. [Default: False]
        :param mean_dimensions: list of dimensions in point_cloud to be calcualted by averaging. Saved in property dimension_name+mean_suffix.[Defaul: []]
        :param mode_dimensions: list of dimensions in point_cloud to be calcualted by mode. Saved in property dimension_name+mode_suffix. [Defaul: []]
        :param var_dimensions: list of dimensions in point_cloud to be calcualted by variance. Saved in property dimension_name+var_suffix. [Defaul: []]
        :param max_dimensions: list of dimensions in point_cloud to be calcualted by max(). Saved in property dimension_name+max_suffix. [Defaul: []]
        :param min_dimensions: list of dimensions in point_cloud to be calcualted by min(). Saved in property dimension_name+min_suffix. [Defaul: []]
        :param mean_suffix: string appended to the name of the dimensions calculated by averaging. [Default: '']
        :param mode_suffix: string appended to the name of the dimensions calculated by mode. [Default: '']
        :param var_suffix: string appended to the name of the dimensions calculated by variance. [Default: '_var']
        :param max_suffix: string appended to the name of the dimensions calculated by maximum. [Default: '_max']
        :param min_suffix: string appended to the name of the dimensions calculated by minimum. [Default: '_min']
        :param adjust_pixel_size: if True, pixel_size is recalcualted homogeneously spacing the whole raster range. [Default: False]
        :param raster_size: dimensions in pixels of the output image.
        :param centre: centre of point_cloud used to determine which points are inside the raster_size. If it is not specified the centre of point_cloud is used.
        :param down_mode: If True, the dimensions specifid in mean_dimensions are obtained by choosen the lower point in Z in each pixel. The other dimensions are not computed. Not available with numba. [Default: False]
        :param numba: if True, the computations are perfomed using numba library. Note that the numba specifications are required to use it. [Default: True]
        :param blocks: blocks input for numba library. [Default: 128]
        :param threads_per_block: threads per block input for numba library. [Default: 64]
        """

        if down_mode: numba=False

        if  isinstance(pixel_size,numbers.Number): pixel_size = [pixel_size, pixel_size]

        # Limits of the raster
        if raster_size is None:
            min_x, min_y, _ = point_cloud.header.mins
            max_x, max_y, _ = point_cloud.header.maxs

            # Calculate number of pixels
            n_x = np.ceil((max_x - min_x) / pixel_size[0]).astype('uint')
            n_y = np.ceil((max_y - min_y) / pixel_size[1]).astype('uint')

            # New size of voxel
            step_x = (max_x - min_x) / n_x if adjust_pixel_size else pixel_size[0]
            step_y = (max_y - min_y) / n_y if adjust_pixel_size else pixel_size[1]

        else:
            # Dimensions given
            n_x = raster_size[0].astype('uint')
            n_y = raster_size[1].astype('uint')

            # New size of voxel
            step_x = pixel_size[0]
            step_y = pixel_size[1]

            # Limits of the raster in coords
            centre = np.mean(point_cloud.xyz, axis=0) if centre is None else centre

            min_x = centre[0] - raster_size[0]/2 * step_x
            max_x = centre[0] + raster_size[0]/2 * step_x
            min_y = centre[1] - raster_size[1]/2 * step_y
            max_y = centre[1] + raster_size[1]/2 * step_y

            # Select points inside the limits
            in_size = np.all((point_cloud.xyz[:, 0] > min_x, point_cloud.xyz[:, 0] < max_x, point_cloud.xyz[:, 1] > min_y, point_cloud.xyz[:, 1] < max_y), axis=0)
            point_cloud = point_cloud[in_size]

        setattr(self, 'pixel_size', np.array((step_x, step_y)))

        # Calculate coordinates of each point
        dim_x = np.trunc((point_cloud.xyz[:, 0] - min_x) / step_x).astype('uint')
        dim_y = np.trunc((point_cloud.xyz[:, 1] - min_y) / step_y).astype('uint')

        # Move coordinates out of range to the last coordinate in each dimension
        dim_x[dim_x == n_x] = n_x - 1
        dim_y[dim_y == n_y] = n_y - 1

        if numba:
                
            # Define lineal index for each point
            dim = (dim_x + (dim_y * n_x)).astype('uint')
            
            # Get an array with the occupied voxel indexes sorted, the order of the idx in the point cloud and number of times for each idx
            idx, order, n_points = np.unique(dim, return_inverse=True, return_counts=True)

            n_pixels = n_points.shape[0]

            # Load to device order and n_points
            d_order = cuda.to_device(order)
            d_npoints = cuda.to_device(n_points)

            for property_name in mean_dimensions:
                # Load property in device
                d_property = np.asarray(getattr(point_cloud, property_name))
                dtype = d_property.dtype
                if len(d_property.shape)==1: d_property = d_property.reshape(-1,1)
                d_property = cuda.to_device(np.ascontiguousarray(d_property.astype(np.float64)))

                d_aux = cuda.to_device(np.ascontiguousarray(np.zeros(shape=(n_pixels, d_property.shape[1]), dtype=np.float64)))
                d_out = cuda.to_device(np.ascontiguousarray(np.zeros(shape=(n_pixels, d_property.shape[1]), dtype=np.float64)))

                # Mean
                utils_cuda.sum[blocks, threads_per_block](d_property, d_order, d_aux)
                cuda.synchronize
                utils_cuda.mean_sum[blocks, threads_per_block](d_aux, d_npoints, d_out)
                cuda.synchronize
        
                # Reshape
                out = np.zeros((n_y*n_x, d_property.shape[1]), dtype=dtype)
                out[idx, :] =  d_out.copy_to_host()
                out = out.reshape((n_y, n_x, -1)).astype(dtype)
                out = np.transpose(out, axes=[1,0,2])

                # Set
                setattr(self, property_name + mean_suffix, out)

            for property_name in mode_dimensions:
                # Load property in device
                d_property = np.asarray(getattr(point_cloud, property_name))
                dtype = d_property.dtype
                dmax = d_property.max()
                if len(d_property.shape)==1: d_property = d_property.reshape(-1,1)

                d_property = cuda.to_device(np.ascontiguousarray(d_property.astype(np.int32)))

                d_aux =cuda.to_device(np.ascontiguousarray(np.zeros(shape=(n_pixels, dmax+1), dtype=np.int32)))
                d_aux_2 = cuda.to_device(np.ascontiguousarray(np.zeros(shape=(n_pixels), dtype=np.int32)))
                d_out = cuda.to_device(np.ascontiguousarray(np.zeros(shape=(n_pixels, d_property.shape[1]), dtype=np.int32)))

                # mode
                utils_cuda.sum_class[blocks, threads_per_block](d_property, d_order, d_aux)
                cuda.synchronize
                utils_cuda.mode_column[blocks, threads_per_block](d_aux, d_aux_2, d_out)
                cuda.synchronize

                # Reshape
                out = np.zeros((n_y*n_x, d_out.shape[1]), dtype=dtype)
                out[idx, :] =  d_out.copy_to_host()
                out = out.reshape((n_y, n_x, -1))
                out = np.transpose(out, axes=[1,0,2])

                # Set property
                setattr(self, property_name + mode_suffix, out)


            for property_name in max_dimensions:
                # Load property in device
                d_property = np.asarray(getattr(point_cloud, property_name))
                dtype = d_property.dtype
                if len(d_property.shape)==1: d_property = d_property.reshape(-1,1)
                d_property = cuda.to_device(np.ascontiguousarray(d_property.astype(np.float64)))
                
                d_out = cuda.to_device(np.ascontiguousarray(np.zeros(shape=(n_pixels, d_property.shape[1]), dtype=np.float64)))

                # max
                utils_cuda.max[blocks, threads_per_block](d_property, d_order, d_out)
                cuda.synchronize
        
                # Reshape
                out = np.zeros((n_y*n_x, d_property.shape[1]), dtype=dtype)
                out[idx, :] =  d_out.copy_to_host()
                out = out.reshape((n_y, n_x, -1)).astype(dtype)
                out = np.transpose(out, axes=[1,0,2])

                # Set
                setattr(self, property_name + max_suffix, out)

            for property_name in min_dimensions:
                # Load property in device
                d_property = np.asarray(getattr(point_cloud, property_name))
                dtype = d_property.dtype
                if len(d_property.shape)==1: d_property = d_property.reshape(-1,1)

                d_out = cuda.to_device(np.ascontiguousarray(d_property.max()*np.ones(shape=(n_pixels, d_property.shape[1]), dtype=np.float64)))
                d_property = cuda.to_device(np.ascontiguousarray(d_property.astype(np.float64)))
                
                # max
                utils_cuda.min[blocks, threads_per_block](d_property, d_order, d_out)
                cuda.synchronize
        
                # Reshape
                out = np.zeros((n_y*n_x, d_property.shape[1]), dtype=dtype)
                out[idx, :] =  d_out.copy_to_host()
                out = out.reshape((n_y, n_x, -1)).astype(dtype)
                out = np.transpose(out, axes=[1,0,2])

                # Set
                setattr(self, property_name + min_suffix, out)

            for property_name in var_dimensions:
                # Load property in device
                d_property = np.asarray(getattr(point_cloud, property_name))
                dtype = d_property.dtype
                if len(d_property.shape)==1: d_property = d_property.reshape(-1,1)
                d_property = cuda.to_device(np.ascontiguousarray(d_property.astype(np.float64)))

                d_aux = cuda.to_device(np.ascontiguousarray(np.zeros(shape=(n_pixels, d_property.shape[1]), dtype=np.float64)))
                d_mean = cuda.to_device(np.ascontiguousarray(np.zeros(shape=(n_pixels, d_property.shape[1]), dtype=np.float64)))
                d_aux_2 = cuda.to_device(np.ascontiguousarray(np.zeros(shape=(n_pixels, d_property.shape[1]), dtype=np.float64)))
                d_out = cuda.to_device(np.ascontiguousarray(np.zeros(shape=(n_pixels, d_property.shape[1]), dtype=np.float64)))

                # Mean
                utils_cuda.sum[blocks, threads_per_block](d_property, d_order, d_aux)
                cuda.synchronize
                utils_cuda.mean_sum[blocks, threads_per_block](d_aux, d_npoints, d_mean)
                cuda.synchronize

                # Var
                utils_cuda.squared_dist_sum[blocks, threads_per_block](d_property, d_order, d_mean, d_aux_2)
                cuda.synchronize
                utils_cuda.mean_sum[blocks, threads_per_block](d_aux_2, d_npoints, d_out)
                cuda.synchronize
        
                # Reshape
                out = np.zeros((n_y*n_x, d_property.shape[1]), dtype=dtype)
                out[idx, :] =  d_out.copy_to_host()
                out = out.reshape((n_y, n_x, -1)).astype(dtype)
                out = np.transpose(out, axes=[1,0,2])

                # Set
                setattr(self, property_name + var_suffix, out)

            if occupation:
                out = np.zeros((n_y*n_x, 1), dtype='bool')
                out[idx, :] =  True
                out = out.reshape((n_y, n_x, -1))
                out = np.transpose(out, axes=[1,0,2])
                setattr(self, 'occupation', out)


            setattr(self, 'parent_idx', dim)

        else:

            # Define lineal index for each point
            idx_orig = (dim_x + (dim_y * n_x)).astype('uint')

            # Order based in the idx values
            shift_index = np.argsort(idx_orig)
            idx = idx_orig[shift_index]

            # Get an array with the occupied voxels and number of times for each one of them
            occ_pixels, n_index, n_points = np.unique(idx, return_index=True, return_counts=True)
            occ_pixels = np.asarray([occ_pixels, n_points]).transpose().astype('int')

            # Dimensions x and y sorted and unique
            dim = np.vstack((dim_x, dim_y)).transpose()
            dim = dim[shift_index, :]
            dim = dim[n_index, :]

            # Initialising the parameters of each pixel
            n_pixels = occ_pixels.shape[0]

            mean = False
            if len(mean_dimensions) != 0:
                mean = True
                if not mean_dimensions is np.ndarray: mean_dimensions = np.asarray(mean_dimensions)
                data_mean, rt_mean, data_size_mean, dtype_mean = self.__prepare_properties(mean_dimensions, point_cloud, n_x, n_y, shift_index)

            mode = False
            if len(mode_dimensions) != 0:
                mode = True
                if not mode_dimensions is np.ndarray: mode_dimensions = np.asarray(mode_dimensions)
                data_mode, rt_mode, data_size_mode, dtype_mode = self.__prepare_properties(mode_dimensions, point_cloud, n_x, n_y, shift_index)
                data_mode = data_mode.astype('int')
                rt_mode = rt_mode.astype('int')

            var = False
            if len(var_dimensions) != 0:
                var = True
                if not var_dimensions is np.ndarray: var_dimensions = np.asarray(var_dimensions)
                data_var, rt_var, data_size_var, dtype_var = self.__prepare_properties(var_dimensions, point_cloud, n_x, n_y, shift_index)

            max = False
            if len(max_dimensions) != 0:
                max = True
                if not max_dimensions is np.ndarray: max_dimensions = np.asarray(max_dimensions)
                data_max, rt_max, data_size_max, dtype_max = self.__prepare_properties(max_dimensions, point_cloud, n_x, n_y, shift_index)

            min = False
            if len(min_dimensions) != 0:
                min = True
                if not min_dimensions is np.ndarray: min_dimensions = np.asarray(min_dimensions)
                data_min, rt_min, data_size_min, dtype_min = self.__prepare_properties(min_dimensions, point_cloud, n_x, n_y, shift_index)

            if occupation:
                rt_occupation = np.zeros((n_x, n_y, 1), dtype='bool')

            # Calculating the parameters of each pixel
            if not down_mode:
                last_point = 0
                for i in range(n_pixels):
                    next_point = last_point + occ_pixels[i, 1]
                    points = slice(last_point, next_point, 1)

                    if mean:
                        rt_mean[dim[i, 0], dim[i, 1], :] = np.mean(data_mean[points, :], axis=0)
                    if max:
                        rt_max[dim[i, 0], dim[i, 1], :] = np.max(data_max[points, :], axis=0)
                    if min:
                        rt_min[dim[i, 0], dim[i, 1], :] = np.min(data_min[points, :], axis=0)
                    if mode:
                        for j in range(data_mode.shape[1]):
                            rt_mode[dim[i, 0], dim[i, 1], j] = np.bincount(data_mode[points,j]).argmax()
                    if var:
                        for j in range(data_var.shape[1]):
                            rt_var[dim[i, 0], dim[i, 1], j] = utils.var(data_var[points,j])

                    if occupation:
                        rt_occupation[dim[i, 0], dim[i, 1]] = True

                    last_point = next_point

            # Calculating the parameters of each pixel as the point with the lowest Z.
            else:
                height_sort = getattr(point_cloud, 'z')[shift_index]
                last_point = 0
                for i in range(n_pixels):
                    next_point = last_point + occ_pixels[i, 1]
                    points = slice(last_point, next_point, 1)

                    # Select the point with the lower Z
                    point = np.arange(last_point, next_point)[np.argmin(height_sort[points], axis=0)]

                    if mean:
                        rt_mean[dim[i, 0], dim[i, 1], :] = data_mean[point, :]

                    if occupation:
                        rt_occupation[dim[i, 0], dim[i, 1]] = True

                    last_point = next_point

            # Set properties
            for i in range(len(mean_dimensions)):
                if i != 0:
                    setattr(self, mean_dimensions[i] + mean_suffix, rt_mean[..., data_size_mean[i-1]:data_size_mean[i]].astype(dtype_mean[i]))
                else:
                    setattr(self, mean_dimensions[i] + mean_suffix, rt_mean[..., 0:data_size_mean[i]].astype(dtype_mean[i]))

            for i in range(len(mode_dimensions)):
                if i != 0:
                    setattr(self, mode_dimensions[i] + mode_suffix, rt_mode[..., data_size_mode[i-1]:data_size_mode[i]].astype(dtype_mode[i]))
                else:
                    setattr(self, mode_dimensions[i] + mode_suffix, rt_mode[..., 0:data_size_mode[i]].astype(dtype_mode[i]))

            for i in range(len(var_dimensions)):
                if i != 0:
                    setattr(self, var_dimensions[i] + var_suffix, rt_var[..., data_size_var[i-1]:data_size_var[i]].astype(dtype_var[i]))
                else:
                    setattr(self, var_dimensions[i] + var_suffix, rt_var[..., 0:data_size_var[i]].astype(dtype_var[i]))

            for i in range(len(max_dimensions)):
                if i != 0:
                    setattr(self, max_dimensions[i] + max_suffix, rt_max[..., data_size_max[i-1]:data_size_max[i]].astype(dtype_max[i]))
                else:
                    setattr(self, max_dimensions[i] + max_suffix, rt_max[..., 0:data_size_max[i]].astype(dtype_max[i]))

            for i in range(len(min_dimensions)):
                if i != 0:
                    setattr(self, min_dimensions[i] + min_suffix, rt_min[..., data_size_min[i-1]:data_size_min[i]].astype(dtype_min[i]))
                else:
                    setattr(self, min_dimensions[i] + min_suffix, rt_min[..., 0:data_size_min[i]].astype(dtype_min[i]))

            setattr(self, 'parent_idx', idx_orig)

            if occupation:
                setattr(self, 'occupation', rt_occupation)
    # ==================================================================================================================
    # Properties

    @property
    def pixel_size(self):
        """
        Getter of pixel_size.

        :return: size of pixel as float.
        """
        return self.__pixel_size

    @pixel_size.setter
    def pixel_size(self, pixel_size):
        """
        Setter of pixel_size.

        :param pixel_size: size of pixel as float.
        :return: None.
        """

        self.__pixel_size = pixel_size

    @property
    def parent_idx(self):
        """
        Getter of parent_idx.

        :return: Mx1 numpy.array. M is the number of points in the original point cloud. The value of each point is the index of the pixelised image point cloud to which it belongs.
        """
        return self.__parent_idx

    @parent_idx.setter
    def parent_idx(self, parent_idx):
        """
        Setter of parent_idx.

        :param parent_idx: Mx1 numpy.array. M is the number of points in the original point cloud. The value of each point is the index of the pixelised image point cloud to which it belongs.
        :return: None.
        """

        self.__parent_idx = parent_idx

    # ==================================================================================================================
    # Methods


    def get_parent_idx(self, indexes):
        '''
        Method that returns a boolean array with True in those points in the original point cloud that are in any voxel specified in indexes.

        :param indexes: boolean or int array. If it is a boolean array, it can be MxN or a 1 column array.
        :return: Mx1 as bool numpy.ndarray.
        '''

        if not indexes is np.ndarray: indexes = np.asarray(indexes)
        if indexes.dtype == 'bool': indexes = np.where(indexes.reshape(-1))[0]

        iloc, _ = ismember(self.parent_idx, indexes)

        return iloc
    

    def continuous_polyline(self, max_distance, property, value=True):
        """
        Function to make the Raster.property with the values speficied image continuous in X and in Y.
        It calculate the coordinates XY between consecutive pixels, based on Raster.parent_idx, interpolating them
        computing the line equation that merge both points.

        To make it continuous in X and in Y, coordinates are evaluated for each X between both pixels and also for each
        Y between them.

        If the distance in pixels between 2 consecutive points is greater than max_distance they are not merged.

        :param max_distance: maximum distance in pixel to merge 2 consecutive points.
        :param property: property that is going to be made continuous.
        :param value: value of the element in the property.
        :return: boolean np.ndarray MxN. True values in the original ones plus the interpolated ones.
        """
        try:
            continuous_raster = deepcopy(getattr(self, property))
        except:
            raise ValueError('{} is not a Raster property'.format(property))

        order = np.where(continuous_raster == value)
        # Indexes of the points in axis_sec in each pixel.
        axis_sec_idx = self.parent_idx[order[0], order[1]]
        # Array that sort this indexes.
        axis_sec_idx = np.argsort(axis_sec_idx)
        # Sort order_pixels in such a way that the pixels contains the axis_sec points by order.
        order[0][:] = np.take_along_axis(order[0], axis_sec_idx, axis=0)
        order[1][:] = np.take_along_axis(order[1], axis_sec_idx, axis=0)

        # Make Raster.property with values = value continuous.
        for i in range(len(order[0]) - 1):
            # Coords
            first_x = order[0][i]
            first_y = order[1][i]
            last_x = order[0][i + 1]
            last_y = order[1][i + 1]

            # If the distance between these 2 points is greater than a threshold, they are not merge.
            if np.linalg.norm(np.array([first_x, first_y]) - np.array([last_x, last_y])) > max_distance:
                continue

            # If the line is neither vertical nor horizontal.
            if first_x != last_x and first_y != last_y:

                # Line parameters. y = m * x + n
                m = (last_y - first_y) / (last_x - first_x)
                n = - m * first_x + first_y

                # Line values.
                # A value for each X. Make the polyline continuous in X. y = m * x + n
                step = 1 if first_x < last_x else -1

                x = np.arange(first_x + step, last_x, step=step)
                y = np.round(m * x + n).astype('int')

                continuous_raster[x, y] = value

                # A value for each Y. Make the polyline continuous in Y. x = (y - n) / m
                step = 1 if first_y < last_y else -1

                y = np.arange(first_y + step, last_y, step=step)
                x = np.round((y - n) / m).astype('int')

                continuous_raster[x, y] = value

            # If the line is vertical.
            elif first_x == last_x:
                # Vertical Y.
                step = 1 if first_y < last_y else -1
                y = np.arange(first_y + step, last_y, step=step)
                # Constant X.
                x = np.zeros(y.shape, dtype='int')
                x[:] = first_x

                continuous_raster[x, y] = value

            # If the line is horizontal.
            else:
                # Horizontal X.
                step = 1 if first_x < last_x else -1
                x = np.arange(first_x + step, last_x, step=step)
                # Constant X.
                y = np.zeros(x.shape, dtype='int')
                y[:] = first_y

                continuous_raster[x, y] = value

        return continuous_raster
    

    @staticmethod
    def __prepare_properties(dimensions, point_cloud, n_x, n_y, shift_index):
        '''
        Method for generate the arrays to voxelise the data.
        '''
        data_size = np.zeros(len(dimensions), dtype='uint')
        dtype = np.zeros(len(dimensions), dtype='object')

        for i in range(len(dimensions)):
            d_property = np.asarray(getattr(point_cloud, dimensions[i]))

            if len(d_property.shape)==1: 
                d_property = d_property.reshape(-1,1)

            data_size[i] = d_property.shape[1]
            dtype[i] = d_property.dtype

        data_size = np.cumsum(data_size)
        

        # Initialising the matrix with the PointClouds properties sorted that are calculated with mean() and the matrix where the result will be saved
        data = np.zeros((point_cloud.xyz.shape[0], data_size[-1]))
        rt_data = np.zeros((n_x, n_y, data_size[-1]))

        for i in range(len(dimensions)):
            d_property = np.asarray(getattr(point_cloud, dimensions[i]))[shift_index]
            if len(d_property.shape)==1: d_property = d_property.reshape(-1,1)

            if i != 0:
                data[:, data_size[i-1]:data_size[i]] = d_property
            else:
                data[:, 0:data_size[i]] = d_property

        return data, rt_data, data_size, dtype    