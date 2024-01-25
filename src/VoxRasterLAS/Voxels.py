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
from ismember import ismember
from .utils import utils_cuda
from .utils import utils
from numba import cuda
import laspy
import copy
import numbers
from numba import cuda


class Voxels(object):
    def __init__(self, point_cloud: laspy.lasdata.LasData=None, voxel_size: float=None, 
                 random: str=[], mean: list=[], mode: list=[], var: list=[], cov: list=[], centroid: str=[], 
                 random_suffix: str ='', mean_suffix: str='', mode_suffix: str='', var_suffix: str ='_var', cov_suffix: str ='_cov',
                 neighbours: bool=False, grid: bool=False, pca_local: bool=False, scale_eigenvalues: bool=False, calc_all: bool = False, adjust_voxel_size: bool=False,
                 numba=True, blocks = 128, threads_per_block = 64):
        """
        Voxels constructor.
        A Voxel object is generated from the laspy.lasdata.LasData object, specifying a voxel_size (voxel size) and which las dimensions you need to calculate and how.
        The voxelised point_cloud is return as laspy.lasdata.LasData in the property las. 
        The dimensions to be computed must be specified in the lists which indicate by which method they are calculated. for each of these lists, 
        the suffix to store the calcualted dimension must be specified. For instance, if var =['z'] and var_suffic='_var', the variance of z is saved as 'z_var'.
        If mean = ['xyz'] and mean_suffix = '', the average of xyz is saved as 'xyz'.
        It is also possible to computed the centroid of the voxels using the argument centroid.
        Note that all dimensions must be 1D array. The only dimension name allowed to call and write several dimensions in a LasData is 'xyz'.

        :param point_cloud: point cloud to be vonxelised.
        :param voxel_size: size of voxel. It can numeric or 3x1 array.
        :param random: list of dimensions in point_cloud to be calcualted by randomly picking 1 point per voxel. [Defaul: []]
        :param mean: list of dimensions in point_cloud to be calcualted by averaging. [Defaul: []]
        :param mode: list of dimensions in point_cloud to be calcualted by mode. [Defaul: []]
        :param var: list of dimensions in point_cloud to be calcualted by variance. [Defaul: []]
        :param cov: list of dimensions in point_cloud to be calcualted by covariance. For insntace: [['x','y'], ['x','intensity']]. Only with numba. [Defaul: []]
        :param centroid: if different of [], it calcualtes the centroid of each voxel and save it with the name specified in this str. Note it must be ['xyz'] or a list with 3 str. [Default: []]
        :param random_suffix: string appended to the name of the dimensions calculated by randomly picking 1 point per voxel. [Default: '']
        :param mean_suffix: string appended to the name of the dimensions calculated by averaging. [Default: '']
        :param mode_suffix: string appended to the name of the dimensions calculated by mode. [Default: '']
        :param var_suffix: string appended to the name of the dimensions calculated by variance. [Default: '_var']
        :param cov_suffix: string appended to the name of the dimensions calculated by covariance. [Default: '_cov']
        :param neighbours: if True, neighbours indexes are calcualted referenced to the array las.xyz. [Default: False]
        :param grid: if True, 3D boolean occupation grid and indexes_grid are computed. Only with numba. [Default: False]
        :param pca_local: if True, pca at voxel level is calculated. [Default: False]
        :param scale_eigenvalues: if True, eigenvalues of pca_local are normalised. [Default: False]
        :param calc_all: if True, all the dimensions in point_cloud are computed by mean but the ones specefied in mode. The rest of the specified dimensions are calcualted with their method. [Default: False]
        :param adjust_voxel_size: if True, voxel_size is recalcualted homogeneously spacing the whole point cloud range. [Default: False]
        :param numba: if True, the computations are perfomed using numba library. Note that the numba specifications are required to use it. [Default: True]
        :param blocks: blocks input for numba library. [Default: 128]
        :param threads_per_block: threads per block input for numba library. [Default: 64]
        """

        # Set all properties as None
        for p in dir(Voxels):
            if isinstance(getattr(Voxels,p),property):
                setattr(self, p, None)
        if point_cloud is None or voxel_size is None:
            return
        
        if  isinstance(voxel_size,numbers.Number): voxel_size = [voxel_size, voxel_size, voxel_size]
        
        # Check lists of properties
        mean  = self.check_list(mean)
        mode = self.check_list(mode)
        var = self.check_list(var)
        cov = self.check_list(cov)
        # ==============================================================================================================
        # Add all the dimensions if calc_all
        if calc_all:
            las_dimension_names = list(point_cloud.point_format.dimension_names)
            # Compute xyz together to speed up the process. Check possible problems with the name of mean properties.
            las_dimension_names.remove('X'), las_dimension_names.remove('Y'), las_dimension_names.remove('Z')
 
            las_dimension_names.append('x')
            las_dimension_names.append('y')
            las_dimension_names.append('z')

            mean = np.asarray(las_dimension_names)
            
            # Remove dimensions which are calculated differently
            aux, _ = ismember(mean, mode)
            mean = mean[~aux]

        # ==============================================================================================================
        # Get max and min values
        min_x, min_y, min_z = point_cloud.header.mins
        max_x, max_y, max_z = point_cloud.header.maxs

        # Calculate number of voxels. It cannot be 0.
        n_x = np.ceil((max_x - min_x) / voxel_size[0]).astype('int') if np.ceil((max_x - min_x) / voxel_size[0]) != 0 else 1
        n_y = np.ceil((max_y - min_y) / voxel_size[1]).astype('int') if np.ceil((max_y - min_y) / voxel_size[1]) != 0 else 1
        n_z = np.ceil((max_z - min_z) / voxel_size[2]).astype('int') if np.ceil((max_z - min_z) / voxel_size[2]) != 0 else 1

        # New size of voxel. It cannot be 0.
        step_x = (max_x - min_x) / n_x if max_x - min_x != 0 and adjust_voxel_size else voxel_size[0]
        step_y = (max_y - min_y) / n_y if max_y - min_y != 0 and adjust_voxel_size else voxel_size[1]
        step_z = (max_z - min_z) / n_z if max_z - min_z != 0 and adjust_voxel_size else voxel_size[2]

        # Calculate coordinates of each point
        dim_x = np.trunc((point_cloud.x - min_x) / step_x).astype('uint')
        dim_y = np.trunc((point_cloud.y - min_y) / step_y).astype('uint')
        dim_z = np.trunc((point_cloud.z - min_z) / step_z).astype('uint')

        # Move coordinates out of range to the last coordinate in each dimension
        dim_x[dim_x == n_x] = n_x - 1
        dim_y[dim_y == n_y] = n_y - 1
        dim_z[dim_z == n_z] = n_z - 1

        # Define lineal index for each point
        idx_ = dim_x + (dim_y * n_x) + (dim_z * n_x * n_y)

        if numba:
            # Get an array with the occupied voxel indexes sorted, the order of the idx in the point cloud and number of times for each idx
            idx, row_idx, order, n_points = np.unique(idx_, return_index=True, return_inverse=True, return_counts=True)

            # Count number of occupied voxels
            n_voxels = idx.shape[0]
        else:
            # Order based in the idx values
            shift_index = np.argsort(idx_)
            idx = idx_[shift_index]

            # Get an array with the occupied voxel indexes and number of times for each one but cumulative
            idx, row_idx, n_points = np.unique(idx, return_index=True, return_counts=True)
            n_points = np.cumsum(n_points)
            occ_voxels = np.asarray([idx, n_points]).transpose().astype('uint')
            # Count number of occupied voxels
            n_voxels = occ_voxels.shape[0]

        # ==============================================================================================================
        # Initialise voxel voxel_size variables
        setattr(self, 'voxel_size', np.array((step_x, step_y, step_z)))
        setattr(self, 'grid_size', np.array((n_x, n_y, n_z)))
        setattr(self, 'min_coords', point_cloud.header.mins)

        if grid:
            setattr(self, 'indexes_grid', self.compute_indexes_xyz(idx))

            grid = np.zeros((self.grid_size[0],self.grid_size[1],self.grid_size[2]), dtype=np.bool_)
            grid[self.indexes_grid[:,0], self.indexes_grid[:,1], self.indexes_grid[:,2]] = True
            setattr(self, 'grid', grid)
            
        # ==============================================================================================================
        # Initialise las object
        header = copy.deepcopy(point_cloud.header)
        header.point_count = n_voxels
        setattr(self, 'las', laspy.LasData(header))
        # ==============================================================================================================
        # Calculate neighbours for the voxels
        if neighbours:
            # Offset of each neighbour from the idx of the voxel (26 neighbours)
            neighbor_offset = np.array([(- n_x * n_y - n_x - 1), (- n_x * n_y - n_x), (- n_x * n_y - n_x + 1),
                                        (- n_x * n_y - 1), (- n_x * n_y), (- n_x * n_y + 1),
                                        (- n_x * n_y + n_x - 1), (- n_x * n_y + n_x), (- n_x * n_y + n_x + 1),
                                        (- n_x - 1), (-n_x), (-n_x + 1),
                                        (- 1), (+ 1),
                                        (+ n_x - 1), (+ n_x), (+ n_x + 1),
                                        (+ n_x * n_y - n_x - 1), (+ n_x * n_y - n_x), (+ n_x * n_y - n_x + 1),
                                        (+ n_x * n_y - 1), (+ n_x * n_y), (+ n_x * n_y + 1),
                                        (+ n_x * n_y + n_x - 1), (+ n_x * n_y + n_x), (+ n_x * n_y + n_x + 1)])

            # Need to reshape for broadcasting
            neighbor_offset = neighbor_offset.reshape(1, len(neighbor_offset))

            # idx of the neighbours of each voxel in idx Nx26
            vx_neighbours = (idx.reshape(-1, 1) + neighbor_offset).astype('uint')
            # Avoid problems of data type
            n_x = n_x.astype(vx_neighbours.dtype)
            n_y = n_y.astype(vx_neighbours.dtype)
            n_z = n_z.astype(vx_neighbours.dtype)

            # Generate a mask with True in the indexes of the neighbours that do not exist
            mask = np.zeros(vx_neighbours.shape, dtype=bool)

            # Voxels in upper limit X do not have these neighbours
            mask[np.ix_(idx % n_x == (n_x-1), [2, 5, 8, 11, 13, 16, 19, 22, 25])] = True
            # Voxels in lower limit X do not have these neighbours
            mask[np.ix_(idx % n_x == 0, [0, 3, 6, 9, 12, 14, 17, 20, 23])] = True
            # Voxels in upper limit Y do not have these neighbours
            mask[np.ix_(np.trunc(idx / n_x) % n_y == (n_y - 1), [6, 7, 8, 14, 15, 16, 23, 24, 25])] = True
            # Voxels in lower limit Y do not have these neighbours
            mask[np.ix_(np.trunc(idx / n_x) % n_y == 0, [0, 1, 2, 9, 10, 11, 17, 18, 19])] = True
            # Greater than the maximum (Upper limit Z)
            mask[vx_neighbours > ((n_x * n_y * n_z) - 1)] = True
            # Lower than the minimum (Lower limit Z)
            mask[vx_neighbours < 0] = True

            # Remove no existing neighbours (because do not exist voxels with that indexes)
            vx_neighbours = vx_neighbours.reshape(-1)
            mask = mask.reshape(-1)
            iloc, iloc_idx = ismember(vx_neighbours, idx)
            mask[~iloc] = True

            # Calculate the neighbours by row index, not by its idx
            vx_neighbours[iloc] = iloc_idx

            # Put as 0 the elements masked to have more space
            vx_neighbours[mask] = 0

            # Reshape to the original shape and apply the mask
            vx_neighbours = vx_neighbours.reshape(-1, neighbor_offset.shape[1]).astype('uint')
            mask = mask.reshape(-1, neighbor_offset.shape[1])

            vx_neighbours = np.ma.masked_array(vx_neighbours, mask=mask)

            # Set property
            setattr(self, 'neighbours', vx_neighbours)

            del vx_neighbours, mask, iloc, iloc_idx, neighbor_offset
        
        # ==============================================================================================================
        if centroid!=[]:
            centroids = self.compute_centroids(indexes_global=idx)
            if centroid == ['xyz']:
                setattr(self.las, 'xyz', centroids)
            else:
                for i in range(len(centroid)):
                    this_property_name = centroid[i]
                    self.las.add_extra_dim(laspy.point.format.ExtraBytesParams(this_property_name, 'float64'))

                    setattr(self.las, this_property_name, centroids[:,i])

        # ==============================================================================================================
        if random!=[]:
            if not random is np.ndarray: random = np.asarray(random)
            if not 'shift_index' in locals():
                shift_index = np.arange(len(idx_))
            data_random, vx_data_random, data_random_size, random, reshape_random, dtype_random = self.__prepare_properties_nonnumba(random, point_cloud, n_voxels, shift_index)
            if not len(random) == 0:
                vx_data_random[:] = data_random[row_idx,:]
                self.__set_property_nonnumba(random, vx_data_random, data_random_size, reshape_random, random_suffix, dtype_random)

        # ==============================================================================================================
        # Calculate properties with numba
        if numba:
            # Load to device order and n_points
            d_order = cuda.to_device(order)
            d_npoints = cuda.to_device(n_points)

            for property_name in mean:
                
                # Load property in device
                d_property = self.__property_to_device(point_cloud, property_name)
                dtype = self.__dtype_las_property(point_cloud.header, property_name)

                d_out  = self.__mean_cuda(d_property, n_voxels, d_order, d_npoints, blocks, threads_per_block)
                
                # Set property
                self.__set_property(property_name, d_out, dtype, mean_suffix)

            # Variance properties
            for property_name in var:

                # Load property in device
                d_property = self.__property_to_device(point_cloud, property_name)
                dtype = self.__dtype_las_property(point_cloud.header, property_name)

                # Load mean if it is computed
                d_mean = None
                if len(mean) !=0:
                    if property_name in mean:
                        d_mean = self.__property_to_device(self.las, property_name+ mean_suffix)

                d_out  = self.__var_cuda(d_property, d_mean, n_voxels, d_order, d_npoints, blocks, threads_per_block)

                # Set property
                self.__set_property(property_name, d_out, dtype, var_suffix)


            # cov properties
            for property_name in cov:

                # Load property in device
                d_property_0 = self.__property_to_device(point_cloud, property_name[0])
                dtype = self.__dtype_las_property(point_cloud.header, property_name[0])             

                # Load property in device
                d_property_1 = self.__property_to_device(point_cloud, property_name[1])
                dtype = self.__dtype_las_property(point_cloud.header, property_name[1])  

                # Load mean if it is computed
                d_mean_0 = None
                d_mean_1 = None
                if len(mean) !=0:
                    if property_name[0] in mean:
                        d_mean_0 = self.__property_to_device(self.las, property_name[0]+ mean_suffix)
                    if property_name[1] in mean:
                        d_mean_1 = self.__property_to_device(self.las, property_name[1]+ mean_suffix)

                d_out = self.__cov_cuda(d_property_0, d_property_1, d_mean_0, d_mean_1, n_voxels, d_order, d_npoints, blocks, threads_per_block)

                # Set property
                property_name = property_name[0] + '_' + property_name[1]
                self.__set_property(property_name, d_out, dtype, cov_suffix)

            # mode properties
            for property_name in mode:
                
                # Load property in device
                d_property = self.__property_to_device(point_cloud, property_name, dtype=np.int32)
                dtype = self.__dtype_las_property(point_cloud.header, property_name)
                       
                d_aux = cuda.to_device(np.ascontiguousarray(np.zeros(shape=(n_voxels, getattr(point_cloud, property_name).max()+1), dtype=np.int32)))
                d_out = cuda.to_device(np.ascontiguousarray(np.zeros(shape=n_voxels, dtype=np.int32)))
                d_nclass = cuda.to_device(np.ascontiguousarray(np.zeros(shape=n_voxels, dtype=np.int32)))

                # Mode
                utils_cuda.sum_class[blocks, threads_per_block](d_property, d_order, d_aux)
                cuda.synchronize
                utils_cuda.mode_column[blocks, threads_per_block](d_aux, d_nclass, d_out)
                cuda.synchronize

                # Set property   
                self.__set_property(property_name, d_out, dtype, mode_suffix)

            # ==============================================================================================================        
            # Set parent_idx property
            setattr(self, 'parent_idx', order)
        
            # ==============================================================================================================  
            if pca_local:

                var_result = np.zeros((n_voxels,3)) # x_x, y_y,z_z
                cov_result = np.zeros((n_voxels,3)) #x_y, x_z, y_z
                var_names = ['x', 'y', 'z']
                cov_names = [['x','y'],['x','z'],['y', 'z']]
                
                # compute var
                for i in range(len(var_names)):
                    property_name = var_names[i]
                    try:
                        var_result[:,i] = getattr(self.las, property_name + var_suffix)
                    except:
                        d_property = self.__property_to_device(point_cloud, property_name)
                        d_mean = None
                        # Load means if they are already calculated
                        if len(mean) !=0:
                            if property_name in mean:
                                d_mean = self.__property_to_device(self.las, property_name+ mean_suffix)

                        var_result[:,i]  = self.__var_cuda(d_property, d_mean, n_voxels, d_order, d_npoints, blocks, threads_per_block).copy_to_host().reshape(-1)

                # compute cov
                for i in range(len(cov_names)):
                    property_name = cov_names[i]
                    if len(cov) !=0:
                        if property_name in cov:
                            cov_result[:,i] = getattr(self.las, property_name[0] + '_' + property_name[1] + cov_suffix)
                            continue
                        elif property_name.reverse() in cov:
                            cov_result[:,i] = getattr(self.las, property_name[1] + '_' + property_name[0] + cov_suffix)
                            continue
                    
                    d_property_0 = self.__property_to_device(point_cloud, property_name[0])
                    d_property_1 = self.__property_to_device(point_cloud, property_name[1])
                    d_mean_0 = None
                    d_mean_1 = None
                    # Load means if they are already calculated
                    if len(mean) !=0:
                        if property_name[0] in mean:
                            d_mean_0 = self.__property_to_device(self.las, property_name[0] + mean_suffix)
                        if property_name[1] in mean:
                            d_mean_1 = self.__property_to_device(self.las, property_name[1] + mean_suffix)

                    cov_result[:,i]  = self.__cov_cuda(d_property_0, d_property_1, d_mean_0, d_mean_1, n_voxels, d_order, d_npoints, blocks, threads_per_block).copy_to_host().reshape(-1)

                eigenvectors = np.zeros((n_voxels,3,3))
                eigenvectors[:,0,0] = var_result[:,0] # x_x
                eigenvectors[:,0,1] = cov_result[:,0] # x_y
                eigenvectors[:,0,2] = cov_result[:,1] # x_z
                eigenvectors[:,1,0] = cov_result[:,0] # x_y
                eigenvectors[:,1,1] = var_result[:,1] # y_y
                eigenvectors[:,1,2] = cov_result[:,2] # y_z
                eigenvectors[:,2,0] = cov_result[:,1] # x_z
                eigenvectors[:,2,1] = cov_result[:,2] # y_z
                eigenvectors[:,2,2] = var_result[:,2] # z_z

                del var_result, cov_result

                eigenvalues, eigenvectors = np.linalg.eig(eigenvectors)
                eigenvectors = np.real(eigenvectors) # It must be always real, but can be some erros if all covariances are close to 0
                # Scale the eigenvalues to sum up 1
                if scale_eigenvalues:
                    eigenvalues = eigenvalues / np.sum(eigenvalues, 1).reshape(-1, 1)

                # Sort the eigenvalues and eigenvectors
                sorted_index = np.argsort(-eigenvalues, axis=1)
                eigenvalues = np.take_along_axis(eigenvalues, sorted_index, axis=1)
                order_broadcasting = np.zeros(eigenvectors.shape, dtype='uint')
                for i in range(order_broadcasting.shape[1]):
                    order_broadcasting[:, i, :] = sorted_index
                eigenvectors = np.take_along_axis(eigenvectors, order_broadcasting, axis=2)
                
                # Recalculate the 3rd eigenvector because if the order is changed 1st x 2nd = -3rd.
                eigenvectors[:, :, 2] = np.cross(eigenvectors[:, :, 0], eigenvectors[:, :, 1])

                setattr(self,'eiv',eigenvalues)
                setattr(self,'eig',eigenvectors)          

        # ==================================================================================================================
        # If not using numba
        else:
            # Calculate the size of each properties
            # ==============================================================================================================
            # Properties calculated using mean()
            if len(mean) != 0:
                mean_properties = True
                if not mean is np.ndarray: mean = np.asarray(mean)
                data_mean, vx_data_mean, data_mean_size, mean, reshape_mean, dtype_mean = self.__prepare_properties_nonnumba(mean, point_cloud, n_voxels, shift_index)
            if len(mean) == 0:
                mean_properties = False
            # ==============================================================================================================
            # Properties calculated using mode()
            if len(mode) != 0:
                mode_properties = True
                if not mode is np.ndarray: mode = np.asarray(mode)
                data_mode, vx_data_mode, data_mode_size, mode, reshape_mode, dtype_mode = self.__prepare_properties_nonnumba(mode, point_cloud, n_voxels, shift_index)
            if len(mode) == 0:
                mode_properties = False
            # ==============================================================================================================
            # Properties calculated using var()
            if len(var) != 0:
                var_properties = True
                if not var is np.ndarray: var = np.asarray(var)
                data_var, vx_data_var, data_var_size, var, reshape_var, dtype_var = self.__prepare_properties_nonnumba(var, point_cloud, n_voxels, shift_index)
            if len(var) == 0:
                var_properties = False

            # ==============================================================================================================
            # parent_idx
            _, vx_parent_idx = np.unique(idx_, return_inverse=True)

            # ==============================================================================================================
            # Calculating the parameters of each voxel
            if mean_properties or mode_properties or var_properties:
                for i in range(n_voxels):
                    
                    # Select points in this voxel
                    if i != 0:
                        points = slice(occ_voxels[i-1, 1],occ_voxels[i, 1], 1)
                    else:
                        points = slice(0, occ_voxels[i, 1], 1)

                    # Calculate each voxel
                    # mean()
                    if mean_properties:
                        vx_data_mean[i, :] = np.mean(data_mean[points], axis=0)
                    # mode()
                    if mode_properties:
                        for j in range(data_mode.shape[1]):
                            vx_data_mode[i, j] = np.bincount(data_mode[points, j].astype('int')).argmax()
                    # var()
                    if var_properties:
                        for j in range(data_var.shape[1]):
                            vx_data_var[i, j] = utils.var(data_var[points, j])

            # ==============================================================================================================
            # Set mean properties
            if mean_properties:
                self.__set_property_nonnumba(mean, vx_data_mean, data_mean_size, reshape_mean, mean_suffix, dtype_mean)

            # Set mode properties
            if mode_properties:
                self.__set_property_nonnumba(mode, vx_data_mode, data_mode_size, reshape_mode, mode_suffix, dtype_mode)

            # Set var properties
            if var_properties:
                self.__set_property_nonnumba(var, vx_data_var, data_var_size, reshape_var, var_suffix, dtype_var)

            # Set Voxels properties
            setattr(self, 'parent_idx', vx_parent_idx)
    # ==================================================================================================================

    # Properties

    @property
    def voxel_size(self):
        """
        Getter of voxel_size.

        :return: size of voxel as float.
        """
        return self.__voxel_size

    @voxel_size.setter
    def voxel_size(self, voxel_size):
        """
        Setter of voxel_size.

        :param voxel_size: size of voxel as float.
        :return: None.
        """

        self.__voxel_size = voxel_size

    @property
    def las(self):
        """
        Getter of las.

        :return: voxelised point cloud as laspy.lasdata.LasData.
        """

        return self.__las
    
    @las.setter
    def las(self, las):
        """
        Setter of las.

        :param las: voxelised point cloud as laspy.lasdata.LasData.
        :return: None.
        """

        self.__las = las
    
    @property
    def parent_idx(self):
        """
        Getter of parent_idx.

        :return: Mx1 numpy.array. M is the number of points in the original point cloud. The value of each point is the index of the voxelised point cloud to which it belongs.
                 The index is the position in the array las.xyz. -1 value means that point is not in the voxelised point cloud.
        """

        return self.__parent_idx

    @parent_idx.setter
    def parent_idx(self, parent_idx):
        """
        Setter of parent_idx.

        :return: Mx1 numpy.array. Contains the index row in las.xyz to which each point in the original point cloud belongs.
                 -1 value means that point is not in the voxelised point cloud.
        :return: None.
        """

        self.__parent_idx = parent_idx

    @property
    def neighbours(self):
        """
        Getter of neighbours.

        :return: Nx26 numpy.ma.core.MaskedArray with the index row in las.xyz of the neighbours of each voxel.
        """

        return self.__neighbours

    @neighbours.setter
    def neighbours(self, neighbours):
        """
        Setter of neighbours.

        :param neighbours: Nx26 numpy.ma.core.MaskedArray with the index row in las.xyz of the neighbours of each voxel.
        :return: None.
        """

        self.__neighbours = neighbours

    @property
    def eiv(self):
        """
        Getter of eiv.

        :return: Nx3 numpy array of eigenvalues.
        """

        return self.__eiv

    @eiv.setter
    def eiv(self, eiv):
        """
        Setter of eiv.

        :param eiv: Nx3 numpy array of eigenvalues.
        :return: None.
        """

        self.__eiv = eiv

    @property
    def eig(self):
        """
        Getter of eig.

        :return: Nx3x3 numpy array of eigenvectors.
        """

        return self.__eig

    @eig.setter
    def eig(self, eig):
        """
        Setter of eig.

        :param eig: Nx3x3 numpy array of eigenvectors.
        :return: None.
        """

        self.__eig = eig


    @property
    def grid_size(self):
        """
        Getter of grid_size.

        :return: size of grid in number of voxels.
        """
        return self.__grid_size

    @grid_size.setter
    def grid_size(self, grid_size):
        """
        Setter of grid_size.

        :param grid_size: size of grid in number of voxels.
        :return: None.
        """

        self.__grid_size = grid_size


    @property
    def min_coords(self):
        """
        Getter of min_coords.

        :return: minimun coordinates of the point cloud.
        """
        return self.__min_coords

    @min_coords.setter
    def min_coords(self, min_coords):
        """
        Setter of min_coords.

        :param min_coords: size of grid in number of voxels.
        :return: None.
        """

        self.__min_coords = min_coords


    @property
    def indexes_grid(self):
        """
        Getter of indexes_grid.

        :return: Nx3 grid indexes_grid of self.las points.
        """
        return self.__indexes_grid

    @indexes_grid.setter
    def indexes_grid(self, indexes_grid):
        """
        Setter of indexes_grid.

        :param indexes_grid: Nx3 grid indexes_grid of self.las points.
        :return: None.
        """

        self.__indexes_grid = indexes_grid

    @property
    def grid(self):
        """
        Getter of grid.

        :return: NxMxW occupation grid.
        """
        return self.__grid

    @grid.setter
    def grid(self, grid):
        """
        Setter of grid.

        :param indexes_grid: NxMxW occupation grid.
        :return: None.
        """

        self.__grid = grid
    # ==================================================================================================================
    # Methods


    def get_parent_idx(self, indexes):
        '''
        Method that returns a boolean array with True in those points in the original point cloud that are in any voxel specified in indexes.

        :param indexes: boolean or int array.
        :return: Mx1 as bool numpy.ndarray.
        '''

        if not indexes is np.ndarray: indexes = np.asarray(indexes)
        if indexes.dtype == 'bool': indexes = np.where(indexes)[0]

        iloc, _ = ismember(self.parent_idx, indexes)

        return iloc
    
    
    def __getitem__(self, indexes):
        '''
        Method for selectiong the specified voxels.
        Property las usese the laspy __getitem__ method.
        Property neighbours is recomputed to calculated the row indexes in the new las.xyz array.
        Property parent_idx is also recomputed by recalculating the row indexes and writting a -1 in those points that are in not selected voxels.

        :param indexes: boolean or int array.
        return: Voxels.
        '''
        
        if not indexes is np.ndarray: indexes = np.asarray(indexes)
        if indexes.shape == (): indexes = indexes.reshape(-1)
        # getitem in las

        # Create object
        vx_selected = Voxels()
        setattr(vx_selected, 'las', self.las[indexes])
        setattr(vx_selected, 'voxel_size', self.voxel_size)
        setattr(vx_selected, 'grid_size', self.grid_size)
        setattr(vx_selected, 'min_coords', self.min_coords)

        if self.indexes_grid is not None:
            setattr(vx_selected, 'indexes_grid', self.indexes_grid[indexes])
            grid = np.zeros((vx_selected.grid_size[0],vx_selected.grid_size[1],vx_selected.grid_size[2]), dtype=np.bool_)
            grid[vx_selected.indexes_grid[:,0], vx_selected.indexes_grid[:,1], vx_selected.indexes_grid[:,2]] = True
            setattr(vx_selected, 'grid', grid)

        if not self.eiv is None:
            setattr(vx_selected, 'eiv', self.eiv[indexes])
        if not self.eiv is None:
            setattr(vx_selected, 'eig', self.eig[indexes])

        # Neighbours
        if not self.neighbours is None:
            vx_selected.neighbours = self.neighbours[indexes]

            # It is necessary to update the indexes of the neighbours
            # Reshape to use ismember
            vx_neighbours = vx_selected.neighbours.data.reshape(-1)
            mask = vx_selected.neighbours.mask.reshape(-1)

            # ismember works with int indexes
            if indexes.dtype == 'bool': indexes = np.where(indexes)[0]
            iloc, iloc_idx = ismember(vx_neighbours, indexes)

            # Update mask
            mask[~iloc] = True
            # Re-calculate the neighbours by row index
            vx_neighbours[iloc] = iloc_idx

            # Put as 0 the elements masked to have more space
            vx_neighbours[mask] = 0

            # Reshape to the original shape and apply the mask
            vx_neighbours = vx_neighbours.reshape(-1, vx_selected.neighbours.shape[1]).astype('uint')
            mask = mask.reshape(-1, vx_selected.neighbours.shape[1])

            setattr(vx_selected, 'neighbours', np.ma.masked_array(vx_neighbours, mask=mask))


        # parent_idx
        if not self.parent_idx is None:

            parent_idx = copy.deepcopy(self.parent_idx)

            # Update voxel row indexes
            if indexes.dtype == 'bool': indexes = np.where(indexes)[0]
            iloc, iloc_idx = ismember(parent_idx, indexes)

            parent_idx[~iloc] = -1 # -1 means this point in the original point is not in any voxel
            parent_idx[iloc] = iloc_idx

            setattr(vx_selected, 'parent_idx', parent_idx)

        return vx_selected
    
    
    def __len__(self):
        '''
        Method that returns the number of voxels.

        return: len(self.las)
        '''
        return len(self.las)
    

    def compute_indexes_global(self, indexes:np.array) -> np.array:
        '''
        Compute the global index using the indexes x,y,z

        :param indexes: Nx3 numpy.array() with indexes x, y and z of voxels.
        :return global_indexes Nx1 numpy.array()
        '''
        global_indexes = indexes[:,0] + (indexes[:,1] * self.grid_size[0]) + (indexes[:,2] * self.grid_size[0] * self.grid_size[1])
        return global_indexes

    def compute_indexes_xyz(self, global_indexes: np.array) -> np.array:
        '''
        Compute indexes x, y and z.

        :param global indexes Nx1 numpy.array()
        :return indexes_xyz Nx3 numpy.array()
        '''
        indexes_xyz = np.zeros((len(global_indexes),3), dtype='uint')

        indexes_xyz[:,2] = np.floor_divide(global_indexes, (self.grid_size[0]*self.grid_size[1]))
        remaninder = np.remainder(global_indexes, (self.grid_size[0]*self.grid_size[1]))
        indexes_xyz[:,1] = np.floor_divide(remaninder, self.grid_size[0])
        indexes_xyz[:,0] = np.remainder(remaninder, self.grid_size[0])

        return indexes_xyz


    def compute_centroids(self, indexes_xyz=None, indexes_global=None) -> np.array:
        '''
        Compute the centroids (x,y,z) of the voxels using their indexes.
        Use indexe_xyz or indexes_global.

        :param indexes_xyz: global indexes Nx3 numpy.array(). [Default:None]
        :param indexes_global: global indexes Nx1 numpy.array(). [Default:None]
        :return centroids: Nx3 centroids x,y,z.
        '''

        if indexes_global is not None:
            indexes_xyz = self.compute_indexes_xyz(indexes_global)

        centroids = np.zeros(indexes_xyz.shape)
        centroids[:, 0] = np.round(indexes_xyz[:, 0] * self.voxel_size[0] + self.min_coords[0] + 0.5*self.voxel_size[0], len(str(self.las.header.scale[0])) - 2)
        centroids[:, 1] = np.round(indexes_xyz[:, 1] * self.voxel_size[1] + self.min_coords[1] + 0.5*self.voxel_size[1], len(str(self.las.header.scale[1])) - 2)
        centroids[:, 2] = np.round(indexes_xyz[:, 2] * self.voxel_size[2] + self.min_coords[2] + 0.5*self.voxel_size[2], len(str(self.las.header.scale[2])) - 2)

        return centroids


    def __set_property(self, property_name, d_out, dtype, suffix):

        if type(d_out) is cuda.cudadrv.devicearray.DeviceNDArray:
            d_out = d_out.copy_to_host()

        if suffix =='':
                setattr(self.las, property_name, d_out)
        else:
            this_property_name = property_name + suffix
            try:
                self.las.add_extra_dim(laspy.point.format.ExtraBytesParams(this_property_name, dtype))
                setattr(self.las, this_property_name, d_out)
            except:
                raise ValueError("LAS format does not allow to add extra dimensions with several columns such as {0}".format(this_property_name))


    @staticmethod
    def __dtype_las_property(header, property_name):
        '''
        Method to know the laspy property data type without read it.
        It uses header.point_format.dimension_by_name(property_name).

        :param info_property: laspy_object.header
        :param property_name: name of the property.
        '''
        if property_name=='x' or property_name=='y' or property_name=='z':
            return 'float64'
        
        n_bits = str(header.point_format.dimension_by_name(property_name).num_bits)

        if str(header.point_format.dimension_by_name(property_name).kind) == 'DimensionKind.SignedInteger':
            kind = 'int'
        elif str(header.point_format.dimension_by_name(property_name).kind) == 'DimensionKind.UnsignedInteger':
            kind = 'uint'
        elif str(header.point_format.dimension_by_name(property_name).kind) == 'DimensionKind.BitField':
            kind = 'uint'
            n_bits=str(8)
        elif str(header.point_format.dimension_by_name(property_name).kind) == 'DimensionKind.FloatingPoint':
            kind = 'float'

        return kind+n_bits


    @staticmethod
    def __property_to_device(point_cloud, property_name, dtype=np.float64):
            d_property = cuda.to_device(np.ascontiguousarray(np.asarray(getattr(point_cloud, property_name)).astype(dtype)))
            return d_property
    

    @staticmethod
    def check_list(list_properties):
        '''
        Faster without xyz
        '''
        if len(list_properties) !=0:
            if 'xyz' in list_properties:
                list_properties.append('x'), list_properties.append('y'), list_properties.append('z')
                list_properties.remove('xyz')
        return list_properties


    @staticmethod
    def __mean_cuda(d_property, n_voxels, d_order, d_npoints, blocks, threads_per_block):

        d_out = cuda.to_device(np.ascontiguousarray(np.zeros(shape=n_voxels, dtype=np.float64)))

        # Mean
        utils_cuda.mean[blocks, threads_per_block](d_property, d_order, d_npoints, d_out)
        cuda.synchronize

        return d_out
    

    @staticmethod
    def __var_cuda(d_property, d_mean, n_voxels, d_order, d_npoints, blocks, threads_per_block):

        # Load or compute mean  
        if d_mean is None:
            d_mean = Voxels.__mean_cuda(d_property, n_voxels, d_order, d_npoints, blocks, threads_per_block)
        
        d_out = cuda.to_device(np.ascontiguousarray(np.zeros(shape=n_voxels, dtype=np.float64)))

        # Var
        utils_cuda.var[blocks, threads_per_block](d_property, d_order, d_mean, d_npoints, d_out)
        cuda.synchronize

        return d_out
    

    @staticmethod
    def __cov_cuda(d_property_0, d_property_1, d_mean_0, d_mean_1, n_voxels, d_order, d_npoints, blocks, threads_per_block):

        # Load or compute mean  
        if d_mean_0 is None:
            d_mean_0  = Voxels.__mean_cuda(d_property_0, n_voxels, d_order, d_npoints, blocks, threads_per_block)

        # Load or compute mean  
        if d_mean_1 is None:
            d_mean_1  = Voxels.__mean_cuda(d_property_1, n_voxels, d_order, d_npoints, blocks, threads_per_block)

        d_out = cuda.to_device(np.ascontiguousarray(np.zeros(shape=n_voxels, dtype=np.float64)))

        # Var
        utils_cuda.cov[blocks, threads_per_block](d_property_0, d_property_1, d_order, d_mean_0, d_mean_1, d_npoints, d_out)
        cuda.synchronize

        return d_out
    

    @staticmethod
    def __prepare_properties_nonnumba(dimensions, point_cloud, n_voxels, shift_index):
        '''
        Method for generate the arrays to voxelise the data.
        '''
        data_size = np.zeros(len(dimensions), dtype='uint')
        keep = np.ones(len(dimensions), dtype='bool')
        reshape = np.zeros(len(dimensions), dtype='bool')
        dtype = np.zeros(len(dimensions), dtype='object')

        for i in range(len(dimensions)):
            d_property = np.asarray(getattr(point_cloud, dimensions[i]))

            if np.all(d_property==0): # if all are 0 do not compute it
                keep[i] = False
                continue

            if len(d_property.shape)==1: 
                d_property = d_property.reshape(-1,1)
                reshape[i] = True

            data_size[i] = d_property.shape[1]
            dtype[i] = str(d_property.dtype)

        # remove properties
        dimensions = dimensions[keep]
        data_size = data_size[keep]
        reshape = reshape[keep]
        dtype = dtype[keep]

        if len(dimensions)== 0:
            return None, None, None, dimensions, None, None

        data_size = np.cumsum(data_size)

        # Initialising the matrix with the LAS properties sorted and the matrix in which the result will be saved
        data = np.zeros((point_cloud.xyz.shape[0], data_size[-1]))
        vx_data = np.zeros((n_voxels, data_size[-1]))

        for i in range(len(dimensions)):
            d_property = np.asarray(getattr(point_cloud, dimensions[i]))[shift_index]
            if len(d_property.shape)==1: d_property = d_property.reshape(-1,1)

            if i != 0:
                data[:, data_size[i-1]:data_size[i]] = d_property
            else:
                data[:, 0:data_size[i]] = d_property

        return data, vx_data, data_size, dimensions, reshape, dtype
    

    def __set_property_nonnumba(self, dimensions, vx_data, data_size, reshape, suffix, dtype):

        if suffix =='':
            for i in range(len(dimensions)):
                if i != 0:
                    if reshape[i]:
                        setattr(self.las, dimensions[i], vx_data[:,data_size[i-1]:data_size[i]].reshape(-1))
                    else:
                        setattr(self.las, dimensions[i], vx_data[:,data_size[i-1]:data_size[i]])
                else:
                    if reshape[i]:
                        setattr(self.las, dimensions[i], vx_data[:, 0:data_size[i]].reshape(-1))
                    else:
                        setattr(self.las, dimensions[i], vx_data[:, 0:data_size[i]])
        else:
            for i in range(len(dimensions)):
                this_property_name = dimensions[i] + suffix
                error = False
                if i!=0:
                    error = data_size[i] - data_size[i-1] > 1
                else:
                    error = data_size[i] > 1
                if error:
                    raise ValueError("LAS format does not allow to add extra dimensions with several columns such as {0}".format(this_property_name))

                self.las.add_extra_dim(laspy.point.format.ExtraBytesParams(this_property_name, dtype[i]))

                if i != 0:
                    setattr(self.las, this_property_name, vx_data[:,data_size[i-1]:data_size[i]].reshape(-1))
                else:
                    setattr(self.las, this_property_name, vx_data[:, 0:data_size[i]].reshape(-1))
