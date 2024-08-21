# VoxRasterLAS
This library rasterises and voxelises 3D point clouds. It takes a laspy object as input and performs feature extraction.

Created by [Daniel Lamas Novoa](https://orcid.org/0000-0001-7275-183X), [Mario Soilán Rodríguez](https://orcid.org/0000-0001-6545-2225), and [Belén Riveiro Rodríguez](https://orcid.org/0000-0002-1497-4370) from [GeoTech Group](https://geotech.webs.uvigo.es/en/), [CINTECX](http://cintecx.uvigo.es/gl/), [UVigo](https://www.uvigo.gal/).


## Overview
This library rasterises and voxelises laspy objects, performing feature extraction at the raster and voxel level, respectively.
To speed up the process, operations are performed using GPUs with the Numba library.

### Voxelisation
 Takes the laspy object and return a Voxel object which contains the following properties:
 - voxel_size:  size of voxel.
 - neighbours:  Nx26 numpy.ma.core.MaskedArray with the index row in las.xyz of the neighbours of each voxel.
 - parent_idx: Mx1 numpy.array. M is the number of points in the original point cloud. The value of each point is the index of the voxelised point cloud to which it belongs. The index is the position in the array las.xyz. -1 value means that point is not in the voxelised point cloud.
 - eig: Nx3 numpy array of eigenvalues. Result of PCA at voxel level.
 - eiv: Nx3x3 numpy array of eigenvectors. Result of PCA at voxel level.
 - grid:  NxMxW occupation grid.
 - indexes_grid: Nx3 grid indexes_grid of self.las points.
 - las: laspy object with the voxelised point cloud. New fields are created containing the features extractured to each field of the laspy object. The features are:
    - mean
    - mode
    - variance
    - centroid
    - random selection

and the method:
- get_parent_idx: returns a boolean array with True in those points in the original point cloud that are in any voxel specified in indexes.

### Rasterisation
 Takes the laspy object and return a Raster object which contains the following properties:
 - pixel_size: size of pixel.
 - parent_idx: Mx1 numpy.array. M is the number of points in the original point cloud. The value of each point is the index of the pixelised image point cloud to which it belongs.
 - feature: MxN raster that contain the feature extracted at pixel level to each field of the point cloud. The features are:
   - occupation
   - minimum
   - maximum
   - variance
   - mode

  and the method:
  - get_parent_idx: returns a boolean array with True in those points in the original point cloud that are in any voxel specified in indexes.

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{lamas_voxrasterlas_2024,
	address = {Vigo, Spain},
	title = {VoxRasterLAS: A Python library for fast voxelisation and rasterisation of LAS 3D point clouds using GPUs.},
	url = {https://3dgeoinfoeg-ice.webs.uvigo.es/proceedings},
	abstract = {Voxelisation and rasterisation are common techniques for structuring point clouds. In addition, feature extraction at the voxel and raster level is also a widely used technique to facilitate the analysis or segmentation process of point clouds. We present an open-source Python library available in PyPi called VoxRasterLAS for voxelisation and rasterisation of point clouds, which allows feature extraction at voxel and raster level. The library utilizes laspy objects, a library that deal with LAS fi les directly. Moreover, GPUs are used to accelerate operations. The paper presents computational times, use cases, and a comparison with other libraries.},
	author = {Lamas, Daniel and Soilán, Mario and Riveiro, Belen},
	booktitle = {31st International Workshop on Intelligent Computing in Engineering},
	month = jul,
	year = {2024},
	pages = {667--676},
}
```

## Licence
VoxRasterLAS

Copyright (C) 2024 GeoTECH Group <geotech@uvigo.gal>

Copyright (C) 2024 Daniel Lamas Novoa <daniel.lamas.novoa@uvigo.gal>

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program in ![COPYING](https://github.com/GeoTechUVigo/VoxRasterLAS/blob/main/COPYING). If not, see <https://www.gnu.org/licenses/>.


## Installation
This package can also be used with and without NUMBA package version 0.57. To use with NUMBA, Nvidia drivers and CUDA SDK must be preinstalled (check numba instructions https://numba.pydata.org/numba-doc/latest/user/installing.html):

LAS decompressor might be installed via pip compatible with laspy package.

To install VoxRasterLAS (available in pip):
```
python3 -m pip install VoxRasterLAS==0.1.0
```
