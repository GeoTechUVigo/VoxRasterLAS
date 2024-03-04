# VoxRasterLAS
This library rasterises and voxelises 3D point clouds. It takes a laspy object as input and performs feature extraction.

## Overview
This library rasterises and voxelises laspy objects, performing feature extraction at the raster and voxel level, respectively.
To speed up the process, operations are performed using GPUs with the Numba library.

### Voxelisation
Takes a laspy object and return a Voxels object which contains the following properties:
- 

### Rasterisation
Takes a laspy object and return a Raster object which contains the following properties:
-


## Citation
If you find our work useful in your research, please consider citing:
```
TODO:INTRODUCIR PUBLICACION
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