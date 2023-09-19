# VoxRasterLAS
Library to voxelise point clouds according to the LAS format, and rasterise them. It uses the numba library, which allows the GPU to be used with python.

## Overview


## Citation


## Licence
VoxRasterLAS

Copyright (C) 2023 GeoTECH Group <geotech@uvigo.gal>

Copyright (C) 2023 Daniel Lamas Novoa <daniel.lamas.novoa@uvigo.gal>

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program in ![COPYING](https://github.com/GeoTechUVigo/VoxRasterLAS/blob/main/COPYING). If not, see <https://www.gnu.org/licenses/>.


## Installation
This package can also be used with and without NUMBA package version 0.57. To use with NUMBA, Nvidia drivers and CUDA SDK must be preinstalled (check numba instructions https://numba.pydata.org/numba-doc/latest/user/installing.html):

LAS decompressor might be installed via pip compatible with laspy package.

To install VoxRasterLAS (available in test.pypi):
```
python3 -m pip install --extra-index-url https://test.pypi.org/simple/ VoxRasterLAS==0.0.23
```