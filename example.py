import laspy
from VoxRasterLAS.Voxels import Voxels
from VoxRasterLAS.Raster import Raster


cloud_path = "data/BaileyTruss_000.las"

# Read point cloud
las = laspy.read(cloud_path)

# Raster
rt_numba    = Raster(las, grid=0.01, min_dimensions=['z'], numba=True)
rt_nonnumba    = Raster(las, grid=0.01, min_dimensions=['z'], numba=False)

# Voxels
vx_numba = Voxels(las, grid=[0.02,0.2,0.2], mean=['xyz'], centroid=['a', 'b', 'c'], numba=True)
vx_nonnumba = Voxels(las, grid=[0.02,0.2,0.2], mean=['xyz'], centroid=['a', 'b', 'c'], numba=False)

# Voxels random
vx_nonnumba_random = Voxels(las, grid=[0.02,0.2,0.2], random=['xyz', 'classification'], neighbours=True, numba=False)

# Voxelised LAS object
new_las = vx_nonnumba.las

# Neighbours
nb = vx_nonnumba_random.neighbours