from metpy.io import GempakGrid
from metpy.units import units
import metpy.calc as mcal
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import numpy as np

gempak_dir = "/home/andrew/Research/microburst/nam"
gem_file = "nam40_2022081012f001"
path = os.path.join(gempak_dir, gem_file)
grid = GempakGrid(path)

for info in grid.gdinfo():
	print(info)

def create_3d_grid(grid,vcoord,levels):
	# Create arrays for the 3-D grids
	temp = []
	dewp = []
	u_winds = []
	v_winds = []
	for level in levels:
		tempk = np.asarray(grid.gdxarray(parameter='TMPK', coordinate=vcoord, level=int(level))).squeeze() * units.kelvin
		specific_hum = np.asarray(grid.gdxarray(parameter='SPFH', coordinate=vcoord, level=int(level))).squeeze() * units.percent
		pres = np.asarray(grid.gdxarray(parameter='PRES', coordinate=vcoord, level=int(level))).squeeze() * units.hPa
		dewpk = mcal.dewpoint_from_specific_humidity(pres, tempk, specific_hum)
		u = np.asarray(grid.gdxarray(parameter='UREL', coordinate=vcoord, level=int(level))).squeeze() * units.knots
		v = np.asarray(grid.gdxarray(parameter='VREL', coordinate=vcoord, level=int(level))).squeeze() * units.knots
		u = u.to('m/s')
		v = v.to('m/s')
		temp.append(tempk)
		dewp.append(dewpk)
		u_winds.append(u)
		v_winds.append(v)
	return pres, temp, dewp, u_winds, v_winds

# Define the pressure levels
pres_levels = np.arange(1000,75.0,-25.0)
pres_coord = "PRES"
# Define the height levels 
height_levels = np.arange(0.0,9000.0,500.0)
height_coord = "ZAGL"

pres, temp, dewp, u_winds, v_winds = create_3d_grid(grid, height_coord, height_levels)

