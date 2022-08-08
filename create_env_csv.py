import csv
import os
from convective_functions import *
from netCDF4 import Dataset
from metpy.units import units

sounding_directory = "E:/Microburst/valid_soundings/"
data_directory1 = "D:/Microburst/cm1out/"
data_directory2 = "G:/Microburst/cm1out/"
data_directory3 = "E:/Microburst/cm1out/"
last_file = "cm1out_000120.nc"

save_dir = "E:/Microburst/"
savename = "sounding_environments.csv"

def get_valid_soundings(path):
	sounding_list = pd.read_csv(path)
	return sounding_list.values[:, 0]

def get_max_wind(path,last_file):
	nc = Dataset(os.path.join(path,last_file))
	# get winds
	maxwspd = nc.variables["sws"][:][0]
	linearmx = np.asarray(maxwspd).flatten()
	peak_max = np.max(linearmx)
	return peak_max

def getEnvironmentInfo(df):
	pressure = df["pres"].values * units.hPa
	temp = df["temp"].values * units.degC
	dewp = df["dewp"].values * units.degC
	hght = df["hght"].values
	# Convert height from MSL to AGL
	hght = [H - hght[0] for H in hght]
	hght = hght * units.meter
	u = df["u"].values * units('m/s')
	v = df["v"].values * units('m/s')

	sfc_temp = temp[0].magnitude
	sfc_dewp = dewp[0].magnitude
	dewp_depression = sfc_temp - sfc_dewp

	pwat = precipitable_water(pressure,dewp)

	LI_500 = lifted_index(pressure,temp,dewp)
	LI_700 = lifted_index(pressure, temp, dewp,target_level=700.0)
	LI_300 = lifted_index(pressure, temp, dewp, target_level=300.0)

	col_min_rh = np.nanmin(relh_profile(temp,dewp))
	col_max_rh = np.nanmax(relh_profile(temp,dewp))
	col_rh_delta = col_max_rh - col_min_rh

	mean_rh_1km = layer_mean_relh(hght,temp,dewp,0.0 * units.meter,1000.0*units.meter)
	mean_rh_3km = layer_mean_relh(hght,temp,dewp,0.0 * units.meter,3000.0*units.meter)
	mean_rh_700_500 = layer_mean_relh(hght, temp, dewp, 700.0 * units.hPa, 500.0 * units.hPa, pres=pressure)
	mean_rh_600_300 = layer_mean_relh(hght, temp, dewp, 600.0 * units.hPa, 300.0 * units.hPa, pres=pressure)

	theta_e = theta_e_profile(pressure,temp,dewp)
	theta_e_defficit = thetaE_defficit(pressure,theta_e)
	tv = virtualTemp(pressure,temp,dewp)
	sbcape, sbcin = getCape("surface",pressure,tv, dewp)
	mlcape, mlcin  = getCape("mixed-layer", pressure, tv, dewp)
	mucape, mucin = getCape("most-unstable",pressure, tv, dewp)
	dcape = DCAPE(temp,dewp,pressure,target_pres=500.0*units.hPa)
	BS1km = bulk_shear_magnitude(pressure, hght, u, v, depth=1000)
	BS3km = bulk_shear_magnitude(pressure, hght, u, v, depth=3000)
	BS6km = bulk_shear_magnitude(pressure, hght, u, v, depth=6000)
	BSeff = get_effective_bulk_shear(pressure, temp, dewp, u, v,hght)

	max_wind_03 = max_layer_wind_speed(hght,u,v)
	max_wind_06 = max_layer_wind_speed(hght,u,v,bottom=0.0,top=6000.0)

	CB = craven_brooks_sig_severe(mucape,BSeff)
	ESRH = effective_storm_relative_helicity(hght,pressure,u,v,temp,dewp)
	SCP = supercell_composite_parameter(mucape, mucin, ESRH, BSeff)
	LR3km = lapse_rate(hght,temp)

	# BLD = boundary_layer_depth(hght,temp)
	kp_wind = kuchera_parker_wind_param(pressure,temp,dewp,u,v)
	micro_comp = microburst_composite(pressure,temp,dewp,hght)

	# Surface-based parcel LCL, LFC, EL
	sb_lcl_p, sb_lcl_t, sb_lfc_p, sb_lfc_t, sb_el_p, sb_el_t = getParcelInfo("surface",pressure,tv,dewp)
	# Mixed-layer parcel LCL, LFC, EL
	ml_lcl_p, ml_lcl_t, ml_lfc_p, ml_lfc_t, ml_el_p, ml_el_t = getParcelInfo("mixed-layer",pressure,tv,dewp)
	# Most-unstable parcel LCL, LFC, EL
	mu_lcl_p, mu_lcl_t, mu_lfc_p, mu_lfc_t, mu_el_p, mu_el_t = getParcelInfo("most-unstable",pressure,tv,dewp)

	return [sfc_temp, sfc_dewp, dewp_depression, pwat.magnitude, LI_700, LI_500, LI_300,
			col_min_rh.magnitude, col_max_rh.magnitude, col_rh_delta.magnitude, mean_rh_1km.magnitude,
			mean_rh_3km.magnitude, mean_rh_700_500.magnitude, mean_rh_600_300.magnitude, theta_e_defficit.magnitude,
			sbcape.magnitude, sbcin.magnitude, mlcape.magnitude, mlcin.magnitude, mucape.magnitude, mucin.magnitude, dcape.magnitude,
			BS1km.magnitude, BS3km.magnitude, BS6km.magnitude, BSeff.magnitude, max_wind_03.magnitude, max_wind_06.magnitude,
			CB, ESRH.magnitude, SCP, LR3km.magnitude, kp_wind, micro_comp,
			sb_lcl_p.magnitude, sb_lcl_t.magnitude, sb_lfc_p.magnitude, sb_lfc_t.magnitude, sb_el_p.magnitude, sb_el_t.magnitude,
			ml_lcl_p.magnitude, ml_lcl_t.magnitude, ml_lfc_p.magnitude, ml_lfc_t.magnitude, ml_el_p.magnitude, ml_el_t.magnitude,
			mu_lcl_p.magnitude, mu_lcl_t.magnitude, mu_lfc_p.magnitude, mu_lfc_t.magnitude, mu_el_p.magnitude, mu_el_t.magnitude]

header = ["Run","max_wind","surface_temp", "surface_dewp", "dewpoint_depression", "PWAT", "LI_700", "LI_500", "LI_300",
		  "column_min_RH", "column_max_RH", "column_rh_delta", "mean_rh_1km", "mean_rh_3km", "mean_rh_700_500", "mean_rh_600_300",
		  "thetae_defficit", "SBCAPE", "SBCIN", "MLCAPE", "MLCIN", "MUCAPE", "MUCIN","DCAPE", "BS1km", "BS3km",
		  "BS6km","BSeff", "max_wind_03km", "max_wind_06km", "Cravenbrooks", "ESRH", "SCP", "LR3km","KP_wind", "Microburst_composite",
		  "sb_lcl_p", "sb_lcl_t", "sb_lfc_p", "sb_lfc_t", "sb_el_p", "sb_el_t",
		  "ml_lcl_p", "ml_lcl_t", "ml_lfc_p", "ml_lfc_t", "ml_el_p", "ml_el_t",
		  "mu_lcl_p", "mu_lcl_t", "mu_lfc_p", "mu_lfc_t", "mu_el_p", "mu_el_t"]

with open(os.path.join(save_dir, savename), 'w',newline="") as csvfile:
	csvwriter = csv.writer(csvfile)
	csvwriter.writerow(header)
	for sounding in os.listdir(sounding_directory):
		df = read_snd(os.path.join(sounding_directory, sounding))
		# For each sounding we need to do two things:
		# 1) Get the max wind from the simulation
		# 2) Get the profile data
		sounding_dir = sounding[0:-4]
		# Get the max wind data.
		# Try the first path
		try:
			subpath = os.path.join(data_directory1, sounding_dir)
			max_wind = get_max_wind(subpath, last_file)
		except:
			# Try the second path
			try:
				subpath = os.path.join(data_directory2, sounding_dir)
				max_wind = get_max_wind(subpath, last_file)
			except:
				# Try the third path
				subpath = os.path.join(data_directory3, sounding_dir)
				max_wind = get_max_wind(subpath, last_file)
		# Get the sounding profile
		df = read_snd(os.path.join(sounding_directory, sounding))
		# Now pass the dataframe to the environment function
		data = getEnvironmentInfo(df)
		data.insert(0,max_wind)
		data.insert(0,sounding_dir)
		csvwriter.writerow(data)

csvfile.close()
