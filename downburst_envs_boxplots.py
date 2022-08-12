import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm 
from statsmodels.formula.api import ols 

env_csv_path = "/home/andrew/Research/microburst/sounding_environments.csv"
save_path = "/home/andrew/Research/microburst/images/"


def composite_param(df):
	bs_filter = np.where(np.asarray(df["BS6km"]) < 12.0, 1.0, 0.0)
	# sbcin_filter = np.where(np.asarray(df["SBCIN"]) < -25.0, 0.0, 1.0)
	temp = df["surface_temp"]/30.0
	#dewp_depress = df["dewpoint_depression"]/9.0
	dd_filter = np.where(np.asarray(df["dewpoint_depression"]) < 7.5,0.0,1.0)
	#esrh = (10.0 - df["ESRH"])/10.0
	#li_700 = df["LI_700"]/-3.0
	lr3km = df["LR3km"]/7.5
	max_wind_3km = df["max_wind_03km"]/10.0
	mlcin = (50.0+df["MLCIN"])/25.0
	ml_el_p = 170.0/df["ml_el_p"]
	ml_el_t = df["ml_el_t"]/-60.0
	ml_lfc_t = df["ml_lfc_t"]/15.0

	#li_700 = np.where(np.asarray(li_700) < 0.0, 0.0, li_700)
	ml_el_t = np.where(np.asarray(ml_el_t) < 0.0, 0.0, ml_el_t)
	mlcin = np.where(np.asarray(mlcin) < 0.0, 0.0, mlcin)

	# Winning combo: temp * max_wind_3km * ml_el_p * ml_el_t * ml_lfc_t * mlcin 
	composite = temp * max_wind_3km  * ml_el_p * ml_el_t  * ml_lfc_t * mlcin * bs_filter * dd_filter
	# print("Relative Weights:")
	# for i in range(0,len(composite)):
	# 	print(f'Comp: {round(composite[i],3)}, temp: {round(temp[i],3)}, max_wind_03km: {round(max_wind_3km[i],3)}\
	# 		03km LR: {round(lr3km[i],3)}, ML_el_p: {round(ml_el_p[i],3)}, ML_el_t: {round(ml_el_t[i],3)}, \
	# 		ML_lfc_t: {round(ml_lfc_t[i],3)}, MLCIN: {round(mlcin[i],3)}')
	return composite

def create_boxplot(windspeeds,variable,title,ylabel,savename):
	print(f'Correlation between Max Wind and {ylabel}:')
	print(np.corrcoef(df["max_wind"],variable)[0][1])
	moderate_wind_thres = 20.0
	severe_wind_thres = 26.0
	wind_boxes = {
		"Weak": [],
		"Moderate": [],
		"Severe": [],
	}

	for i,wind in enumerate(windspeeds):
		#print(wind,variable[i])
		if wind >= severe_wind_thres:
			wind_boxes["Severe"].append(variable[i])
		elif wind >= moderate_wind_thres:
			wind_boxes["Moderate"].append(variable[i])
		else:
			wind_boxes["Weak"].append(variable[i])

	data, keys = wind_boxes.values(), wind_boxes.keys()
	labels = []
	for i,label in enumerate(keys): 
		labels.append(label + f'\nn = {len(wind_boxes[label])}')
	plt.figure()
	plt.boxplot(data)
	plt.xticks(range(1, len(labels) + 1), labels)
	plt.title(title)
	plt.ylabel(ylabel)
	plt.savefig(os.path.join(save_path,savename))
	#plt.show()
	plt.close()

df = pd.read_csv(env_csv_path)

# # Some cool stats model
# print(df.columns)
# variable_string = "max_wind ~ LI_700 + surface_temp + dewpoint_depression + LR3km + max_wind_03km + \
# 					MLCIN + ml_el_p + ml_el_t + ml_lfc_t"
# model = ols(variable_string,data=df).fit()
# print(model.params)
# print(model.summary())


composite = composite_param(df)
composite_savename = "wind_composite.png"
create_boxplot(df["max_wind"],composite,f'Max Wind vs. Composite','Composite',composite_savename)
# for i in range(2,len(df.columns)):
# 	variable = df.columns[i]
# 	savename = f'winds_{variable}.png'
# 	create_boxplot(df["max_wind"],df[variable],f'Max Wind vs. {variable}',variable,savename)

