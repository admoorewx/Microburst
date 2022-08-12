import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from convective_functions import *
from metpy.units import units

wind_damage_csv = "/home/andrew/Research/microburst/wind_damage_reports.csv"
save_path = "/home/andrew/Research/microburst/images/"
sounding_directory = "/home/andrew/Research/microburst/soundings/"

def round_hour(time):
	string_time = str(time)
	if len(string_time) == 3: 
		string_time = string_time + "0"
	string_hour = string_time[0:2]
	string_min = string_time[2:]
	if int(string_min) >= 30:
		hour = int(string_hour) + 1 
		if hour >= 24:
			hour = 0 
		string_hour = str(hour).zfill(2)
	return string_hour + "00"


def create_boxplot(windspeeds,variable,title,ylabel,savename):
	print(f'Correlation between Max Wind and {ylabel}:')
	print(np.corrcoef(windspeeds,variable)[0][1])
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


# Import the data
wind_damage_data = pd.read_csv(wind_damage_csv)

# Get the estimate wind speeds 
estimated_speeds = wind_damage_data.iloc[:,1]

# Get the lats
lats = wind_damage_data.iloc[:,8]
lons = wind_damage_data.iloc[:,9]

# Get the descriptions 
descriptions = wind_damage_data.iloc[:,10]

# Apply correction 
correction_factor = 0.8
estimated_speeds = estimated_speeds * correction_factor 
# Convert to m/s
estimated_speeds = estimated_speeds * 0.51444

# Need to match the report with the corresponding sounding
# Soundings are in the form of: 20180123_0400_38.440_-82.630
years = wind_damage_data.iloc[:,3]
months = wind_damage_data.iloc[:,4]
days = wind_damage_data.iloc[:,5]
hours = wind_damage_data.iloc[:,6]

hours = [int(str(hour)[0:2]) for hour in hours]

sounding_wind = {}

for filename in os.listdir(sounding_directory):
	year = int(filename[0:4])
	month = int(filename[4:6])
	day = int(filename[6:8])
	hour = int(filename[9:11])
	lat = float(filename[14:20])
	lon = float(filename[21:-4])

	inds = np.where(np.asarray(days) == day)[0]
	for ind in inds: 
		if round(lats[ind],2) == round(lat,2):
			if round(lons[ind],2) == round(lon,2):
				if days[ind] == day:
					if months[ind] == month:
						if years[ind] == year:
							sounding_wind[filename] = estimated_speeds[ind]
							print(filename)
							print(years[ind], months[ind], days[ind], hours[ind], lats[ind], lons[ind])
							print("")


winds = []
params = []
for key,value in sounding_wind.items():
	# Read in the .snd file, return a dataframe
	try:
		df = read_snd(os.path.join(sounding_directory,key))
		if df.empty == False:
			# Pass the dataframe to the composite parameter function
			comp_param = composite_param_from_sounding(df)
			winds.append(value)
			params.append(comp_param)
	except:
		print(f'WARNING: Error processing sounding: {key}')


title = f'Estimated LSR Wind Speed vs. Composite Parameter\n(n = {len(winds)})'
ylabel = "Composite Parameter"

plt.figure()
plt.scatter(winds,params)
plt.xlabel("Estimated LSR Wind Speed (m/s)")
plt.ylabel(ylabel)
plt.title(title)
plt.savefig(os.path.join(save_path,"comp_param_wind_speed_bulk.png"))

savename = "winds_vs_comp_param_boxplot.png"
create_boxplot(winds,params,title,ylabel,savename)
