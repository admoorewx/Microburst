import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from convective_functions import *
from metpy.units import units

wind_damage_csv = "/home/andrew/Research/microburst/wind_damage_reports.csv"
save_path = "/home/andrew/Research/microburst/images/"
sounding_csv = "/home/andrew/Research/microburst/sounding_environments.csv"

# Import the data
wind_damage_data = pd.read_csv(wind_damage_csv)
sounding_env_data = pd.read_csv(sounding_csv)

## Wind Damage/LSR data
# Get the estimate wind speeds 
estimated_speeds = wind_damage_data.iloc[:,1]
# Get the lats
lats = wind_damage_data.iloc[:,8]
lons = wind_damage_data.iloc[:,9]
# Get the descriptions 
descriptions = wind_damage_data.iloc[:,10]
# # Apply correction 
# correction_factor = 0.8
# estimated_speeds = estimated_speeds * correction_factor 
# # Convert to m/s
estimated_speeds = estimated_speeds * 0.51444
# Need to match the report with the corresponding sounding
# Soundings are in the form of: 20180123_0400_38.440_-82.630
years = wind_damage_data.iloc[:,3]
months = wind_damage_data.iloc[:,4]
days = wind_damage_data.iloc[:,5]

### CM1 Wind data
events = list(sounding_env_data.iloc[:,0])
modeled_winds = list(sounding_env_data.iloc[:,1])

# Match LSR and CM1 data
lsr_winds = []
cm1_winds = []
for i,event in enumerate(events):
	year = int(event[0:4])
	month = int(event[4:6])
	day = int(event[6:8])
	hour = int(event[9:11])
	lat = float(event[14:20])
	lon = float(event[21:])

	inds = np.where(np.asarray(days) == day)[0]
	for ind in inds: 
		if round(lats[ind],2) == round(lat,2):
			if round(lons[ind],2) == round(lon,2):
				if days[ind] == day:
					if months[ind] == month:
						if years[ind] == year:
							lsr_winds.append(estimated_speeds[ind])
							cm1_winds.append(modeled_winds[i])
							# print(event)
							# print(years[ind], months[ind], days[ind], lats[ind], lons[ind])
							# print("")


diffs = np.array(lsr_winds) - np.array(cm1_winds)
bias = np.mean(diffs)
std = np.std(diffs)
print(f"Bias: {bias}")
print(f'Std.: {std}')
print(f'Correlation: {np.corrcoef(lsr_winds,cm1_winds)[0][1]}')

over = []
under = []
close = []
window = 2.5
for i,wind in enumerate(lsr_winds): 
	if wind > 25.7: 
		diff = wind - cm1_winds[i]
		if abs(diff) < window: 
			close.append(diff)
		else: 
			if diff > 0: 
				over.append(diff)
			else: 
				under.append(diff)

total = len(over) + len(under) + len(close)
print(f'Percent within +/- 5 knots: {100.0*len(close)/total}')
print(f'Percent Under-Estimated by 5+ knots: {100.0*len(under)/total}')
print(f'Percent Over-Estimated by 5+ knots: {100.0*len(over)/total}')
print(f'Average Under-Estimation: {np.mean(under)/0.51444}')
print(f'Average Over-Estimation: {np.mean(over)/0.51444}')



plt.figure()
plt.scatter(lsr_winds,cm1_winds)
plt.axhline(y=25.7,color='k')
plt.axvline(x=25.7,color='k')
plt.title("LSR Estimated Speed vs. Max CM1 Modeled Wind Speed")
plt.xlabel("LSR Estimated Wind (m/s)")
plt.ylabel("CM1 Modeled Wind (m/s)")
plt.savefig(os.path.join(save_path,"lsr_cm1_wind_compare.png"))

