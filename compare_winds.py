import os
import numpy as np
from netCDF4 import Dataset
import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import pandas as pd



# Schooner paths
# data_directory = "/scratch/admoore/cm1out/"
# save_directory = "/scratch/admoore/images/"

#PC Paths
valid_soundings_path = "E:/Microburst/valid_soundings.csv"
data_directory1 = "D:/Microburst/cm1out/"
data_directory2 = "G:/Microburst/cm1out/"
data_directory3 = "E:/Microburst/cm1out/"
save_directory = "E:/Microburst/images/analysis/"

ncfile_base = "cm1out"

colors = ["firebrick","sienna","sandybrown","goldenrod","gold","lime","green","darkgreen","darkcyan","darkturquoise","dodgerblue",
          "blue","navy","darkorchid","palevioletred","crimson","lightcoral","orange","lightgray","dimgray","lawngreen","khaki",
          "cyan","crimson","slateblue","yellow", "olive", "darkred", "indigo", "lightpink","firebrick","sienna","sandybrown","goldenrod",
          "lime","darkcyan"]

linestyles = ["-","--"]

def get_valid_soundings(path):
    sounding_list = pd.read_csv(path)
    return sounding_list.values[:, 0]

def get_max_wind(ncfile):
    nc = Dataset(ncfile)
    # get time of data
    time = nc.variables["time"][0]
    # get winds
    maxwspd = nc.variables["sws"][:][0]
    linearmx = np.asarray(maxwspd).flatten()
    peak_max = np.max(linearmx)
    return [int(time), peak_max]

def plot_peak_winds(winds):
    over15 = 0
    over22 = 0
    over20 = 0
    over26 = 0
    plt.figure(figsize=(12, 9))
    for key,value in winds.items():
        i = list(winds.keys()).index(key)
        # This was the old way of assigning line styles
        # if i > 15:
        #     l = 1
        # else:
        #     l = 0
        xmax = value[-1,0]
        wm = np.nanmax(value[:,1])
        l = 1
        color = "slategray" # Establish a base color
        # print(key,wm)
        # Tally up and color by wind maximum (wm)
        if wm > 15.0:
            over15  = over15 + 1
            color = "steelblue"
        if wm > 20.0:
            over20 = over20 + 1
            color = "dodgerblue"
        if wm > 22.0:
            over22 = over22 + 1
            color = "mediumblue"
        if wm > 26.0:
            over26 = over26 + 1
            color = "indigo"
            l = 0

        plt.plot(value[:, 0], value[:, 1], linestyle=linestyles[l], color=color, label=key)
    plt.xlabel("Model integration time (s)",fontsize=25)
    plt.ylabel("Peak Wind Speed (m/s)",fontsize=25)
    plt.title(f'10 Meter Max Wind',fontsize=30)
    plt.xlim(0,xmax)
    plt.ylim(0,40)
    plt.grid(alpha=0.5)
    # For now, not showing a legend since there are SO MANY (100) lines being drawn. But here's the code for future reference:
    # legend = plt.legend(bbox_to_anchor=(1,1),loc='upper left')
    # plt.gcf().canvas.draw()
    # invFigure = plt.gcf().transFigure.inverted()
    # legend_pos = legend.get_window_extent()
    # legend_coord = invFigure.transform(legend_pos)
    # legend_xmax = legend_coord[1,0]
    # ax_pos = plt.gca().get_window_extent()
    # ax_coord = invFigure.transform(ax_pos)
    # ax_xmax = ax_coord[1, 0]
    # shift = 1 - (legend_xmax - ax_xmax)
    # plt.gcf().tight_layout(rect=(0, 0, shift, 1))

    plt.savefig(save_directory+"cm1out_pkwinds.png")
    #plt.show()
    plt.close()
    print("Some Max wind stats:")
    print(f'{over15} over 15 m/s')
    print(f'{over20} over 20 m/s')
    print(f'{over22} over 22 m/s')
    print(f'{over26} over 26 m/s')

def max_winds_from_run(path):
    wind_data = []
    ncfile_base = "cm1out"
    for filename in os.listdir(path):
        if ncfile_base in filename:
            try:
                path = os.path.join(subpath, filename)
                wind_data.append(get_max_wind(path))
            except:
                print("\nWARNING: could not get wind data for file: " + path)
                print("\n")
    return wind_data

def wind_stats(peak_winds):
    pk_winds = []
    for key, value in peak_winds.items():
        winds = value[:, 1]
        pk_winds.append(np.nanmax(winds))
    plt.figure()
    plt.hist(pk_winds, bins=np.arange(0,35,1), range=(0,35), align='mid')
    plt.xlabel("Peak Wind Speed (m/s)")
    plt.ylabel("Frequency")
    title = f'CM1 Peak Wind Distribution\nMean: {round(np.nanmean(pk_winds),2)}\nStandard Deviation: {round(np.std(pk_winds),2)}'
    plt.title(title)
    plt.savefig(save_directory + "cm1out_pkwind_dist.png")




peak_winds = {}
total = 0
valid_soundings = get_valid_soundings(valid_soundings_path)
for run in valid_soundings:
    run = run[0:-4] # remove the ".snd" from the end of the filename
    print(f'Finding run {run}...')
    try:
        # Try the first path
        try:
            subpath = os.path.join(data_directory1, run)
            wind_data = max_winds_from_run(subpath)
        except:
            # Try the second path
            try:
                subpath = os.path.join(data_directory2, run)
                wind_data = max_winds_from_run(subpath)
            except:
                # Try the third path
                subpath = os.path.join(data_directory3, run)
                wind_data = max_winds_from_run(subpath)
        # After we found the right path, append the data to the dictionary
        peak_winds[run] = np.sort(wind_data, axis=0)
        total = total + 1
        print("Run found.")
    # If there was a problem:
    except:
        print(f'WARNING: Could not use run {run}')

print("Plotting...")
plot_peak_winds(peak_winds)
wind_stats(peak_winds)
print(f'{total} runs included.')
print("!!! Success !!!")
