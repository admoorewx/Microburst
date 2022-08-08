import os
import numpy as np
from netCDF4 import Dataset
import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from metpy.plots import StationPlot
from metpy.units import units
from collections import OrderedDict

runname = "20190624_1900_30.790_-85.680"

# Schooner paths
# data_directory = "/scratch/admoore/cm1out/CHS_0628202000/"
# save_directory = "/scratch/admoore/images/CHS_0628202000/"

# PC paths
data_directory = f'E:/Microburst/cm1out/{runname}/'
save_directory = f'E:/Microburst/images/{runname}/'

# Tunable params
title = runname
title_font_size = 14
ncfile_base = "cm1out"


def plot_wind(ncfile):
    nc = Dataset(ncfile)
    var = nc.variables["wspd"][:][0]
    refl = nc.variables["cref"][:][0]
    xx = np.linspace(0,len(var),len(var))
    yy = np.linspace(0,len(var),len(var))
    X, Y = np.meshgrid(xx,yy)

    # get time of data
    time = nc.variables["time"][0]

    # get the wind arrow data
    skip = 25
    barb_points_x = X[::skip,::skip]
    barb_points_y = Y[::skip,::skip]
    u_wind = nc.variables['u10'][:][0][::skip,::skip] 
    u_wind = np.array(u_wind, dtype=float) * units.meter/units.second
    v_wind = nc.variables['v10'][:][0][::skip,::skip] 
    v_wind = np.array(v_wind, dtype=float) * units.meter/units.second

    # set limts and interval of Reflectivity (dBz)
    dbz_thres = 35.0
    refl_levels = [dbz_thres]

    # create the figure
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(1,1,1)

    # plot the wind arrows
    stationplot = StationPlot(ax, barb_points_x, barb_points_y)
    stationplot.plot_arrow(u_wind,v_wind)

    # plot the wind speed
    var_levels = np.arange(0,40.0,1.0)
    plt.contourf(X,Y, var, levels=var_levels, cmap=get_cmap("jet"), alpha=0.5)

    # Add a color bar
    cbar = plt.colorbar()
    cbar.set_label("Wind Magnitude (m/s)")

    # Add the refl contour - but only if max dbz is >= 35!
    flatrefl = np.asarray(refl).flatten()
    maxdbz = np.max(flatrefl)
    if maxdbz >= 35.0:
        plt.contour(X, Y, refl, levels=refl_levels, colors='white', alpha=1.0)

    # title and save
    plt.title(f'{title} 10-meter Wind Field (t={time} seconds)', fontsize=title_font_size)
    plt.savefig(save_directory+"cm1out_wind_sfc_"+str(int(time))+".png")
    #plt.show()
    plt.close(fig)

def plot_wind_swath(ncfile):
    nc = Dataset(ncfile)
    var = nc.variables["sws"][:][0]
    xx = np.linspace(0, len(var), len(var))
    yy = np.linspace(0, len(var), len(var))
    X, Y = np.meshgrid(xx, yy)

    # get time of data
    time = nc.variables["time"][0]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1)
    var_levels = np.arange(0, 40.0, 1.0)
    plt.contourf(X, Y, var, levels=var_levels, cmap=get_cmap("jet"), alpha=0.5)

    # Add a color bar
    cbar = plt.colorbar()
    cbar.set_label("Wind Magnitude (m/s)")

    # Add a contour based on set wind threshold
    wind_thres = 26.0
    flatvar = np.asarray(var).flatten()
    maxwind = np.max(flatvar)
    if maxwind >= wind_thres:
        plt.contour(X, Y, var, levels=[wind_thres], colors='k', linewidths=0.5, alpha=1.0)

    plt.title(f'{title} Max 10-meter Wind Swath (t={time} seconds)', fontsize=title_font_size)
    #plt.grid(color='k',alpha=0.9)
    plt.savefig(save_directory + "cm1out_sws_" + str(int(time)) + ".png")
    # plt.show()
    plt.close(fig)

def plot_sfc_pres(ncfile):
    nc = Dataset(ncfile)
    var = nc.variables["psfc"][:][0] / 100.0 # convert to mb
    xx = np.linspace(0,len(var),len(var))
    yy = np.linspace(0,len(var),len(var))
    X, Y = np.meshgrid(xx,yy)

    # get time of data
    time = nc.variables["time"][0]

    plt.figure(figsize=(12,9))

    # set limts and interval (mb)
    minPres = 980.0
    maxPres = 1020.0
    interval = 1.0

    var_levels = np.arange(minPres,maxPres,interval)
    plt.contourf(X,Y, var, levels=var_levels, cmap=get_cmap("coolwarm"), alpha=0.5)
    #plt.contourf(X,Y,var,cmap=get_cmap("jet"),alpha=0.5)

    # Add a color bar
    cbar = plt.colorbar()
    cbar.set_label("Sfc Pressure (mb)")

    plt.title(f'{title} SFC Pressure (t={time} seconds)', fontsize=title_font_size)
    plt.savefig(save_directory+"cm1out_psfc_"+str(int(time))+".png")
    #plt.show()
    plt.close()

def plot_cape(ncfile):
    nc = Dataset(ncfile)
    cape = nc.variables["cape"][:][0]
    xx = np.linspace(0,len(cape),len(cape))
    yy = np.linspace(0,len(cape),len(cape))
    X, Y = np.meshgrid(xx,yy)

    # get time of data
    time = nc.variables["time"][0]

    plt.figure(figsize=(12,9))

    # set limts and interval (dBz)
    cape_interval = 100.0
    minCape = 0.0
    maxCape = 4000.0

    cape_levels = np.arange(minCape,maxCape,cape_interval)
    plt.contourf(X,Y, cape, levels=cape_levels, cmap=get_cmap("turbo"), alpha=0.5)

    # Add a color bar
    cbar = plt.colorbar()
    cbar.set_label("CAPE (J/kg)")

    plt.title(f'{title} SB CAPE (t={time} seconds)', fontsize=title_font_size)
    plt.savefig(save_directory+"cm1out_cape_"+str(int(time))+".png")
    #plt.show()
    plt.close()

def plot_cin(ncfile):
    nc = Dataset(ncfile)
    cin = nc.variables["cin"][:][0]
    xx = np.linspace(0,len(cin),len(cin))
    yy = np.linspace(0,len(cin),len(cin))
    X, Y = np.meshgrid(xx,yy)

    # get time of data
    time = nc.variables["time"][0]

    plt.figure(figsize=(12,9))

    # set limts and interval
    cin_interval = 10.0
    minCin = 0.0
    maxCin = 500.0

    cin_levels = np.arange(minCin, maxCin, cin_interval)
    plt.contourf(X, Y, cin, levels=cin_levels, cmap=get_cmap("turbo"), alpha=0.5)

    # Add a color bar
    cbar = plt.colorbar()
    cbar.set_label("CIN (J/kg)")

    plt.title(f'{title} SB CIN (t={time} seconds)', fontsize=title_font_size)
    plt.savefig(save_directory+"cm1out_cin_"+str(int(time))+".png")
    #plt.show()
    plt.close()

def plot_cref(ncfile):
    nc = Dataset(ncfile)
    var = nc.variables["cref"][:][0] 
    xx = np.linspace(0,len(var),len(var))
    yy = np.linspace(0,len(var),len(var))
    X, Y = np.meshgrid(xx,yy)

    # get time of data
    time = nc.variables["time"][0]

    plt.figure(figsize=(12,9))

    # set limts and interval (dBz)
    minPres = 0.0
    maxPres = 95.0
    interval = 0.5 

    var_levels = np.arange(minPres,maxPres,interval)
    plt.contourf(X,Y, var, levels=var_levels, cmap=get_cmap("gist_ncar"), alpha=0.5)

    # Add a color bar
    cbar = plt.colorbar()
    cbar.set_label("Reflectivity (dBZ)")

    plt.title(f'{title} SFC Composite Reflectivity (t={time} seconds)', fontsize=title_font_size)
    plt.savefig(save_directory+"cm1out_cref_"+str(int(time))+".png")
    #plt.show()
    plt.close()

def get_max_wind(ncfile):
    nc = Dataset(ncfile)
    # get time of data
    time = nc.variables["time"][0]
    # get winds
    maxwspd = nc.variables["sws"][:][0]
    length = len(maxwspd)
    width = len(maxwspd[0])
    num_elements = length * width
    linearmx = np.asarray(maxwspd).flatten()
    peak_max = np.max(linearmx)


    # above_10 = np.where(linear > 10.0)[0]
    # print(above_10)
    #print((len(above_10)/num_elements) * 100.0)

    return [int(time), peak_max]

def plot_peak_winds(winds):

    plt.figure(figsize=(12,9))
    plt.plot(winds[:,0],winds[:,1],color='k', label="Peak WSPD")
    plt.xlabel("Model integration time (s)")
    plt.ylabel("Peak Wind Speed (m/s)")
    plt.title(f'{title} Peak Winds', fontsize=title_font_size)
    plt.xlim(0,winds[-1,0])
    plt.ylim(0,25)
    plt.grid(alpha=0.5)
    plt.savefig(save_directory+"cm1out_pkwind.png")
    #plt.show()
    plt.close()


peak_winds = []
for filename in os.listdir(data_directory):
    if ncfile_base in filename:
        try:
            #peak_winds.append(get_max_wind(os.path.join(data_directory,filename)))
            # plot_sfc_pres(os.path.join(data_directory,filename))
            #plot_wind(os.path.join(data_directory,filename))
            #plot_wind_swath(os.path.join(data_directory, filename))
            plot_cref(os.path.join(data_directory,filename))
            # plot_cape(os.path.join(data_directory, filename))
            # plot_cin(os.path.join(data_directory, filename))
        except:
            print("\nWARNING: could not create images for file: "+filename)
            print("\n")

# peak_winds = np.sort(peak_winds,axis=0)
# plot_peak_winds(peak_winds)
