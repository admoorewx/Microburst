from datetime import datetime, timedelta
import metpy
from metpy.units import units
from siphon.simplewebservice.wyoming import WyomingUpperAir
import matplotlib.pyplot as plt
import matplotlib.colors as Colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import metpy.plots as mpplots
import metpy.calc as mcalc
from netCDF4 import Dataset
import numpy as np
import os, sys

# Schooner paths
# data_directory = "/scratch/admoore/cm1out/CHS_0628202000/"
# save_directory = "/scratch/admoore/images/CHS_0628202000/"

# PC Paths
data_directory = "D:/Microburst/cm1out/"
save_directory = "D:/Microburst/images/"


def getData(ncfile):
    nc = Dataset(ncfile)
    # get valid time of the data file
    valid_time = nc.variables["time"][0]
    # get the center point of the grid (just to start)
    x = nc.variables["xh"]
    xmid = int(len(x)/2.0)
    y = nc.variables["yh"]
    ymid = int(len(y) / 2.0)
    # Get the data from this point
    theta = np.asarray(nc.variables["th"][0][:,ymid,xmid]) * units.kelvin
    pres = np.asarray(nc.variables["prs"][0][:,ymid,xmid]) / 100.0 * units.hPa # originally in Pa
    height = np.asarray(nc.variables["zh"][:]) * 1000.0 * units.meter # originally in km
    mixr = np.asarray(nc.variables["qv"][0][:,ymid,xmid]*1000.0) * units('g/kg') # originally in kg/kg
    uwind = np.asarray(nc.variables["u"][0][:,ymid,xmid])
    vwind = np.asarray(nc.variables["v"][0][:,ymid,xmid])

    # convert to traditional variables
    temp = mcalc.temperature_from_potential_temperature(pres,theta)
    e = mcalc.vapor_pressure(pres,mixr)
    dewp = mcalc.dewpoint(e)
    # convert from m/s to knots
    uwind = uwind * 1.943844 * units.knots
    vwind = vwind * 1.943844 * units.knots

    # return the data
    return temp, dewp, height, pres, uwind, vwind, valid_time

def getParcelInfo(type,p,t,td):
    if type == "mixed-layer":
        # define depth of ML parcel
        depth = 100.0 * units.hPa
        top_of_layer = (p[0] - depth) # get the top of the layer
        mid_layer = (p[0] - (depth/2.0)) # get the mid-point of the layer
        inds = np.where(p >= top_of_layer) # get all values within the layer
        temp = np.mean(t[inds]) # find the average temp
        dewp = np.mean(td[inds]) # find the average dewp
        inds = np.where(p <= mid_layer) # get the profile above the mid-layer point
        p = p[inds]
        t = t[inds]
        td = td[inds]
        p = np.insert(p,0,mid_layer) # add in the mid-level point so we can lift from this point
        t = np.insert(t,0,temp)
        td = np.insert(td,0,dewp)
        parcel_path = mcalc.parcel_profile(p,temp,dewp)
    elif type == "surface":
        parcel_path = mcalc.parcel_profile(p, t[0], td[0])
    elif type == "most-unstable":
        thetae = mcalc.equivalent_potential_temperature(p, t, td)
        ind = np.where(np.nanmax(thetae.magnitude))[0][0]
        parcel_path = mcalc.parcel_profile(p, t[ind], td[ind])
        # for i in range(0,len(parcel_path)):
        #     print(i,p[i],t[i],td[i],parcel_path[i])
    else:
        print("ERROR: unkown parcel-type. Options are 'most-unstable', 'mixed-layer' or 'surface'.")
    # calculate the LCL, LFC, and EL
    lcl_p, lcl_t = mcalc.lcl(p[0],t[0],td[0])
    lfc_p, lfc_t = mcalc.lfc(p,t,td)
    el_p, el_t = mcalc.el(p,t,td)
    # find cape/cin
    cape, cin = mcalc.cape_cin(p,t,td, parcel_path)
    return parcel_path, p, t, lcl_p, lcl_t, lfc_p, lfc_t, el_p, el_t, cape, cin

def virtualTemp(p,t,td):
    if t.units != units('degC'):
        t = t.to('degC')
        td = td.to('degC')
    if p.units != units('hPa'):
        p = p.to('hPa')
    # es = mcalc.saturation_vapor_pressure(t)
    e = mcalc.saturation_vapor_pressure(td)
    w = ((621.97 * (e/(p-e)))/1000.0).to('g/kg')
    return mcalc.virtual_temperature(t,w).to('degC')

def get_effective_bulk_shear(p,t,td,u,v):
    # get the effective inflow layer
    eff_layer = effective_inflow_layer(p,t,td)
    if len(eff_layer) > 0:
        # get the bottom of the effective inflow layer
        eff_bottom = eff_layer[0]
        # get the most-unstable parcel profile
        thetae = mcalc.equivalent_potential_temperature(p,t,td)
        ind = np.where(np.nanmax(thetae.magnitude))[0]
        parcel_path = mcalc.parcel_profile(p,t[ind],td[ind])
        # need to get half the height of the mu parcel path and then find bulk shear
        # Find the EL pressure/temp
        el_p, el_t = mcalc.el(p,t,td,parcel_path)
        # if the el_p is NaN, then the actual EL is above the sounding data (i.e., there's missing data at the top of the sounding)
        # just set the el_p and el_t to the top most pres/temp
        if np.isnan(el_p):
            el_p = p[-1]
            el_t = t[-1]
        # get the index of the EL pres - do this by finding the smallest diff between P levels and EL_P
        diffs = [np.abs(P.magnitude - el_p.magnitude) for P in p]
        el_ind = diffs.index(np.min(diffs))
        # get the index of the bottom of the eff. inflow layer
        eff_ind = np.where(p == eff_bottom)[0][0]
        # get the depth of the layer
        depth = p[eff_ind] - p[el_ind]
        # find the eff bulk wind shear
        us,vs = mcalc.bulk_shear(p,u,v,bottom=p[eff_ind],depth=depth)
        shear_mag = mcalc.wind_speed(us,vs)
        return shear_mag.magnitude # Gives the shear in knots
    else:
        return 0.0 # Return an effective bulk wind shear value of 0.0 knots if there is not eff. layer

def effective_inflow_layer(p,t,td):
    tv = virtualTemp(p,t,td)
    min_cape = 100.0 * units('J/kg')
    min_cin = -250.0 * units('J/kg')
    parcel_paths = [mcalc.parcel_profile(p,tv[i],td[i]) for i in range(0,len(p))]
    inds = []
    for path in parcel_paths:
        cape,cin = mcalc.cape_cin(p,tv,td,path)
        if cape >= min_cape and cin >= min_cin:
            inds.append(np.where(tv == path[0])[0][0])
    return p[inds]

def getDiff(pres,thetae):
    # Just get the numbers:
    pres = [p.magnitude for p in pres]
    pres = np.asarray(pres)
    # get the max temp in the lowest X mb:
    top_lower_layer = 700.0 # mb - Top of the bottom layer
    lower_layer_inds = [np.where(pres >= top_lower_layer)][0]
    max_temp = np.nanmax(thetae[lower_layer_inds])

    # get the min temp in the upper portion
    top_upper_layer = 100.0 # mb - where to stop looking for the min temp
    upper_layer_inds = [np.where(pres >= top_upper_layer)][0]
    #upper_layer_inds = [np.where(pres[upper_layer_inds] <= top_lower_layer)][0] # make sure to exclude the lower layer
    min_temp = np.nanmin(thetae[upper_layer_inds])

    diff = max_temp - min_temp
    return diff

def plotSounding(ncfile):
    # call getData
    t, td, hght, p, u, v, valid_time = getData(ncfile)

    # data manipulation
    mask = p >= 100 * units.hPa

    wind_thin = 3 # thinning for wind barbs
    interval = np.asarray([100,150,200,250,300,350,400,450,500,550,600,625,650,675,700,750,800,825,850,875,900,925,950,975,1000]) * units.hPa
    idx = mcalc.resample_nn_1d(p,interval)
    wspds = mcalc.wind_speed(u[idx],v[idx])

    # Get bulk shear values - returns U and V components
    zero1BS = mcalc.bulk_shear(p, u, v, hght, depth=1000.0 * units.meter,bottom=hght[0])
    zero3BS = mcalc.bulk_shear(p,u,v,hght,depth=3000.0*units.meter,bottom=hght[0])
    zero6BS = mcalc.bulk_shear(p, u, v, hght, depth=6000.0 * units.meter,bottom=hght[0])
    effBS = get_effective_bulk_shear(p,t,td,u,v) # This returns the shear magnitude already
    # Note: the output unit says it's in m/s, but comparing to the SPC archived soundings, it
    # appears the magnitudes are actually in knots. Worth checking on again in the future.
    zero1BS = mcalc.wind_speed(zero1BS[0],zero1BS[1])
    zero3BS = mcalc.wind_speed(zero3BS[0], zero3BS[1])
    zero6BS = mcalc.wind_speed(zero6BS[0], zero6BS[1])

    # Find the theta-e Deficit:
    thetae = mcalc.equivalent_potential_temperature(p, t, td)
    theta_deficit = getDiff(p,thetae)


    # Establish figure
    fig = plt.figure(figsize=(12,12),tight_layout=True)
    skew = mpplots.SkewT(fig,rotation=45)

    # Get virtual temperature
    tv = virtualTemp(p,t,td)

    # get surface parcel info
    parcelType = "surface"
    sb_parcel_path, sb_parcel_p, sb_parcel_t, sb_lcl_p, sb_lcl_t, sb_lfc_p, sb_lfc_t, sb_el_p, sb_el_t, sb_cape, sb_cin = getParcelInfo(parcelType,p,tv,td)

    # get mixed-layer parcel info
    parcelType = "mixed-layer"
    ml_parcel_path, ml_parcel_p, ml_parcel_t, ml_lcl_p, ml_lcl_t, ml_lfc_p, ml_lfc_t, ml_el_p, ml_el_t, ml_cape, ml_cin = getParcelInfo(parcelType,p,tv,td)

    # get most unstable parcel info
    parcelType = "most-unstable"
    mu_parcel_path, mu_parcel_p, mu_parcel_t, mu_lcl_p, mu_lcl_t, mu_lfc_p, mu_lfc_t, mu_el_p, mu_el_t, mu_cape, mu_cin = getParcelInfo(parcelType,p,tv,td)

    # Plot data
    skew.plot(p,t,'r')
    skew.plot(p,td,'g')
    skew.plot(p,tv,'darkred',linestyle='--',alpha=0.5)
    skew.plot_barbs(p[idx], u[idx], v[idx], length=6, barbcolor="k", xloc=1.05)
    # plot parcel path
    skew.plot(mu_parcel_p,mu_parcel_path,'gray')
    skew.shade_cape(mu_parcel_p,mu_parcel_t,mu_parcel_path)
    #skew.shade_cin(p,tv,ml_parcel_path)

    # Plot settings
    skew.ax.set_ylim(1025,100)
    skew.ax.set_xlim(-30,50)
    skew.ax.set_ylabel("Pressure (hPa)")
    skew.ax.set_xlabel("T/Td (C)")

    # Add the relevant special lines to plot throughout the figure
    skew.plot_dry_adiabats(np.arange(233, 533, 20) * units.K,
                           alpha=0.25, color='orangered')
    skew.plot_moist_adiabats(np.arange(233, 400, 5) * units.K,
                             alpha=0.25, color='navy')
    skew.plot_mixing_lines(alpha=0.35,color='darkgreen')
    skew.ax.axvline(0*units.degC, alpha=0.35, color='cyan')

    # Add a title
    plt.title(f'CM1 Center of Domain at time {valid_time} (s)', loc='left',fontsize=18)


    # Add text info
    parcelText = f'--- Thermodynamic Parameters ---\n' \
                 f'--- CAPE (J/kg) | CIN (J/kg)\n' \
                 f'SB: {round(sb_cape.magnitude, 2)} | {round(sb_cin.magnitude,2)}\n' \
                 f'ML: {round(ml_cape.magnitude, 2)} | {round(ml_cin.magnitude, 2)}\n' \
                 f'MU: {round(mu_cape.magnitude, 2)} | {round(mu_cin.magnitude, 2)}\n' \
                 f'Theta-e Deficit: {round(theta_deficit.magnitude, 2)} K\n' \
                 f'\n'\
                 f'--- Shear Parameters ---\n' \
                 f'0-1 km Bulk Shear: {round(zero1BS.magnitude,2)} knots\n' \
                 f'0-3 km Bulk Shear: {round(zero3BS.magnitude,2)} knots\n' \
                 f'0-6 km Bulk Shear: {round(zero6BS.magnitude,2)} knots\n' \
                 f'Eff. Bulk Shear: {round(effBS,2)} knots'

    axt = inset_axes(skew.ax, '100%', '30%', loc=3)
    #axt.text(3.5,1.5,parcelText,fontsize=13)
    axt.text(0.0, 0.0, parcelText, fontsize=13)
    axt.axis('off')


    # insert the hodograph
    ax_hod = inset_axes(skew.ax, '30%', '30%', loc=1)
    h = mpplots.Hodograph(ax_hod,component_range=60.)
    h.add_grid(increment=10)
    hodo_inds = np.where(p >= 100.0*units.hPa)[0]
    h.plot_colormapped(u[hodo_inds],v[hodo_inds],hght[hodo_inds],cmap=plt.get_cmap('jet'),norm=Colors.Normalize(vmin=0.0,vmax=12000.0))

    plt.tight_layout()
    plt.savefig(os.path.join(save_directory,ncfile[-16:-3]+"_snd.png"))
    #plt.show()

for filename in os.listdir(data_directory):
    if filename.endswith(".nc"):
        plotSounding(os.path.join(data_directory,filename))