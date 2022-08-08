# Upper Air
from datetime import datetime, timedelta
from metpy.units import units
import matplotlib.pyplot as plt
import matplotlib.colors as Colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from convective_functions import *
import metpy.plots as mpplots
import metpy.calc as mcalc
import numpy as np
import pandas as pd
import os

csv_dir = "D:/Microburst/soundings/csv/"
save_dir = "D:/Microburst/images/soundings/"

def plotSounding(csv_dir,csvfile):
    # call getData
    file_loc = os.path.join(csv_dir,csvfile)
    p,t,td,u,v,hght = data_from_csv(file_loc,wind_units='knot')

    # data manipulation
    mask = p >= 100 * units.hPa

    wind_thin = 3 # thinning for wind barbs
    interval = np.asarray([100,150,200,250,300,350,400,450,500,550,600,625,650,675,700,750,800,825,850,875,900,925,950,975,1000]) * units.hPa
    idx = mcalc.resample_nn_1d(p,interval)

    # Get bulk shear values - returns U and V components
    zero1BS = mcalc.bulk_shear(p, u, v, hght, depth=1000.0 * units.meter,bottom=hght[0])
    zero3BS = mcalc.bulk_shear(p,u,v,hght,depth=3000.0*units.meter,bottom=hght[0])
    zero6BS = mcalc.bulk_shear(p, u, v, hght, depth=6000.0 * units.meter,bottom=hght[0])
    effBS = get_effective_bulk_shear(p,t,td,u,v,hght) # This returns the shear magnitude already
    # Note: the output unit says it's in m/s, but comparing to the SPC archived soundings, it
    # appears the magnitudes are actually in knots. Worth checking on again in the future.
    zero1BS = mcalc.wind_speed(zero1BS[0],zero1BS[1])
    zero3BS = mcalc.wind_speed(zero3BS[0], zero3BS[1])
    zero6BS = mcalc.wind_speed(zero6BS[0], zero6BS[1])

    # Find the theta-e Deficit:
    thetae = mcalc.equivalent_potential_temperature(p, t, td)
    theta_deficit = thetaE_defficit(p,thetae)

    # Find theta profile:
    theta = get_theta_profile(p,t)

    # Get virtual temperature
    tv = virtualTemp(p,t,td)

    # get surface parcel info
    parcelType = "surface"
    sb_parcel_path = parcel_trajectory(p,tv,td,type=parcelType)
    sb_lcl_p, sb_lcl_t, sb_lfc_p, sb_lfc_t, sb_el_p, sb_el_t = getParcelInfo(parcelType,p,tv,td)
    sbcape, sbcin = getCape(parcelType,p,tv,td)

    # # get mixed-layer parcel info
    # parcelType = "mixed-layer"
    # ml_parcel_path = parcel_trajectory(p,tv,td,type=parcelType)
    # ml_lcl_p, ml_lcl_t, ml_lfc_p, ml_lfc_t, ml_el_p, ml_el_t = getParcelInfo(parcelType,p,tv,td)
    # mlcape, mlcin = getCape(parcelType,p,tv,td)

    # get most unstable parcel info
    parcelType = "most-unstable"
    mu_parcel_path = parcel_trajectory(p,tv,td,type=parcelType)
    mu_lcl_p, mu_lcl_t, mu_lfc_p, mu_lfc_t, mu_el_p, mu_el_t = getParcelInfo(parcelType,p,tv,td)
    mucape, mucin = getCape(parcelType,p,tv,td)

    # Establish figure
    fig = plt.figure(figsize=(12,12),tight_layout=True)
    skew = mpplots.SkewT(fig,rotation=45)

    # Plot data
    skew.plot(p,t,'r')
    skew.plot(p,td,'g')
    skew.plot(p,theta,'b')
    skew.plot(p,tv,'darkred',linestyle='--',alpha=0.5)
    skew.plot_barbs(p[idx], u[idx], v[idx], length=8, barbcolor="k", xloc=1.08)
    # plot parcel path
    skew.plot(p,sb_parcel_path,'gray')
    skew.shade_cape(p,tv,sb_parcel_path)
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
    station = csvfile[0:3]
    date = datetime.strptime(csvfile[4:-4],"%m%d%Y%H")
    plt.title(f'{station} {datetime.strftime(date,"%m/%d/%Y %H UTC")}', loc='left',fontsize=19)

    # Add text info
    parcelText = f'--- Thermodynamic Parameters ---\n' \
                 f'--- CAPE (J/kg) | CIN (J/kg)\n' \
                 f'SB: {round(sbcape.magnitude, 2)} | {round(sbcin.magnitude,2)}\n' \
                 f'MU: {round(mucape.magnitude, 2)} | {round(mucin.magnitude, 2)}\n' \
                 f'Theta-e Deficit: {round(theta_deficit.magnitude, 2)} K\n' \
                 f'\n'\
                 f'--- Shear Parameters ---\n' \
                 f'0-1 km Bulk Shear: {round(zero1BS.magnitude,2)} knots\n' \
                 f'0-3 km Bulk Shear: {round(zero3BS.magnitude,2)} knots\n' \
                 f'0-6 km Bulk Shear: {round(zero6BS.magnitude,2)} knots\n' \
                 f'Eff. Bulk Shear: {round(effBS.magnitude,2)} knots'

#                  f'ML: {round(mlcape.magnitude, 2)} | {round(mlcin.magnitude, 2)}\n' \
    axt = inset_axes(skew.ax, '100%', '30%', loc=3)
    #axt.text(3.5,1.5,parcelText,fontsize=13)
    axt.text(0.0, 0.0, parcelText, fontsize=13)
    axt.axis('off')


    # insert the hodograph
    ax_hod = inset_axes(skew.ax, '30%', '30%', loc=1)
    h = mpplots.Hodograph(ax_hod,component_range=60.)
    h.add_grid(increment=10)
    hodo_inds = np.where(p >= 100.0*units.hPa)[0]
    #hypo = np.hypot(u,v)
    wind_colors = [color_by_height(h) for h in hght]
    h.plot_colormapped(u[hodo_inds], v[hodo_inds], hght[hodo_inds],color=np.asarray(wind_colors))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,station+"_"+datetime.strftime(date,"%m%d%Y%H")+"_snd.png"))
    #plt.show()


print(f'Looking through directory {csv_dir}...')
for filename in os.listdir(csv_dir):
    if filename.endswith(".csv"):
        plotSounding(csv_dir,filename)

# f'--- LCL Pres (hPa) | LCL Temp (C)\n' \
# f'SB: {round(sb_lcl_p.magnitude, 2)} | {round(sb_lcl_t.magnitude, 2)} C\n' \
# f'ML: {round(ml_lcl_p.magnitude, 2)} | {round(ml_lcl_t.magnitude, 2)} C\n' \
# f'MU: {round(mu_lcl_p.magnitude, 2)} | {round(mu_lcl_t.magnitude, 2)} C\n' \
# f'--- LFC Pres (hPa) | LFC Temp (C)\n' \
# f'SB: {round(sb_lfc_p.magnitude, 2)} | {round(sb_lfc_t.magnitude, 2)} C\n' \
# f'ML: {round(ml_lfc_p.magnitude, 2)} | {round(ml_lfc_t.magnitude, 2)} C\n' \
# f'MU: {round(mu_lfc_p.magnitude, 2)} | {round(mu_lfc_t.magnitude, 2)} C\n' \
# f'--- EL Pres (hPa) | EL Temp (C)\n' \
# f'SB: {round(sb_el_p.magnitude, 2)} | {round(sb_el_t.magnitude, 2)} C\n' \
# f'ML: {round(ml_el_p.magnitude, 2)} | {round(ml_el_t.magnitude, 2)} C\n' \
# f'MU: {round(mu_el_p.magnitude, 2)} | {round(mu_el_t.magnitude, 2)} C\n' \