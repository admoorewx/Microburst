import os
import numpy as np
from netCDF4 import Dataset
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import convective_functions as CF
import metpy.plots as mpplots
from metpy.units import units
from scipy import interpolate

sounding_directory = "E:/Microburst/valid_soundings/"
data_directory1 = "D:/Microburst/cm1out/"
data_directory2 = "G:/Microburst/cm1out/"
data_directory3 = "E:/Microburst/cm1out/"
save_directory = "E:/Microburst/images/analysis/"
last_file = "cm1out_000120.nc"
soundings = {}
total = 0

def get_max_wind(path,last_file):
    nc = Dataset(os.path.join(path,last_file))
    # get winds
    maxwspd = nc.variables["sws"][:][0]
    linearmx = np.asarray(maxwspd).flatten()
    peak_max = np.max(linearmx)
    return peak_max

def profile_percentile(soundings,minthres,maxthres,percentile,minheight=0.0,maxheight=16000.0,height_interval=200.0):
    """
    Takes in a number of wind speeds and full profiles in the dictionary 'soundings',
    then sorts them into a profile of all profiles above the wind speed threshold (thres).
    The sorting is determined by the percentile of all profiles at each level.
    Assumes the 'soundings' dict takes the form:
    [run id]: [wind speed, [profile]]
    Each sounding will be interpolated to a standard height profile defined by
    minheight (0 meters by default) and maxheight (16 km by default).
    """
    comp_profile = {}
    standard_height_profile = np.arange(minheight,maxheight,height_interval)
    for key,value in soundings.items():
        # First interpolate the profile to the height profile
        interpfunction = interpolate.interp1d(value[2],value[1],fill_value='extrapolate')
        standardized_data = interpfunction(standard_height_profile)
        if minthres <= value[0] < maxthres:
            for i,val in enumerate(standardized_data):
                if len(comp_profile) < len(standardized_data): # if the comp_profile dict is empty, need to create the keys + value arrays
                    comp_profile[str(i)] = [val]
                else: # if it's not empty, just add as normal
                    comp_profile[str(i)].append(val)
    for ky,valu in comp_profile.items():
        comp_profile[ky] = np.percentile(valu,percentile)
    profile = list(comp_profile.values())
    return profile

def mean_profile(soundings,minthres,maxthres,minheight=0.0,maxheight=16000.0,height_interval=200.0):
    """
    Takes in a number of wind speeds and full profiles in the dictionary 'soundings',
    then sorts them into a mean profile of all profiles above the wind speed threshold (thres).
    Assumes the 'soundings' dict takes the form:
    [run id]: [wind speed, [profile]]
    Each sounding will be interpolated to a standard height profile defined by
    minheight (0 meters by default) and maxheight (16 km by default).
    """
    comp_profile = {}
    standard_height_profile = np.arange(minheight,maxheight,height_interval)
    for key,value in soundings.items():
        # First interpolate the profile to the height profile
        interpfunction = interpolate.interp1d(value[2],value[1],fill_value='extrapolate')
        standardized_data = interpfunction(standard_height_profile)
        if value[0] >= minthres and value[0] < maxthres:
            for i,val in enumerate(standardized_data):
                if len(comp_profile) < len(standardized_data): # if the comp_profile dict is incomplete, need to create the keys + value arrays
                    comp_profile[str(i)] = [val]
                else: # if it's not empty, just add as normal
                    comp_profile[str(i)].append(val)
    for ky,valu in comp_profile.items():
        comp_profile[ky] = np.mean(valu)

    profile = list(comp_profile.values())
    return profile

def median_profile(soundings,minthres,maxthres,minheight=0.0,maxheight=16000.0,height_interval=200.0):
    """
    Takes in a number of wind speeds, data profiles, and corresponding heights (AGL) in the dictionary 'soundings',
    then sorts them into a median profile of all profiles above the wind speed threshold (thres).
    Assumes the 'soundings' dict takes the form:
    [run id]: [wind speed, [data_profile], [height_profile]]
    Each sounding will be interpolated to a standard height profile defined by
    minheight (0 meters by default) and maxheight (16 km by default).
    """
    comp_profile = {}
    standard_height_profile = np.arange(minheight,maxheight,height_interval)
    for key,value in soundings.items():
        # First interpolate the profile to the height profile
        interpfunction = interpolate.interp1d(value[2],value[1],fill_value='extrapolate')
        standardized_data = interpfunction(standard_height_profile)
        if value[0] >= minthres and value[0] < maxthres:
            for i,val in enumerate(standardized_data):
                if len(comp_profile) < len(standardized_data): # if the comp_profile dict is incomplete, need to create the keys + value arrays
                    comp_profile[str(i)] = [val]
                else: # if it's not empty, just add as normal
                    comp_profile[str(i)].append(val)
    for ky,valu in comp_profile.items():
        comp_profile[ky] = np.median(valu)

    profile = list(comp_profile.values())
    return profile

def plot_temp_profiles(soundings,save_directory):
    # Establish figure
    fig = plt.figure(figsize=(12,12),tight_layout=True)
    skew = mpplots.SkewT(fig,rotation=45)
    for key,value in soundings.items():
        # Normalize the heights to height AGL instead of MSL
        # heights = value[2]
        # heights = [H - heights[0] for H in heights]
        # Set the base color and alpha
        temp_color = "dodgerblue"
        dewp_color = "darkorchid"
        alpha = 0.5
        # Get max wind and adjust color/alpha as needed
        max_wind = value[0]
        if max_wind >= 26.0:
            temp_color = "firebrick"
            dewp_color = "green"
            alpha = 1.0
        plt.plot(value[3],value[1],color=temp_color,alpha=alpha)
        plt.plot(value[4], value[1], color=dewp_color, alpha=alpha)
    # Add the relevant special lines to plot throughout the figure
    skew.plot_dry_adiabats(np.arange(233, 533, 20) * units.K, alpha=0.15, color='gray')
    skew.plot_moist_adiabats(np.arange(233, 400, 5) * units.K, alpha=0.15, color='darkgreen')
    plt.xlabel("Temperature (C)",fontsize=23)
    plt.ylabel("Pressure (mb)",fontsize=23)
    plt.title(f'Temperature Profiles',fontsize=28)
    # plt.xlim(-50,40)
    # plt.ylim(0,20000)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory,"temperatre_profiles.png"))
########################################################################################################################
def plot_relh_profiles(soundings,save_directory):
    plt.figure()
    for key,value in soundings.items():
        color = "darkorange"
        alpha = 0.2
        # Create the relh profile:
        relh = CF.relh_profile(value[3],value[4])
        # Normalize the heights to height AGL instead of MSL
        heights = value[2]
        heights = [H - heights[0] for H in heights]
        if value[0] >= 26.0:
            color = "blue"
        plt.plot(relh,heights,color=color,alpha=alpha)
    plt.xlabel("Relative Humidity (%)",fontsize=14)
    plt.ylabel("Height AGL (m)",fontsize=14)
    plt.title(f'Rel. Humidity Profiles',fontsize=18)
    plt.xlim(0,105)
    plt.ylim(0,heights[-1])
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory,"relh_profiles.png"))
########################################################################################################################
def plot_dewpoint_depression_profiles(soundings,save_directory):
    plt.figure()
    for key,value in soundings.items():
        alpha = 0.2
        color = "darkorange"
        # Create the T-Td profile:
        dewp_depression = value[3] - value[4]
        # Normalize the heights to height AGL instead of MSL
        heights = value[2]
        heights = [H - heights[0] for H in heights]
        # pres_profile = value[1]
        # Get max wind and adjust color/alpha as needed
        max_wind = value[0]
        if max_wind >= 26.0:
            alpha = 0.2
            color = "blue"
        plt.plot(dewp_depression,heights,color=color,alpha=alpha)
    plt.xlabel("Dewpoint Depression (C)",fontsize=14)
    plt.ylabel("Height AGL (m)",fontsize=14)
    plt.title(f'Dewpoint Depression Profiles',fontsize=20)
    plt.xlim(0,50)
    plt.ylim(0,heights[-1])
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory,"dewpoint_depression_profiles.png"))
########################################################################################################################
def plot_theta_profiles(soundings,save_directory):
    plt.figure()
    for key,value in soundings.items():
        alpha = 0.2
        color = "darkorange"
        # Create the theta profile
        theta = CF.get_theta_profile(value[1],value[3])
        # Normalize the heights to height AGL instead of MSL
        heights = value[2]
        heights = [H - heights[0] for H in heights]
        # pres_profile = value[1]
        # Get max wind and adjust color/alpha as needed
        max_wind = value[0]
        if max_wind >= 26.0:
            alpha = 0.2
            color = "blue"
        plt.plot(theta,heights,color=color,alpha=alpha)
    plt.xlabel("Potential Temperature (K)",fontsize=14)
    plt.ylabel("Height AGL (m)",fontsize=14)
    plt.title(f'Potential Temperature Profiles',fontsize=20)
    #plt.xlim(0,50)
    plt.ylim(0,heights[-1])
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory,"theta_profiles.png"))
########################################################################################################################
def plot_thetaE_profile(soundings,save_directory):
    plt.figure()
    for key,value in soundings.items():
        alpha = 0.2
        color = "darkorange"
        # Create the theta profile
        thetaE = CF.theta_e_profile(value[1],value[3],value[4])
        # Normalize the heights to height AGL instead of MSL
        heights = value[2]
        heights = [H - heights[0] for H in heights]
        # pres_profile = value[1]
        # Get max wind and adjust color/alpha as needed
        max_wind = value[0]
        if max_wind >= 26.0:
            alpha = 0.2
            color = "blue"
        plt.plot(thetaE,heights,color=color,alpha=alpha)
    plt.xlabel("Theta-E (K)",fontsize=14)
    plt.ylabel("Height AGL (m)",fontsize=14)
    plt.title(f'Eq. Potential Temperature Profiles',fontsize=20)
    #plt.xlim(0,50)
    plt.ylim(0,heights[-1])
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory,"thetaE_profiles.png"))
########################################################################################################################
def plot_uv_profile(soundings,save_directory):
    plt.figure()
    for key,value in soundings.items():
        alpha = 0.2
        u_color = "darkorange"
        v_color = "darkorange"
        # Normalize the heights to height AGL instead of MSL
        heights = value[2]
        heights = [H - heights[0] for H in heights]
        # pres_profile = value[1]
        # Get max wind and adjust color/alpha as needed
        max_wind = value[0]
        if max_wind >= 26.0:
            alpha = 0.2
            u_color = "blue"
            v_color = "blue"
        ax1 = plt.subplot(1,2,1)
        ax1.plot(value[5],heights,color=u_color,alpha=alpha)
        ax2 = plt.subplot(1,2,2)
        ax2.plot(value[6],heights, color=v_color, alpha=alpha)
    ax1.set_xlabel("Wind Speed (m/s)",fontsize=14)
    ax2.set_xlabel("Wind Speed (m/s)", fontsize=14)
    ax1.set_ylabel("Height AGL (m)",fontsize=14)
    plt.suptitle(f'U/V Wind Profiles',fontsize=20)

    ax1.set_xlim(-30,30)
    ax2.set_xlim(-30,30)
    ax1.set_ylim(0,heights[-1])
    ax2.set_ylim(0,heights[-1])
    ax1.grid(alpha=0.5)
    ax2.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory,"uv_profiles.png"))
########################################################################################################################
def plot_mixr_profile(soundings,save_directory):
    plt.figure()
    for key,value in soundings.items():
        alpha = 0.2
        color = "darkorange"
        # Create the theta profile
        mixr = CF.get_mixing_ratio_profile(value[1],value[3],value[4])
        # Normalize the heights to height AGL instead of MSL
        heights = value[2]
        heights = [H - heights[0] for H in heights]
        # pres_profile = value[1]
        # Get max wind and adjust color/alpha as needed
        max_wind = value[0]
        if max_wind >= 26.0:
            alpha = 0.2
            color = "blue"
        plt.plot(mixr,heights,color=color,alpha=alpha)
    plt.xlabel("Mixing Ratio (g/kg)",fontsize=14)
    plt.ylabel("Height AGL (m)",fontsize=14)
    plt.title(f'Mixing RatioProfiles',fontsize=20)
    #plt.xlim(0,50)
    plt.ylim(0,heights[-1])
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory,"mixr_profiles.png"))
########################################################################################################################
def plot_mean_temp_profile(soundings,save_directory):
    # Establish figure
    fig = plt.figure(figsize=(12,12),tight_layout=True)
    skew = mpplots.SkewT(fig,rotation=45)

    # Get the mean temperature profiles for each wind
    temp_profiles = {}
    dewp_profiles = {}
    for key,value in soundings.items():
        press_profile = value[1] # This will be constant for all soundings except for the very first entry (which we'll ignore"
        temp_profiles[key] = [value[0],value[3],press_profile]
        dewp_profiles[key] = [value[0],value[4],press_profile]

    temp_15 = median_profile(temp_profiles,0.0,15.0,minheight=1000.0,maxheight=100.0,height_interval=-25.0)
    temp_svr  = median_profile(temp_profiles,26.0,100.0,minheight=1000.0,maxheight=100.0,height_interval=-25.0)

    dewp_15 = median_profile(dewp_profiles,0.0,15.0,minheight=1000.0,maxheight=100.0,height_interval=-25.0)
    dewp_svr  = median_profile(dewp_profiles,26.0,100.0,minheight=1000.0,maxheight=100.0,height_interval=-25.0)

    t15_25 = profile_percentile(temp_profiles,0.0,15.0,25,minheight=1000.0,maxheight=100.0,height_interval=-25.0)
    t15_75 = profile_percentile(temp_profiles, 0.0, 15.0, 75,minheight=1000.0,maxheight=100.0,height_interval=-25.0)
    tsvr_25 = profile_percentile(temp_profiles,26.0,100.0,25,minheight=1000.0,maxheight=100.0,height_interval=-25.0)
    tsvr_75 = profile_percentile(temp_profiles,26.0,100.0,75,minheight=1000.0,maxheight=100.0,height_interval=-25.0)

    d15_25 = profile_percentile(dewp_profiles,0.0,15.0,25,minheight=1000.0,maxheight=100.0,height_interval=-25.0)
    d15_75 = profile_percentile(dewp_profiles, 0.0, 15.0, 75,minheight=1000.0,maxheight=100.0,height_interval=-25.0)
    dsvr_25 = profile_percentile(dewp_profiles,26.0,100.0,25,minheight=1000.0,maxheight=100.0,height_interval=-25.0)
    dsvr_75 = profile_percentile(dewp_profiles,26.0,100.0,75,minheight=1000.0,maxheight=100.0,height_interval=-25.0)

    # Set a standard pressure profile
    standard_press_profile = np.arange(1000.0,100.0,-25.0)

    skew.plot(standard_press_profile,temp_15,linestyle="-",color="dodgerblue",alpha=1.0)
    skew.plot(standard_press_profile, t15_25, linestyle="-", color="lightskyblue", alpha=0.5)
    skew.plot(standard_press_profile, t15_75, linestyle="-", color="lightskyblue", alpha=0.5)

    skew.plot(standard_press_profile, temp_svr, linestyle="-",color="firebrick", alpha=1.0)
    skew.plot(standard_press_profile, tsvr_25, linestyle="-", color="lightcoral", alpha=0.5)
    skew.plot(standard_press_profile, tsvr_75, linestyle="-", color="lightcoral", alpha=0.5)

    skew.plot(standard_press_profile,dewp_15,linestyle="-",color="darkorchid",alpha=1.0)
    skew.plot(standard_press_profile, d15_25, linestyle="-", color="mediumpurple", alpha=0.5)
    skew.plot(standard_press_profile, d15_75, linestyle="-", color="mediumpurple", alpha=0.5)

    skew.plot(standard_press_profile, dewp_svr, linestyle="-",color="darkgreen", alpha=1.0)
    skew.plot(standard_press_profile, dsvr_25, linestyle="-", color="lightgreen", alpha=0.5)
    skew.plot(standard_press_profile, dsvr_75, linestyle="-", color="lightgreen", alpha=0.5)

    # Add the relevant special lines to plot throughout the figure
    skew.plot_dry_adiabats(np.arange(233, 533, 20) * units.K, alpha=0.15, color='gray')
    skew.plot_moist_adiabats(np.arange(233, 400, 5) * units.K, alpha=0.15, color='darkgreen')
    plt.xlabel("Temperature (C)",fontsize=23)
    plt.ylabel("Pressure (mb)",fontsize=23)
    plt.title(f'Temperature Profiles',fontsize=28)
    # plt.xlim(-50,40)
    # plt.ylim(0,20000)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory,"mean_temp_profiles.png"))
########################################################################################################################
def plot_mean_uv_profile(soundings,save_directory):
    minheight = 0.0
    maxheight = 16000.0
    height_interval = 200.0
    height_profile = np.arange(minheight, maxheight, height_interval)
    # Get the mean U/V profiles for each sounding
    u_profiles = {}
    v_profiles = {}
    for key,value in soundings.items():
        # Normalize the heights to height AGL instead of MSL
        heights = value[2]
        heights = [H - heights[0] for H in heights]
        u_profiles[key] = [value[0],value[5],heights]
        v_profiles[key] = [value[0],value[6],heights]

    u_15 = median_profile(u_profiles,0.0,15.0)
    u_svr  = median_profile(u_profiles,26.0,100.0)
    v_15 = median_profile(v_profiles,0.0,15.0)
    v_svr  = median_profile(v_profiles,26.0,100.0)

    u15_25 = profile_percentile(u_profiles,0.0,15.0,25)
    u15_75 = profile_percentile(u_profiles, 0.0, 15.0, 75)
    usvr_25 = profile_percentile(u_profiles,26.0,100.0,25)
    usvr_75 = profile_percentile(u_profiles,26.0,100.0,75)

    v15_25 = profile_percentile(v_profiles,0.0,15.0,25)
    v15_75 = profile_percentile(v_profiles, 0.0, 15.0, 75)
    vsvr_25 = profile_percentile(v_profiles,26.0,100.0,25)
    vsvr_75 = profile_percentile(v_profiles,26.0,100.0,75)

    plt.figure()
    ax1 = plt.subplot(1,2,1)
    ax1.plot(u_15,height_profile,color="darkorange",alpha=1.0)
    ax1.plot(u15_25, height_profile, color="sandybrown", alpha=0.5)
    ax1.plot(u15_75, height_profile, color="sandybrown", alpha=0.5)

    ax1.plot(u_svr,height_profile,color="blue",alpha=1.0)
    ax1.plot(usvr_25, height_profile, color="lightskyblue", alpha=0.5)
    ax1.plot(usvr_75, height_profile, color="lightskyblue", alpha=0.5)

    ax2 = plt.subplot(1,2,2)
    ax2.plot(v_15, height_profile, color="darkorange", alpha=1.0)
    ax2.plot(v15_25, height_profile, color="sandybrown", alpha=0.5)
    ax2.plot(v15_75, height_profile, color="sandybrown", alpha=0.5)

    ax2.plot(v_svr, height_profile, color="blue", alpha=1.0)
    ax2.plot(vsvr_25, height_profile, color="lightskyblue", alpha=0.5)
    ax2.plot(vsvr_75, height_profile, color="lightskyblue", alpha=0.5)

    ax1.set_xlabel("U Wind Mag. (m/s)",fontsize=14)
    ax2.set_xlabel("V Wind Mag.(m/s)", fontsize=14)
    ax1.set_ylabel("Height (m AGL)",fontsize=14)
    plt.suptitle(f'U/V Wind Profiles',fontsize=20)

    ax1.set_xlim(-30,30)
    ax2.set_xlim(-30,30)
    ax1.set_ylim(minheight,maxheight)
    ax2.set_ylim(minheight,maxheight)
    ax1.grid(alpha=0.5)
    ax2.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory,"mean_uv_profiles.png"))
########################################################################################################################
def plot_mean_dewp_depress_profile(soundings,save_directory):
    minheight = 0.0
    maxheight = 16000.0
    height_interval = 200.0
    height_profile = np.arange(minheight, maxheight, height_interval)
    # Get the relh profiles for each sounding
    depress_profiles = {}
    for key,value in soundings.items():
        # Normalize the heights to height AGL instead of MSL
        heights = value[2]
        heights = [H - heights[0] for H in heights]
        dewpdepress = (value[3] - value[4])
        depress_profiles[key] = [value[0],dewpdepress,heights]

    depress_15 = median_profile(depress_profiles,0.0,15.0)
    depress_svr = median_profile(depress_profiles,26.0,100.0)

    depress15_25 = profile_percentile(depress_profiles,0.0,15.0,25)
    depress15_75 = profile_percentile(depress_profiles, 0.0, 15.0, 75)
    depresssvr_25 = profile_percentile(depress_profiles,26.0,100.0,25)
    depresssvr_75 = profile_percentile(depress_profiles,26.0,100.0,75)

    plt.figure()
    plt.plot(depress_15,height_profile,linestyle='-',color='darkorange',alpha=1.0)
    plt.plot(depress15_25, height_profile, linestyle='-', color='sandybrown', alpha=0.5)
    plt.plot(depress15_75, height_profile, linestyle='-', color='sandybrown', alpha=0.5)

    plt.plot(depress_svr,height_profile,linestyle='-',color='blue',alpha=1.0)
    plt.plot(depresssvr_25, height_profile, linestyle='-', color='lightskyblue', alpha=0.5)
    plt.plot(depresssvr_75, height_profile, linestyle='-', color='lightskyblue', alpha=0.5)

    plt.xlabel("Dewpoint Depression (C)",fontsize=18)
    plt.ylabel("Height (m AGL)",fontsize=18)
    plt.title(f'Dewpoint Depression Profiles',fontsize=21)
    plt.xlim(0,30)
    plt.ylim(minheight, maxheight)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory,"mean_dewp_depress_profiles.png"))
########################################################################################################################
def plot_mean_thetae_profile(soundings,save_directory):
    minheight = 0.0
    maxheight = 16000.0
    height_interval = 200.0
    height_profile = np.arange(minheight, maxheight, height_interval)
    # Get the relh profiles for each sounding
    thetae_profiles = {}
    for key,value in soundings.items():
        # Normalize the heights to height AGL instead of MSL
        heights = value[2]
        heights = [H - heights[0] for H in heights]
        thetae = CF.theta_e_profile(value[1],value[3],value[4])
        thetae_profiles[key] = [value[0],thetae,heights]

    depress_15 = median_profile(thetae_profiles,0.0,15.0)
    depress_svr = median_profile(thetae_profiles,26.0,100.0)

    depress15_25 = profile_percentile(thetae_profiles,0.0,15.0,25)
    depress15_75 = profile_percentile(thetae_profiles, 0.0, 15.0, 75)
    depresssvr_25 = profile_percentile(thetae_profiles,26.0,100.0,25)
    depresssvr_75 = profile_percentile(thetae_profiles,26.0,100.0,75)

    plt.figure()
    plt.plot(depress_15,height_profile,linestyle='-',color='darkorange',alpha=1.0)
    plt.plot(depress15_25, height_profile, linestyle='-', color='sandybrown', alpha=0.5)
    plt.plot(depress15_75, height_profile, linestyle='-', color='sandybrown', alpha=0.5)

    plt.plot(depress_svr,height_profile,linestyle='-',color='blue',alpha=1.0)
    plt.plot(depresssvr_25, height_profile, linestyle='-', color='lightskyblue', alpha=0.5)
    plt.plot(depresssvr_75, height_profile, linestyle='-', color='lightskyblue', alpha=0.5)

    plt.xlabel("Theta-E (K)",fontsize=18)
    plt.ylabel("Height (m AGL)",fontsize=18)
    plt.title(f'Theta-E Profiles',fontsize=21)
    # plt.xlim(0,30)
    plt.ylim(minheight, maxheight)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory,"mean_thetae_profiles.png"))
########################################################################################################################
def plot_mean_relh_profile(soundings,save_directory):
    minheight = 0.0
    maxheight = 16000.0
    height_interval = 200.0
    height_profile = np.arange(minheight, maxheight, height_interval)
    # Get the relh profiles for each sounding
    relh_profiles = {}
    for key,value in soundings.items():
        # Normalize the heights to height AGL instead of MSL
        heights = value[2]
        heights = [H - heights[0] for H in heights]
        relh = CF.relh_profile(value[3],value[4])
        relh_profiles[key] = [value[0],relh,heights]

    relh_15 = median_profile(relh_profiles,0.0,15.0)
    relh_svr  = median_profile(relh_profiles,26.0,100.0)

    relh15_25 = profile_percentile(relh_profiles,0.0,15.0,25)
    relh15_75 = profile_percentile(relh_profiles, 0.0, 15.0, 75)
    relhsvr_25 = profile_percentile(relh_profiles,26.0,100.0,25)
    relhsvr_75 = profile_percentile(relh_profiles,26.0,100.0,75)

    plt.figure()
    plt.plot(relh_15,height_profile,linestyle='-',color='darkorange',alpha=1.0)
    plt.plot(relh15_25, height_profile, linestyle='-', color='sandybrown', alpha=0.5)
    plt.plot(relh15_75, height_profile, linestyle='-', color='sandybrown', alpha=0.5)

    plt.plot(relh_svr,height_profile,linestyle='-',color='blue',alpha=1.0)
    plt.plot(relhsvr_25, height_profile, linestyle='-', color='lightskyblue', alpha=0.5)
    plt.plot(relhsvr_75, height_profile, linestyle='-', color='lightskyblue', alpha=0.5)

    plt.xlabel("Relative Humidity (%)",fontsize=18)
    plt.ylabel("Height (m AGL)",fontsize=18)
    plt.title(f'Rel. Humidity Profiles',fontsize=21)
    plt.xlim(0,105)
    plt.ylim(minheight,maxheight)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory,"mean_relh_profiles.png"))
########################################################################################################################
def plot_dcape_profile(soundings,save_directory):
    minheight = 0.0
    maxheight = 16000.0
    height_interval = 200.0
    height_profile = np.arange(minheight, maxheight, height_interval)
    # Get the dcape profiles for each sounding
    dcape_profiles = {}
    for key,value in soundings.items():
        # Normalize the heights to height AGL instead of MSL
        heights = value[2]
        heights = [H - heights[0] for H in heights]
        dcape = []
        for P in value[1]:
            dcape.append(CF.DCAPE(value[3],value[4],value[1],target_pres=P).magnitude)
        dcape_profiles[key] = [value[0],dcape,heights]

    dcape_15 = median_profile(dcape_profiles,0.0,15.0)
    dcape_svr  = median_profile(dcape_profiles,26.0,100.0)

    dcape15_25 = profile_percentile(dcape_profiles,0.0,15.0,25)
    dcape15_75 = profile_percentile(dcape_profiles, 0.0, 15.0, 75)
    dcapesvr_25 = profile_percentile(dcape_profiles,26.0,100.0,25)
    dcapesvr_75 = profile_percentile(dcape_profiles,26.0,100.0,75)

    plt.figure()
    color = "darkgreen"

    plt.plot(dcape_15,height_profile,linestyle='-',color='darkorange',alpha=1.0,label="Sub-Severe")
    plt.plot(dcape15_25, height_profile, linestyle='-', color='sandybrown', alpha=0.5)
    plt.plot(dcape15_75, height_profile, linestyle='-', color='sandybrown', alpha=0.5)

    plt.plot(dcape_svr,height_profile,linestyle='-',color='blue',alpha=1.0,label="Severe")
    plt.plot(dcapesvr_25, height_profile, linestyle='-', color='lightskyblue', alpha=0.5)
    plt.plot(dcapesvr_75, height_profile, linestyle='-', color='lightskyblue', alpha=0.5)

    plt.xlabel("DCAPE (J/kg)",fontsize=18)
    plt.ylabel("Height (m AGL)",fontsize=18)
    plt.title(f'DCAPE Profiles',fontsize=21)
    plt.xlim(0,1500)
    plt.ylim(minheight,maxheight)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory,"mean_dcape_profiles.png"))
########################################################################################################################
def plot_dryentrain_profile(soundings,save_directory):
    minheight = 0.0
    maxheight = 16000.0
    height_interval = 200.0
    height_profile = np.arange(minheight, maxheight, height_interval)
    plt.figure()
    # Get the dcape profiles for each sounding
    profiles = {}
    for key,value in soundings.items():
        # Normalize the heights to height AGL instead of MSL
        heights = value[2]
        heights = [H - heights[0] for H in heights]
        dryentr = CF.dry_entrain_est(value[3],value[4],value[5],value[6])
        profiles[key] = [value[0],dryentr,heights]

    dcape_15 = median_profile(profiles,0.0,21.0)
    dcape15_25 = profile_percentile(profiles,0.0,21.0,25)
    dcape15_75 = profile_percentile(profiles, 0.0, 21.0, 75)
    dcape_svr  = median_profile(profiles,26.0,100.0)
    dcapesvr_25 = profile_percentile(profiles,26.0,100.0,25)
    dcapesvr_75 = profile_percentile(profiles,26.0,100.0,75)

    plt.plot(dcape_15, height_profile,linestyle='-',color='darkorange',alpha=1.0,label="Sub-Severe")
    plt.plot(dcape15_25, height_profile, linestyle='-', color='sandybrown', alpha=0.5)
    plt.plot(dcape15_75, height_profile, linestyle='-', color='sandybrown', alpha=0.5)
    plt.plot(dcape_svr, height_profile,linestyle='-',color='blue',alpha=1.0,label="Severe")
    plt.plot(dcapesvr_25, height_profile, linestyle='-', color='lightskyblue', alpha=0.5)
    plt.plot(dcapesvr_75, height_profile, linestyle='-', color='lightskyblue', alpha=0.5)

    plt.xlabel("Dry Entrain. Param.",fontsize=18)
    plt.ylabel("Height (m AGL)",fontsize=18)
    plt.title(f'Layer Dry Entrainment Parameter',fontsize=18)
    plt.xlim(0,400)
    plt.ylim(minheight,maxheight)
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory,"mean_dryentrain_profiles.png"))
########################################################################################################################

for sounding in os.listdir(sounding_directory):
    # For each sounding we need to do two things:
    # 1) Get the max wind from the simulation
    # 2) Get the profile data
    # We'll store the soundings in a dict with each item being the profile
    sounding_dir = sounding[0:-4]
    # Get the max wind data.
    try:
        # Try the first path
        try:
            subpath = os.path.join(data_directory1, sounding_dir)
            max_wind = get_max_wind(subpath,last_file)
        except:
            # Try the second path
            try:
                subpath = os.path.join(data_directory2, sounding_dir)
                max_wind = get_max_wind(subpath,last_file)
            except:
                # Try the third path
                subpath = os.path.join(data_directory3, sounding_dir)
                max_wind = get_max_wind(subpath,last_file)
        # Get the sounding profile
        df = CF.read_snd(os.path.join(sounding_directory,sounding))
        # Now combine all data into the soundings dict
        soundings[sounding_dir] = [max_wind,df["pres"].values,df["hght"].values,df["temp"].values,df["dewp"].values,df["u"].values,df["v"].values]
        total = total + 1
    # If there was a problem processing this profile:
    except:
        print(f'WARNING: Could not use run {sounding_dir}')

# Now start creating some plots
# plot_temp_profiles(soundings,save_directory)
plot_relh_profiles(soundings,save_directory)
# plot_dewpoint_depression_profiles(soundings,save_directory)
# plot_theta_profiles(soundings,save_directory)
# plot_thetaE_profile(soundings,save_directory)
# plot_uv_profile(soundings,save_directory)
# plot_mixr_profile(soundings,save_directory)
# plot_mean_temp_profile(soundings,save_directory)
# plot_mean_uv_profile(soundings,save_directory)
# plot_mean_relh_profile(soundings,save_directory)
# plot_dcape_profile(soundings,save_directory)
# plot_dryentrain_profile(soundings,save_directory)
# plot_mean_dewp_depress_profile(soundings,save_directory)
plot_mean_thetae_profile(soundings,save_directory)
print("DONE!")