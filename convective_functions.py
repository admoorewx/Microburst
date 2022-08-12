import numpy as np
import metpy.calc as mcalc
import metpy.constants as mcon
from metpy.units import units
import pandas as pd
########################################################################################################################
def parcel_trajectory(pres, temp, dewp, type="surface"):
    """
    Finds the parcel trajectory given profiles of pressure, temperature, and dewpoint.
    Parcel type options are "surface", "mixed-layer", and "most-unstable". Otherwise will raise an exception.
    If parcel type is "mixed_layer" then assumes the mixed-layer is 100 mb deep from the surface.
    Returns the parcel trajectory.
    """
    if type == "mixed-layer":
        # define depth of ML parcel
        depth = 100.0 * units.hPa
        top_of_layer = (pres[0] - depth) # get the top of the layer
        mid_layer = (pres[0] - (depth/2.0)) # get the mid-point of the layer
        inds = np.where(pres >= top_of_layer) # get all values within the layer
        mean_temp = np.mean(temp[inds]) # find the average temp
        mean_dewp = np.mean(dewp[inds]) # find the average dewp
        inds = np.where(pres <= mid_layer) # get the profile above the mid-layer point
        p = pres[inds]
        t = temp[inds]
        td = dewp[inds]
        p = np.insert(p,0,mid_layer) # add in the mid-level point so we can lift from this point
        t = np.insert(t,0,mean_temp)
        td = np.insert(td,0,mean_dewp)
        parcel_path = mcalc.parcel_profile(p,t,td)
        return parcel_path
    elif type == "surface":
        parcel_path = mcalc.parcel_profile(pres, temp[0], dewp[0])
        return parcel_path
    elif type == "most-unstable":
        thetae = mcalc.equivalent_potential_temperature(pres, temp, dewp)
        ind = np.where(np.nanmax(thetae.magnitude))[0][0]
        parcel_path = mcalc.parcel_profile(pres[ind:], temp[ind], dewp[ind])
        return parcel_path
    else:
        raise Exception("ERROR: unkown parcel-type. Options are 'most-unstable', 'mixed-layer' or 'surface'.")
########################################################################################################################
def K2C(t):
    """
    Convert temperature in Kelvin to Celsius.
    """
    return t - 273.15
########################################################################################################################
def getParcelInfo(type,p,t,td):
    """
    :param type: parcel type. Options are "surface", "mixed-layer", and "most-unstable"
    :param p: pressure profile with units hPa
    :param t: temperature profile with units deg C
    :param td: dewpoint profile with units deg C
    :return: LCL pressure and temp, LFC pressure and temp, EL pressure and temp
    """
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
        # calculate the LCL, LFC, and EL
        lcl_p, lcl_t = mcalc.lcl(mid_layer, temp, dewp)
        lfc_p, lfc_t = mcalc.lfc(p, t, td)
        el_p, el_t = mcalc.el(p, t, td)
        return lcl_p, lcl_t, lfc_p, lfc_t, el_p, el_t
    elif type == "surface":
        parcel_path = mcalc.parcel_profile(p, t[0], td[0])
        # calculate the LCL, LFC, and EL
        lcl_p, lcl_t = mcalc.lcl(p[0],t[0],td[0])
        lfc_p, lfc_t = mcalc.lfc(p,t,td)
        el_p, el_t = mcalc.el(p,t,td)
        return lcl_p, lcl_t, lfc_p, lfc_t, el_p, el_t
    elif type == "most-unstable":
        thetae = mcalc.equivalent_potential_temperature(p, t, td)
        ind = np.where(np.nanmax(thetae.magnitude))[0][0]
        parcel_path = mcalc.parcel_profile(p[ind:], t[ind], td[ind])
        # calculate the LCL, LFC, and EL
        lcl_p, lcl_t = mcalc.lcl(p[ind], t[ind], td[ind])
        lfc_p, lfc_t = mcalc.lfc(p, t, td)
        el_p, el_t = mcalc.el(p, t, td)
        return lcl_p, lcl_t, lfc_p, lfc_t, el_p, el_t
    else:
        print("ERROR: unkown parcel-type. Options are 'most-unstable', 'mixed-layer' or 'surface'.")
        return None

########################################################################################################################
def getCape(type,p,t,td):
    """
    :param type: parcel type. Options are "surface", "mixed-layer", and "most-unstable"
    :param p: pressure profile with units hPa
    :param t: temperature profile with units deg C
    :param td: dewpoint profile with units deg C
    :return: CAPE, and CIN with units J/kg
    """
    if type == "mixed-layer":
        return mcalc.mixed_layer_cape_cin(p,t,td)
    elif type == "surface":
        return mcalc.surface_based_cape_cin(p,t,td)
    elif type == "most-unstable":
        return mcalc.most_unstable_cape_cin(p,t,td)
    else:
        print("ERROR: unkown parcel-type. Options are 'most-unstable', 'mixed-layer' or 'surface'.")
        return None
########################################################################################################################
def virtualTemp(p,t,td):
    """
    Determine the virtual temperature profile from pres, Temp, and dewpoint.
    :param p: pressure profile in hPa
    :param t: temperature profile in deg C
    :param td: dewpoint profile in deg C
    :return: virtual temperature profile in deg C
    """
    if t.units != units('degC'):
        t = t.to('degC')
        td = td.to('degC')
    if p.units != units('hPa'):
        p = p.to('hPa')
    # es = mcalc.saturation_vapor_pressure(t)
    e = mcalc.saturation_vapor_pressure(td)
    w = ((621.97 * (e/(p-e)))/1000.0).to('g/kg')
    return mcalc.virtual_temperature(t,w).to('degC')
########################################################################################################################
def get_effective_bulk_shear(p,t,td,u,v,hght):
    """
    Determine the effective bulk shear value. See www.spc.noaa.gov/exper/mesoanalysis/help/help_eshr.html for details.
    :param p: pressure profile in hPa
    :param t: temperature profile in deg C
    :param td: dewpoint profile in deg C
    :param u: U-wind
    :param v: V-wind
    :return: Effective bulk shear. Note that units will match the input wind units.
    """
    # get the effective inflow layer
    eff_layer = effective_inflow_layer(p,t,td)
    if len(eff_layer) > 0:
        # get the bottom of the effective inflow layer
        bottom_ind = np.where(p == eff_layer[0])[0][0]

        # get the EL of the most unstable parcel
        el_p, el_t = mcalc.el(p,t,td,which='most_cape')
        # if the el_p is NaN, then the actual EL is above the sounding data (i.e., there's missing data at the top of the sounding)
        # just set the el_p and el_t to the top most pres/temp
        if np.isnan(el_p) or el_p is None:
            el_p = p[-1]
        # find the index of the EL
        diffs = [np.abs(P.magnitude-el_p.magnitude) for P in p]
        min = np.nanmin(diffs)
        el_ind = np.where(diffs==min)[0][0]
        # Find the depth of the layer
        depth = (hght[el_ind] - hght[bottom_ind]) / 2.0
        # Submit parameters to bulk_shear
        us,vs = mcalc.bulk_shear(p,u,v,hght,bottom=hght[bottom_ind],depth=depth)
        shear_mag = mcalc.wind_speed(us,vs)
        return shear_mag # Gives the shear in knots
    else:
        return 0.0 * units('knot') # Return an effective bulk wind shear value of 0.0 knots if there is no eff. layer
########################################################################################################################
def effective_inflow_layer(p,t,td):
    """
    Determine the effective inflow layer.
    :param p: pressure profile in hPa
    :param t: temperature profile in deg C
    :param td: dewpoint profile in deg C
    :return: returns the pressure profile (hPa) of the effective inflow layer.
    """
    tv = virtualTemp(p,t,td)
    min_cape = 100.0 * units('J/kg')
    min_cin = -250.0 * units('J/kg')
    inds = []
    for i in range(0,len(p)):
        parcel_path = mcalc.parcel_profile(p[i:],tv[i],td[i])
        cape, cin = mcalc.cape_cin(p[i:],t[i:],td[i:],parcel_path)
        if cape >= min_cape and cin >= min_cin:
            # we found the bottom of the layer
            # add to inds and search up for the top, then exit
            inds.append(i)
            layer_found = True
            j = i
            while layer_found:
                parcel_path = mcalc.parcel_profile(p[j:], tv[j], td[j])
                cape, cin = mcalc.cape_cin(p[j:], t[j:], td[j:], parcel_path)
                if cape >= min_cape and cin >= min_cin:
                    inds.append(j)
                else:
                    layer_found = False
                j = j + 1
            break
    return p[inds]
########################################################################################################################
def thetaE_defficit(pres,thetae):
    """
    Determines the theta-e defficit (difference) between the max value in the lower atmosphere vs. the min value in the
    mid/upper atmostphere. The levels that determine the lower/upper atmosphere can be set manually.
    :param pres: Pressure profile in hPa
    :param thetae: theta-e profile in Kelvin
    :return: The theta-e defficit in Kelvin
    """
    top_lower_layer = 700.0 # mb - Top of the bottom layer
    top_upper_layer = 100.0 # mb - where to stop looking for the min temp
    # Just get the numbers:
    pres = [p.magnitude for p in pres]
    pres = np.asarray(pres)
    # get the max temp in the lowest X mb:
    lower_layer_inds = [np.where(pres >= top_lower_layer)][0]
    max_temp = np.nanmax(thetae[lower_layer_inds])
    # get the min temp in the upper portion
    upper_layer_inds = [np.where(pres >= top_upper_layer)][0]
    #upper_layer_inds = [np.where(pres[upper_layer_inds] <= top_lower_layer)][0] # make sure to exclude the lower layer
    min_temp = np.nanmin(thetae[upper_layer_inds])
    diff = max_temp - min_temp
    return diff
########################################################################################################################
def cleanData(hght, pres, temp, dewp, u, v):
    """
    Checks for NaNs and where the profile data descends, returns cleaned data
    :param hght: height profile (meters)
    :param pres: pressure profile (hPa)
    :param temp: temperature profile (deg C)
    :param dewp: dewpoint profile (deg C)
    :param u: U-wind profile (knots)
    :param v: V-wind profile (knots)
    :return: All input profiles
    """
    i = 1
    while i < len(hght):
        L = [hght[i], temp[i], dewp[i], pres[i], u[i], v[i]]
        if any(np.isnan(L)) or hght[i] <= hght[i - 1]:
            hght = np.delete(hght, i)
            pres = np.delete(pres, i)
            temp = np.delete(temp, i)
            dewp = np.delete(dewp, i)
            u = np.delete(u, i)
            v = np.delete(v, i)
        i = i + 1
    return hght, pres, temp, dewp, u, v
########################################################################################################################
def data_from_csv(csvfile,wind_units="m/s"):
    """
    Read in a CSV file with profile data as a Pandas Dataframe. Returns profile data with metpy units assigned.
    :return:
     hght: height profile (meters)
     pres: pressure profile (hPa)
     temp: temperature profile (deg C)
     dewp: dewpoint profile (deg C)
        u: U-wind profile
        v: V-wind profile
     wind_units: units to assign to the wind profile. Default is "m/s", but soundings are commonly in "knots"
    """
    print(f'Retrieving data from file {csvfile}')
    df = pd.read_csv(csvfile)
    pressure = df['pressure'].values
    temp = df['temperature'].values
    dewp = df['dewpoint'].values
    u = df['u_wind'].values
    v = df['v_wind'].values
    hght = df['height'].values
    hght, pressure, temp, dewp, u, v = cleanData(hght, pressure, temp, dewp, u, v)
    hght = hght * units.meter
    pressure = pressure * units.hPa
    temp = temp * units.degC
    dewp = dewp * units.degC
    u = u * units(wind_units)
    v = v * units(wind_units)
    return pressure, temp, dewp, u, v, hght
########################################################################################################################
def theta_e_profile(p,t,td):
    """
    Return the Theta-E temperature profile
    :param p: pressure profile (hPa)
    :param t: temperature profile (deg C)
    :param td: dewpoint profile (deg C)
    Assumes correct units are already given, otherwise will assign the units noted above
    :return: theta-e profile (Kelvin)
    """
    try:
        return mcalc.equivalent_potential_temperature(p, t, td)
    except:
        p = p * units.hPa
        t = t * units.degC
        td = td * units.degC
        return mcalc.equivalent_potential_temperature(p, t, td)
########################################################################################################################
def bulk_shear_magnitude(p,hght,u,v,depth=6000,bottom=None):
    """
    Return the bulk shear magnitude in knots
    :param p: pressure profile (hPa)
    :param hght: height profile (meters)
    :param u: U-wind profile
    :param v: V-wind profile
    :param depth: Desired depth of the layer to find bulk shear over (meters, passed to metpy bulk_shear, default 6000 m)
    :param bottom: Bottom layer where to start the bulk shear calc (meters, passed to metpy bulk_shear, default is first height level)
    :return: Bulk shear magnitude. Note that units will match the input wind units.
    """
    if bottom is None:
        bottom = hght[0]
    u_comp, v_comp = mcalc.bulk_shear(p, u, v, hght, depth=(depth * units.meter),bottom=bottom)
    BS_mag = mcalc.wind_speed(u_comp,v_comp)
    return BS_mag
########################################################################################################################
def craven_brooks_sig_severe(cape,shear):
    """
    Return the Craven-Brooks sig severe parameter.
    :param cape: Some form of CAPE with units J/kg
    :param shear: Some form of shear. Checked for units, if none, assumed to be knots
    :return: unitless C.B. Sig Severe parameter
    """
    # Check for cape units - just need magnitude
    try:
        cape = cape.magnitude
    except:
        pass
    # check for shear units. If in knots, convert to m/s
    # If no units, assume knots and convert to m/s
    try:
        if shear.units == "knot":
            shear = shear.to(units('m/s'))
    except:
        shear = shear * units('knots')
        shear = shear.to(units('m/s'))
    shear = shear.magnitude
    return cape * shear
########################################################################################################################
def effective_storm_relative_helicity(hght,pres,u,v,t,td):
    """
    Finds the effective storm relative helicity. All inputs must have units described below
    :param hght: height profile (meters)
    :param pres: pressure profile (hPa)
    :param u: U-wind profile (m/s)
    :param v: V-wind profile (m/s)
    :param t: temperature profile (deg C)
    :param td: dewpoint profile (deg C)
    :return: Positive effective storm-relative helicity (m2/s2)
    """
    # get the effective inflow layer, if none, return zero.
    eff_inflow_layer = effective_inflow_layer(pres,t,td)
    if len(eff_inflow_layer) == 0:
        return 0.0 * units('m*m/s*s')
    else:
        # find an estimate for mean storm motion
        u_sm, v_sm = mcalc.bunkers_storm_motion(pres, u, v, hght)[2] # Note 0 - R.M., 1 - L.M., 2 = Mean S.M.
        # get the bottom/top of the eff. inflow layer
        bottom_ind = np.where(pres == eff_inflow_layer[0])[0]
        top_ind = np.where(pres == eff_inflow_layer[-1])[0]
        depth = hght[top_ind] - hght[bottom_ind]
        # find ESRH
        srh = mcalc.storm_relative_helicity(hght,u,v,depth=depth,bottom=hght[bottom_ind],storm_u=u_sm,storm_v=v_sm)
        return srh[0]
########################################################################################################################
def supercell_composite_parameter(mucape,mucin,esrh,BSeff):
    """
    Find the supercell composite parameter with the mucin term.
    :param mucape: most-unstable cape (J/kg)
    :param mucin: most-unstable cin (J/kg)
    :param esrh: effective storm relative helicity (m2/s2)
    :param BSeff: effective bulk shear (m/s)
    :return: supercell composite parameter (unitless)
    """
    # Check units of bulk shear
    if BSeff.units == "knot":
        BSeff = BSeff.to(units('m/s'))
    # use the metpy SCP formulation to start
    scp = mcalc.supercell_composite(mucape,esrh,BSeff)
    # multiply by the CIN term (not included in metpy version) if applicable
    if mucin.magnitude > -40.0:
        return scp[0].magnitude
    else:
        return (scp * (-40.0/mucin.magnitude))[0].magnitude
########################################################################################################################
def lapse_rate(hght,temp,bottom=0.0,depth=3000.0):
    """
    Determine the lapse-rate of some layer beginning at "bottom" that is "depth" meters deep.
    Takes a simple dTemp/dHeight difference between the top and bottom of this layer.
    :param hght: height profile (meters)
    :param temp: temperature profile (meters)
    :param bottom: bottom of the layer (default is 0 meters/surface)
    :param depth: depth of the layer (default is 3 km/3000 meters)
    :return: lapse rate over the layer in temp units / meters
    """
    # find the hght value that matches most closely with the bottom and top of the layer: bottom - depth
    top = bottom + depth
    if bottom == 0.0:
        bottom_ind = 0
    else:
        bottom_diffs = [np.abs(H.magnitude - bottom) for H in hght]
        bottom_ind = bottom_diffs.index(np.nanmin(bottom_diffs))
    top_diffs = [np.abs(H.magnitude - top) for H in hght]
    top_ind = top_diffs.index(np.nanmin(top_diffs))
    dt = temp[top_ind] - temp[bottom_ind]
    dz = hght[top_ind] - hght[bottom_ind]
    dz = dz.to(units('km')) # convert from m to km
    return -1 * dt/dz
########################################################################################################################
def average_lapse_rate(hght,temp,bottom=0.0,depth=3000.0):
    """
    Determine the average lapse-rate of some layer beginning at "bottom" that is "depth" meters deep.
    Takes a simple average of the dTemp/dHeight difference between each sequential layer from the bottom
    of the layer to the top of the layer.
    :param hght: height profile (meters)
    :param temp: temperature profile (meters)
    :param bottom: bottom of the layer (default is 0 meters/surface)
    :param depth: depth of the layer (default is 3 km/3000 meters)
    :return: lapse rate over the layer in temp units / meters
    """
    # find the hght value that matches most closely with the bottom and top of the layer: bottom - depth
    top = bottom + depth
    if bottom == 0.0:
        bottom_ind = 0
    else:
        bottom_diffs = [np.abs(H.magnitude - bottom) for H in hght]
        bottom_ind = bottom_diffs.index(np.nanmin(bottom_diffs))
    top_diffs = [np.abs(H.magnitude - top) for H in hght]
    top_ind = top_diffs.index(np.nanmin(top_diffs))

    # Determine the average lapse rate. Return with units C/km
    lapse_rates = []
    for i in range(bottom_ind, top_ind-1):
        dt = temp[i+1] - temp[i]
        dz = hght[i+1] - hght[i]
        dz = dz.to(units('km')) # convert from m to km
        lapse_rates.append((dt/dz).magnitude)
    avg_lapse_rate = np.nanmean(lapse_rates)
    avg_lapse_rate = avg_lapse_rate * units.degC / units('km')
    return avg_lapse_rate
########################################################################################################################
def boundary_layer_depth(hght,temp):
    """
    Determine the depth of the boundary layer. Begin at the surface and check each height level for a lapse
    rate that is >= target lapse rate. This process is repeated until the lapse rate condition is no longer met,
    and the height at which the condition was previously met is returned as the depth of the boundary layer.
    :param hght: height profile (meters)
    :param temp: temperature profile (meters)
    :return: depth of the boundary layer (meters)
    """
    target_lapse_rate = 0.007 # C/meter - or 8 C/km
    bl_depth = 0.0 * units.meter
    for i in range(4,len(hght)):
        dt = temp[i] - temp[0]
        dz = hght[i] - hght[0]
        print((-1 * (dt/dz).magnitude))
        if (-1 * (dt/dz).magnitude) >= target_lapse_rate:
            bl_depth = dz
            print("Larger")
        else:
            print("BREAK")
            break
    return bl_depth
########################################################################################################################
def color_by_wind(wspd):
    """
    Returns a color based on wind speed
    :param wspd: wind speed with units 'knot'
    :return: color string
    """
    wspd = wspd.magnitude
    if wspd < 20.0:
        return 'cornflowerblue'
    elif wspd < 30.0:
        return 'steelblue'
    elif wspd < 40.0:
        return 'blue'
    elif wspd < 50:
        return 'navy'
    elif wspd < 75:
        return 'blueviolet'
    elif wspd < 100:
        return 'indigo'
    elif wspd < 150:
        return 'darkred'
    else:
        return 'magenta'
########################################################################################################################
def color_by_height(h):
    """
    Returns a color based on height value
    :param h: (float) height value with units 'meter'
    :return: color string
    """
    try:
        h = h.magnitude
    except:
        pass
    if h <= 3000.0:
        return 'firebrick'
    elif h <= 6000.0:
        return 'green'
    elif h <= 9000.0:
        return 'gold'
    else:
        return 'steelblue'
########################################################################################################################
def height_of_theta_min(theta,hght):
    """
    Given input profiles of theta-e and height, find the minimum value and return the height of that value
    """
    if theta.units:
        theta = [th.magnitude for th in theta]
        hght = [h.magnitude for h in hght]

    min_theta = np.nanmin(theta)
    ind = np.where(np.asarray(theta) == min_theta)[0][0]
    return hght[ind]
########################################################################################################################
def relh_profile(temp,dewp):
    """
    Takes in a temperature and dewpoint profile and returns a relative humidity profile.
    Assumes temp and dewpoint already have units assigned, if not, will assign them the unit degC
    """
    # Make sure temp and dewp have units
    try:
        relh = mcalc.relative_humidity_from_dewpoint(temp,dewp)
        return relh*100.0
    except:
        temp = temp * units.degC
        dewp = dewp * units.degC
        relh = mcalc.relative_humidity_from_dewpoint(temp,dewp)
        return relh*100.0
########################################################################################################################
def read_snd(path,use_uv=True,use_omega=False):
    """
    Reads in a .snd file and returns a pandas Dataframe with the profile information, not including metpy units.
    This routine does a final check to see if there is enough valid data (set by parameter min_lines). If so,
    return the dataframe as normal, otherwise an empty dataframe is returned after an error message is displayed.
    .snd files are organized by:
    Param: PRES HEIGHT, TEMP, DEWP, WDIR, WSPD, OMEGA
    units: mb    meters   C     C    deg   knot  ub/s
    use_uv => return the u and v wind components, otherwise return wspd, wdir (default = true).
    use_omega => return the omega profile. default = false.
    """
    min_lines = 10
    data = {
        "pres": [],
        "hght": [],
        "temp": [],
        "dewp": [],
        "wspd": [],
        "wdir": [],
        "omga": [],
        "u": [],
        "v": []
    }
    read = False
    f = open(path,'r')
    for line in f.readlines():
        if "%RAW%" in line:
            read = True
        elif "%END%" in line:
            read = False
        if read:
            line = line.replace(',',"")
            line = line.split()
            try:
                # If this is the first line of data:
                if len(data["pres"]) == 0:
                    data["pres"].append(float(line[0]))
                    data["hght"].append(float(line[1]))
                    data["temp"].append(float(line[2]))
                    data["dewp"].append(float(line[3]))
                    data["wdir"].append(float(line[4]))
                    data["wspd"].append(float(line[5]))
                    data["omga"].append(float(line[6]))
                # If this isn't the first entry into the dict, check for rising pressure with height (i.e. an error).
                elif float(line[0]) < data["pres"][-1]:
                    data["pres"].append(float(line[0]))
                    data["hght"].append(float(line[1]))
                    data["temp"].append(float(line[2]))
                    data["dewp"].append(float(line[3]))
                    data["wdir"].append(float(line[4]))
                    data["wspd"].append(float(line[5]))
                    data["omga"].append(float(line[6]))
                else: # Skip this line due to rising pressure with height
                    continue
            except:
                if "%RAW%" in line:
                    continue # Ignore, this is expected behavior
                else:
                    print(f'WARNING: encountered error reading line in file {path}')
                    print(f'Problem found in this line: {line}')
    f.close()
    if len(data["pres"]) < min_lines:
        print(f"WARNING: Not enough valid data in file {path}")
        return pd.DataFrame()
    else:
        if use_uv:
            # if use_uv, find u and v then remove the wspd and wdir items
            u, v = mcalc.wind_components(data["wspd"] * units.knots,data["wdir"]*units.degrees)
            # Convert to m/s
            u = u.to(units('m/s'))
            v = v.to(units('m/s'))
            u = [U.magnitude for U in u]
            v = [V.magnitude for V in v]
            data["u"] = u
            data["v"] = v
            data.pop("wspd")
            data.pop("wdir")
            if use_omega:
                return pd.DataFrame.from_dict(data)
            else:
                # If not using omega, remove the omega item
                data.pop("omga")
                return pd.DataFrame.from_dict(data)
        else:
            if use_omega:
                return pd.DataFrame.from_dict(data)
            else:
                # Remove the u, v, and omega items
                data.pop("u")
                data.pop("v")
                data.pop("omga")
                return pd.DataFrame.from_dict(data)
########################################################################################################################
def get_theta_profile(pres,temp):
    """
    Returns a profile of potential temperature given a pressure and temp profile.
    Assumes units are already assigned, otherwise assigns hPa and degC respectively.
    """
    try:
        theta = mcalc.potential_temperature(pres,temp)
        return theta
    except:
        pres = pres * units.hPa
        temp = temp * units.degC
        theta = mcalc.potential_temperature(pres,temp)
        return theta
########################################################################################################################
def get_mixing_ratio_profile(pres, temp, dewp):
    """
    Returns a profile of mixing ratios given profiles of pressure, temperature,
    and dewpoint. Assumes units are already assigned, otherwise will assign units
    hPa, and degC.
    """
    # Get the relh profile first:
    relh = relh_profile(temp,dewp)
    # adjust back to decimal form
    relh = relh / 100.0
    # Get the mixing ratio profile
    try:
        mixr = mcalc.mixing_ratio_from_relative_humidity(pres, temp, relh)
        return mixr * 1000.0 # return g/kg
    except:
        pres = pres * units.hPa
        temp = temp * units.degC
        mixr = mcalc.mixing_ratio_from_relative_humidity(pres, temp, relh)
        return mixr * 1000.0 # return g/kg
########################################################################################################################
def find_closest(values,target):
    """
    Assumes the input list or array 'values' is of the same type as the 'target'. Finds the numerical difference between
    every value in 'values' and 'target' and returns the index of the minimum difference.
    """
    try:
        values = [V.magnitude for V in values]
    except:
        pass
    try:
        target = target.magnitude
    except:
        pass
    diffs = [abs(V-target) for V in values]
    min_val = np.min(diffs)
    return diffs.index(min_val)
########################################################################################################################
def layer_mean_relh(height,temp,dewp,bottom,top,pres=None):
    """
    Takes in profiles of height, temperature, and dewpoint and find the average relative humidity of a layer specified
    by 'bottom' and 'top'. Assumes 'bottom' and 'top' are in meters. If a pressure profile is passed ('pres' does not
    equal None), then it is assumed that 'bottom' and 'top' are in hPa. Returns the mean RH.
    """
    # find the relh profile
    relh = relh_profile(temp,dewp)
    # Find the index of the bottom and top of the layer.
    # Need to check if pressure has been passed. If so, check for matching pressures.
    if pres is not None:
        if pres.units == bottom.units and pres.units == top.units:
            pres = [P.magnitude for P in pres]
            bottom_ind = find_closest(pres,bottom.magnitude)
            top_ind = find_closest(pres,top.magnitude)
        else:
            raise Exception("Pressure profile was passed, but 'bottom' and 'top' do not match the units of pressure.")
    # Otherwise, check for matching heights.
    else:
        if height.units == bottom.units and height.units == top.units:
            height = [H.magnitude for H in height]
            bottom_ind = find_closest(height,bottom.magnitude)
            top_ind = find_closest(height,top.magnitude)
        else:
            raise Exception("Only height profile was passed, but 'bottom' and 'top' do not match the units of height.")
    # Check to make sure the bottom_ind is not > top_ind
    if bottom_ind >= top_ind:
        raise Exception("Error: 'bottom' value is greater than or equal to the 'top' value.")
    else:
        # Now find the mean RH
        mean_relh = np.nanmean(relh[bottom_ind:top_ind])
        return mean_relh * units.percent
########################################################################################################################
def lifted_index(pressure,temp,dewp,type="surface",target_level=500.0):
    """
    Takes in profiles of pressure, temperature, and dewpoint, finds a parcel trajectory based on "type", and then
    calculates the LI between the environment and the parcel path at a target pressure value. Assumes parcel type
    is surface-based unless specified, and assumes the target_level is 500 hPa (as in traditial LI calculations).
    Parcel type options are "surface", "mixed-layer", and "most-unstable".
    """
    # First find the parcel path:
    parcel_path = parcel_trajectory(pressure,temp,dewp,type)
    # Find the index of the target pressure level
    try:
        pressure = [P.magnitude for P in pressure]
    except:
        pass
    ind = find_closest(pressure,target_level)
    # Get the target level temperature
    temp_env = temp[ind].magnitude
    # get the parcel profile temp
    temp_parcel = K2C(parcel_path[ind].magnitude)
    # return the difference
    return temp_env - temp_parcel
########################################################################################################################
def precipitable_water(pres,dewp):
    """
    Passes a pressure and dewpoint profile to metpy's precip. water function.
    Returns PWAT in inches instead of the default millimeters.
    """
    pwat = mcalc.precipitable_water(pres,dewp)
    pwat = pwat.to(units.inch)
    return pwat
########################################################################################################################
def unit_check(input):
    try:
        units = input.units
    except:
        units = None
    return units
########################################################################################################################
def DCAPE(temp,dewp,pres,target_pres=None):
    """
    Given a profile of temp, dewp, and pres, this function will return downdraft CAPE (DCAPE).
    This assumes that the temp/dewp profiles are in deg C (assigned deg C if not) and pres and
    target_pres are in hPa (assigned to pres if not).
    This function calculates DCAPE roughly in the same form as Gilmore and Wicker (1998), which follows:
    if target_pres == None:
        1) Find the minimum theta-e value in the profile (no restrictions on depth).
        2) Find the pres, temp, and dewp at the min theta-e level.
        3) Lift this parcel until you reach the LCL.
        4) Based on the LCL temp/pres, find the temperatures along the moist adiabat.
        5) Restrict all profiles from the surface to the level of the min theta-e value.
        6) Deviation from Gilmore & Wicker 1998: Calculate the DCAPE using the formula given
           for the metpy CAPE/CIN.
           Found here: (https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.cape_cin.html)
        7) Return DCAPE in J/kg.
    else:
        Follow the same routine as above, but lift the parcel from the level of the
        target pressure instead.
    """
    if unit_check(temp) == None:
        temp = temp * units.degC
    if unit_check(dewp) == None:
        dewp = dewp * units.degC
    if unit_check(pres) == None:
        pres = pres * units.hPa
    if target_pres == None:
        # Find min Theta-e
        theta_e = theta_e_profile(pres, temp, dewp)
        min_ind = np.where(theta_e == np.min(theta_e))[0][0]
    else:
        min_ind = find_closest(pres,target_pres)
    # Find pres, temp, and dewp at this level.
    ref_temp = temp[min_ind]
    ref_dewp = dewp[min_ind]
    ref_pres = pres[min_ind]
    # Lift this min theta-e parcel to it's LCL to get the correct moist adiabat
    lcl_p, lcl_t = mcalc.lcl(ref_pres,ref_temp,ref_dewp)
    parcel_profile = mcalc.moist_lapse(pres,lcl_t,reference_pressure=lcl_p)
    # Find the differences between the env and the parcel profile
    profile_difference = parcel_profile[0:min_ind].magnitude - temp[0:min_ind].magnitude
    # Integrate and multiply by Rd to get DCAPE.
    dcape = mcon.dry_air_gas_constant * (np.trapz(profile_difference,x=np.log(pres[0:min_ind].magnitude)) * units.kelvin)
    return dcape
########################################################################################################################
def kuchera_parker_wind_param(pres,temp,dewp,u,v):
    """
    Takes in profiles of pressure, temp, dewp, u and v winds to find the wind damage
    parameter described by Kuchera and Parker 2006.
    Their damaging wind parameter is calculated as:
    DMGWND = WINDINF/8.0 * DCAPE/800.0
    Where WINDINF is the wind magnitude at the top of the surface-based inflow layer.
    """
    # Find the WINDINF value first. Basically, we will move up the profile and find the
    # 400 mb LI. As soon as we have a negatively buoyant parcel then that's the top of the
    # surface inflow layer.
    ind_400 = find_closest(pres,400.0)
    temp_400 = temp[ind_400]
    top_inflow_ind = -1
    for i,P in enumerate(pres):
        parcel_profile = mcalc.parcel_profile(pres[i:], temp[i], dewp[i])
        mini_400_ind = find_closest(pres[i:],400.0)
        parcel_temp = parcel_profile[mini_400_ind]
        diff = parcel_temp - temp_400
        if diff <= 0.0:
            top_inflow_ind = i
            break
    if top_inflow_ind == -1:
        print("Warning: Could not find top of the surface inflow layer")
        return 0.0
    else:
        u_top = u[top_inflow_ind]
        v_top = v[top_inflow_ind]
        windinf = mcalc.wind_speed(u_top,v_top)
    # Find DCAPE
    dcape = DCAPE(temp,dewp,pres)
    parameter = (windinf.magnitude/8.0) * (dcape.magnitude/800.0)
    return parameter
########################################################################################################################
def microburst_composite(pres,temp,dewp,hght):
    """
    Source: https://www.spc.noaa.gov/exper/mesoanalysis/help/help_mbcp.html
    The microburst composite is a weighted sum of the following:
    SBCAPE, DCAPE, SBLI, 0-3 km Lapse Rate, Vertical Totals, PWAT.
    We'll take in a thermodyanmic profile of press, temp, and dewp to find each
    of these and return the microburst composite.
    """
    # SBCAPE
    sbcape,sbcin = mcalc.surface_based_cape_cin(pres,temp,dewp)
    if sbcape.magnitude < 3100.0:
        sbcape_weight = 0.0
    elif sbcape.magnitude >= 4000.0:
        sbcape_weight = 2.0
    else:
        sbcape_weight = 1.0
    # DCAPE
    dcape = DCAPE(temp,dewp,pres)
    if dcape.magnitude >= 1300.0:
        dcape_weight = 3.0
    elif dcape.magnitude >= 1100.0:
        dcape_weight = 2.0
    elif dcape.magnitude >= 900.0:
        dcape_weight = 1.0
    else:
        dcape_weight = 0.0
    # SBLI
    li = lifted_index(pres,temp,dewp)
    if li <= -10.0:
        li_weight = 3.0
    elif li <= -9.0:
        li_weight = 2.0
    elif li <= -8:
        li_weight = 1.0
    else:
        li_weight = 0.0
    # 0-3 km lapse rate - the lapse_rate function uses the 0-3 km layer by default
    lr = lapse_rate(hght,temp)
    if lr.magnitude >= 8.4:
        lr_weight = 1.0
    else:
        lr_weight = 0.0
    # Vertical Totals - difference between the 850 and 500 mb temperatures.
    ind_500 = find_closest(pres,500.0)
    ind_850 = find_closest(pres,850.0)
    vt = temp[ind_850] - temp[ind_500]
    if vt.magnitude >= 29.0:
        vt_weight = 3.0
    elif vt.magnitude >= 28:
        vt_weight = 2.0
    elif vt.magnitude >= 27:
        vt_weight = 1.0
    else:
        vt_weight = 0.0
    # PWAT
    pwat = precipitable_water(pres,dewp)
    if pwat.magnitude > 1.5:
        pwat_weight = 0.0
    else:
        pwat_weight = -5.0
    micro_comp = sbcape_weight + dcape_weight + li_weight + lr_weight + vt_weight + pwat_weight
    if micro_comp < 0.0:
        return 0.0
    else:
        return micro_comp
########################################################################################################################
def max_layer_wind_speed(hght,u,v,bottom=0.0,top=3000.0):
    """
    Given a profile of hegith and u/v winds, find the maximum wind within the layer
    defined by bottom:top. If bottom/top are not given, assume the layer is the 0-3 km layer.
    return the max wind magnitude in m/s.
    """
    ind_bottom = find_closest(hght,bottom)
    ind_top = find_closest(hght,top)
    wind_speeds = mcalc.wind_speed(u[ind_bottom:ind_top],v[ind_bottom:ind_top])
    max_wind = np.max(wind_speeds)
    return max_wind
########################################################################################################################
def dry_entrain_est(temp,dewp,u,v):
    """
    Given profiles of temp, dewp, u and v, finds a very rough estimate of how much dry air
    entrainment could occur at a given level. This function finds a profile of RH and then
    determines the value:
    (100-RHmean) * (BWDmean)
    where RHmean = the average RH value between a given layer and the layer above and below the
    given layer.
    BWDmean = the average bulk wind difference between a given layer and the layer above and
    below the given layer. Larger values imply that greater mixing of dry air is possible due
    to vertical wind shear between two layers.
    """
    dry_entrain = []
    if unit_check(u) == None:
        u = u * units('m/s')
        v = v * units('m/s')
    relh = relh_profile(temp,dewp)
    for i in range(0,len(relh)):
        if i == 0:
            RHmean = np.mean([relh[i], relh[i+1]])
            Vtop = v[i+1] - v[i]
            Utop = u[i+1] - u[i]
            bwdtop = mcalc.wind_speed(Utop,Vtop)
            BWDmean = bwdtop
        elif i == (len(relh)-1):
            RHmean = np.mean([relh[i], relh[i-1]])
            Vbot = v[i] - v[i-1]
            Ubot = u[i] - u[i-1]
            bwdbot = mcalc.wind_speed(Ubot,Vbot)
            BWDmean = bwdbot
        else:
            RHmean = np.mean([relh[i], relh[i-1], relh[i+1]])
            Vtop = v[i+1] - v[i]
            Utop = u[i+1] - u[i]
            Vbot = v[i] - v[i-1]
            Ubot = u[i] - u[i-1]
            bwdtop = mcalc.wind_speed(Utop,Vtop)
            bwdbot = mcalc.wind_speed(Ubot,Vbot)
            BWDmean = (bwdtop + bwdbot) / 2.0
        value = (100.0 - RHmean) * BWDmean.magnitude
        dry_entrain.append(value)
    return dry_entrain
################################################################################################################
def composite_param_from_sounding(df):
    """
    This is a test composite parameter. The input is a dataframe
    containing a single sounding profile with the format: 
    [pres(hPa), hght (meters), temp (C), dewp (C), U (m/s), V (m/s)]
    Output is a single non-dimensional value representing the potential for severe wet
    microbust. Values can range from zero to (theoretically) infinity, 
    but are typically less than 10.
    """
    # Filters
    max_bulk_shear = 12.0 # Maximum allowed 0-6 km bulk shear in m/s
    min_dewp_depress = 7.5 # Minimum allowed dewp. depression (C)
    # Normalizing Factors
    temp_factor = 30.0 # C
    lr_factor = 7.5 # K/km
    wind_factor = 10.0 # m/s
    cin_additive = 50.0 # J/kg
    cin_factor = 25.0 # J/kg
    el_p_factor = 170.0 # hPa
    el_t_factor = -60.0 # C
    lfc_t_factor = 15.0 # C

    # Get profile data from dataframe
    pres = df["pres"].values * units.hPa
    height = df["hght"].values * units.meter 
    temp = df["temp"].values * units.degC 
    dewp = df["dewp"].values * units.degC 
    u = df["u"].values * units('m/s')
    v = df["v"].values * units('m/s')

    # Find MLCIN 
    mlcape, mlcin = getCape("mixed-layer",pres,temp,dewp)

    # Get 0-3 km max wind 
    max_wind_3km = max_layer_wind_speed(height,u,v,bottom=0.0,top=3000.0)

    # Find 0-3 km lapse rate 
    lr3km = lapse_rate(height,temp)

    # Find 0-6 km bulk shear
    BS6km = bulk_shear_magnitude(pres, height, u, v, depth=6000)

    # Get surface temperature 
    surface_temp = temp[0]

    # Get dewpoint depression
    dewp_depress = surface_temp - dewp[0]

    # Find ML EL temp, pres and LFC temp
    ml_lcl_p, ml_lcl_t, ml_lfc_p, ml_lfc_t, ml_el_p, ml_el_t = getParcelInfo("mixed-layer",pres,temp,dewp)

    # Just get the magnitudes of all params
    mlcin = mlcin.magnitude
    max_wind_3km = max_wind_3km.magnitude
    BS6km = BS6km.magnitude
    surface_temp = surface_temp.magnitude
    dewp_depress = dewp_depress.magnitude
    ml_el_t = ml_el_t.magnitude
    ml_el_p = ml_el_p.magnitude
    ml_lfc_t = ml_lfc_t.magnitude

    # Apply initial filters
    if BS6km < 12.0: 
        bs_filter = 1.0
    else: 
        bs_filter = 0.0
    if dewp_depress >= min_dewp_depress: 
        dd_filter = 1.0
    else:
        dd_filter = 0.0

    # Normalize
    mlcin = (cin_additive + mlcin)/cin_factor
    surface_temp = surface_temp/temp_factor
    max_wind_3km = max_wind_3km/wind_factor
    ml_el_t = ml_el_t/el_t_factor
    ml_el_p = el_p_factor/ml_el_p
    ml_lfc_t = ml_lfc_t/lfc_t_factor
    lr3km = lr3km/lr_factor

    # Do final checks
    if mlcin < 0.0:
        mlcin = 0.0
    if ml_el_t < 0.0:
        ml_el_t = 0.0
    if surface_temp < 0.0:
        surface_temp = 0.0
    if ml_lfc_t < 0.0:
        ml_lfc_t = 0.0
    # Check for nans in the EL/LFC data. This happens if these levels cannot be computed.
    if np.isnan(ml_el_p):
        ml_el_p = 0.0
    if np.isnan(ml_el_t):
        ml_el_t = 0.0
    if np.isnan(ml_lfc_t):
        ml_lfc_t = 0.0

    values = [surface_temp, max_wind_3km, ml_el_t, ml_el_p, ml_lfc_t, mlcin, bs_filter, dd_filter]
    composite = surface_temp * max_wind_3km  * ml_el_p * ml_el_t  * ml_lfc_t * mlcin * bs_filter * dd_filter * lr3km
    if composite < 0.0 or np.isnan(composite):
        print("WARNING: Something went wrong with parameter calculation:")
        print(values)
    return composite
################################################################################################################

################################################################################################################
################################################################################################################