# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 11:35:50 2019

@author: Gijs Hendrickx
"""

import numpy as np
import pandas as pd
import datetime
import os
from netCDF4 import Dataset

from Temperature import delta_Tc


def find_nearest(array, value):
    """
    Function to find the nearest value in a dataset to the one needed.

    Parameters
    ----------
    array : array
        Array containing the data.
    value : numeric
        Value needed.

    Returns
    -------
    array[idx] : numeric
        The value of the dataset that is nearest to the requested value.
    idx : integer
        The index at which the nearest value can be found in the array.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def data_coor(lat, lon):
    """
    Give the DD coordinates of the latitude and longitude of the two SST data
    sets (NOAA and Kaplan) that are closest to the desired latitude and
    longitude.

    Parameters
    ----------
    lat : numeric
        Latitude [DD coordinates].
    lon : numeric
        Longitude [DD coordinates].

    Returns
    -------
    latN : numeric
        Nearest matching latitude in the NOAA data set [DD coordinates].
    lonN : numeric
        Nearest matching longitude in the NOAA data set [DD coordinates].
    latK : numeric
        Nearest matching latitude in the Kaplan data set [DD coordinates].
    lonK : numeric
        Nearest matching longitude in the Kaplan data set [DD coordinates].
    """
    # # NOAA data set
    # define lats and lons present in NOAA data set
    latsN = np.linspace(-89.875, 89.875, 720)
    lonsN = np.linspace(.125, 359.875, 1440)
    # find nearest matching lat and lon
    latN, _ = find_nearest(latsN, lat)
    lonN, _ = find_nearest(lonsN, lon)

    # # Kaplan data set
    # define lats and lons present in Kaplan data set
    latsK = np.linspace(-87.5, 87.5, 36)
    lonsK = np.linspace(2.5, 357.5, 72)
    # find nearest matching lat and lon
    latK, _ = find_nearest(latsK, lat)
    lonK, _ = find_nearest(lonsK, lon)

    return np.array([[latN, lonN],
                     [latK, lonK]])


def read_SST_nc(lat, lon, filename, path='', latlon=0):
    """
    Read the SST time-series written in a .nc-file and extracts the data for
    the given latitude and longitude.

    Parameters
    ----------
    lat : numeric
        Lattitude [DD coordinates].
    lon : numeric
        Longitude [DD coordintes].
    file : string
        File-name with SST-data.
    path : string, optional
        Path / directory to file.
    latlon : True/False, optional
        Return the used latitude and longitude [DD coordinates].

    Returns
    -------
    time.data : array
        Time as indicated in file.
    SST.data : array
        Daily SST data [deg C].
    latnc : numeric, optional
        Used latitude [DD coordinates].
    lonnc : numeric, optional
        Used longitude [DD coordinates].

    NetCDF4-reader needed to read .nc-files:
    "conda install -c anaconda netcdf4".

    References:
        NOAA (1981 - 2019, abs, daily):
    https://www.esrl.noaa.gov/psd/data/gridded/data.noaa.oisst.v2.highres.html
        Kaplan (1856 - 2019, anom, monthly):
    https://www.esrl.noaa.gov/psd/data/gridded/data.kaplan_sst.html
    """
    filef = os.path.join(path, filename)

    # # Read dataset
    dset = Dataset(filef, mode='r')
    time = dset['time'][:]
    lats = dset['lat'][:]
    lons = dset['lon'][:]
    # Find best fit for lat and lon within dataset
    latnc, lati = find_nearest(lats, lat)
    lonnc, loni = find_nearest(lons, lon)
    # Read SST-data at specified location
    SST = dset['sst'][:, lati, loni]
    dset.close()

    # # define output
    if latlon:
        output = time.data, SST.data, latnc, lonnc
    else:
        output = time.data, SST.data

    return output


def SST_file_w(lat, lon, YEAR_start, YEAR_end,
               DATE0='01-01-1800', path=None, latlon=False):
    """
    Create a file with SST time-series at a given latitude and longitude; and
    between given years (YEAR_start to YEAR_end). Data in txt-file is written
    as: 'YYYY \t MM \t DD \t SST'.

    Parameters
    ----------
    lat : numeric
        Latitude [DD coordinates].
    lon : numeric
        Longitude [DD coordinates].
    YEAR_start : numeric
        Start year of SST time-series [yr].
    YEAR_end : numeric
        End year of SST time-series [yr].
    DATE0 : string, optional
        Time stamp of data [DD-MM-YYYY].
    path : None, string, optional
        Path / directory to the location at which the txt-file must be saved
        and from which SST data is extracted.
    latlon : True/False, optional
        Return the used latitude and longitude [DD coordinates].

    Returns
    -------
    File with SST time-series.
    """
    # # extract data
    t = []
    sst = []
    for i in range(YEAR_start, YEAR_end + 1):
        ti, ssti, latN, lonN = read_SST_nc(lat, lon,
                                           'sst.day.mean.{0}.nc'.format(i),
                                           'SST_data', latlon=1)
        if ssti.mean() < -999:
            print('\nNo SST data available (land).\n')
            break
        t = np.concatenate((t, ti))
        sst = np.concatenate((sst, ssti))
        print('{0} completed'.format(i))

    # # fill data if land
    if ssti.mean() < -999:
        n = (datetime.datetime(YEAR_end, 12, 31) -
             datetime.datetime(YEAR_start, 1, 1)).days
        t = np.linspace(ti[0], ti[0] + n - 1, n)
        sst = ssti[0] * np.ones(n)

    # # rewrite to dates
    d0 = datetime.datetime.strptime(DATE0, '%d-%m-%Y').date()
    t = np.array([d0 + datetime.timedelta(days=t[i]) for i in range(len(t))])

    # # create path + filename
    filename = ('SST_timeseries_lat{0}lon{1}_Y{2}_{3}.txt'
                .format(latN, lonN, YEAR_start, YEAR_end))
    filef = os.path.join(path, filename)

    # # write data into txt-file
    year = np.array([t[i].year for i in range(len(t))])
    month = np.array([t[i].month for i in range(len(t))])
    day = np.array([t[i].day for i in range(len(t))])
    data = pd.DataFrame({'year': year,
                         'month': month,
                         'day': day,
                         'sst': sst})
    with open(filef, 'w') as f:
        data.to_csv(f, header=True, index=False, sep='\t')
    print('File written: {0}'.format(filef))

    # # define output
    if latlon:
        output = np.array([latN, lonN])
    else:
        output = []

    return output


def Tc_file_w(lat, lon, uc, Cf, Iz, K0,
              DATE0='01-01-1800', path=None, YEAR_start=1981, YEAR_end=2019,
              ap=.4, rd=500, nu=1e-6, alpha=1e-7,  k=.5918,
              removefile=False):
    """
    Create / append a file with the coral temperature time-series at a given
    latitude and longitude as part of the feedback loop of the dynamic coral
    growth model. The data in txt-file is written the same as the SST-file:
        'YYYY \\t MM \\t DD \\t Tc'.

    If no SST-file on the time-series is available, the option is given to
    write one within this function. NOTE: writing this file takes some time and
    it is preferred to have one beforehand. More information: 'SST_file_w?'

    Parameters
    ----------
    lat : numeric
        Latitude [DD coordinates].
    lon : numeric
        Longitude [DD coordinates].
    YEAR_start : numeric
        Start year of SST time-series [yr].
    YEAR_end : numeric
        End year of SST time-series [yr].
    uc : numeric, array, string
        Constricted in-canopy flow [m s^-1].
        Options: [float]: average value; 'file': time-series of uc.
        NOTE: time-series must have same lengths.
    Cf : numeric
        Friction coefficient.
    Iz : numeric, array, string
        Coral surface-averaged light-intensity [mol photons m^-2 s^-1].
        Options: [float]: average value; 'file': time-series of Iz.
        NOTE: time-series must have same lengths.
    K0 : numeric
        Species and morphology dependent coefficient [K0 ~ 1e-11].
    DATE0 : string, optional
        Time stamp of data [DD-MM-YYYY].
    path : string, optional
        Path / directory to the location at which the txt-file must be saved
        and from which SST (and/or uc and/or Iz) data is extracted.
    ap : numeric, optional
        Absorptivity of the coral [-].
    rd : numeric, optional
        Velocity boundary layer wall-coordinate.
    nu : numeric, optional
        Kinematic viscosity [m^2 s^-1].
    alpha : numeric, optional
        Thermal diffusivity [m^2 s^-1].
    k : numeric, optional
        Thermal conductivity [J m^-1 s^-1 K^-1].
    removefile : boolean, optional
        Remove Tc time-series file if it exists before writing the file; i.e.
        [False]: append to existing file, or create a new one if nonexisting;
        [True]: remove old version of the time-series (if existing) and start
        a new time-series.
        For new computations, [True] is recommended to make sure that no old
        data is used. In case placed within the same computations multiple
        times, [False] is recommended so data is not lost.

    NOTE: for uc and Iz to be loaded as time-series, a string must be filled in
    for these parameters; e.g. 'file'. For this option to work, both time-
    series -- as well as the time-series of the SST data -- must have the same
    length and must be in the same folder (see path).

    Returns
    -------
    File with coral temperature time-series. (Corrected for the TBL-effects, if
    data is provided.)
    """
    # # Check files
    # SST-file
    [latN, lonN], [_, _] = data_coor(lat, lon)
    SST_filename = ('SST_timeseries_lat{0}lon{1}_Y{2}_{3}.txt'
                    .format(latN, lonN, YEAR_start, YEAR_end))
    SST_file = os.path.join(path, SST_filename)

    if os.path.isfile(SST_file):
        SST_data = pd.read_csv(SST_file, sep='\t', skiprows=1,
                               names=['year', 'month', 'day', 'sst'])
    else:
        if input('No SST time-series found. '
                 'Write SST-file? [y/n]') != 'y':
            raise NameError('File {0} is not found.'.format(SST_file))
        else:
            SST_file_w(lat, lon, YEAR_start, YEAR_end, DATE0=DATE0, path=path)
    # uc-file
    if isinstance(uc, str):
        uc_filename = ('uc_timeseries_lat{0}lon{1}_Y{2}_{3}.txt'
                       .format(lat, lon, YEAR_start, YEAR_end))
        if not path:
            uc_file = uc_filename
        else:
            uc_file = os.path.join(path, uc_filename)
        if os.path.isfile(uc_file):
            uc_data = pd.read_csv(uc_file, sep='\t', skiprows=1,
                                  names=['year', 'month', 'day', 'uc'])
        else:
            if input('No uc time-series found. '
                     'Continue without flow-effects? [y/n]') != 'y':
                raise NameError('File {0} is not found.'.format(uc_file))
            else:
                uc_data = 1000.
    else:
        uc_data = uc
    # Iz-file
    if isinstance(Iz, str):
        Iz_filename = ('Iz_timeseries_lat{0}lon{1}_Y{2}_{3}.txt'
                       .format(lat, lon, YEAR_start, YEAR_end))
        if not path:
            Iz_file = Iz_filename
        else:
            Iz_file = os.path.join(path, Iz_filename)
        if os.path.isfile(Iz_file):
            Iz_data = pd.read_csv(uc_file, sep='\t', skiprows=1,
                                  names=['year', 'month', 'day', 'Iz'])
        else:
            if input('No Iz time-series found. '
                     'Continue without light-effects? [y/n]') != 'y':
                raise NameError('File {0} is not found.'.format(Iz_file))
            else:
                Iz_data = 0.
    else:
        Iz_data = Iz
    # correct uc and Iz input
    if (isinstance(uc, float) or isinstance(uc, str) or
            isinstance(Iz, float) or isinstance(Iz, str)):
        pass
    else:
        raise NameError('Input uc and Iz unknown. '
                        'Expected float and/or string.')

    # # calculations
    # relative temperature increase at coral tissue
    dTc = delta_Tc(uc_data, Iz_data, Cf, K0,
                   ap=ap, alpha=alpha, k=k, rd=rd, nu=nu)
    # absolute temperature at coral tissue
    Tc_data = SST_data
    Tc_data['tc'] = SST_data.sst + dTc
    Tc_data = Tc_data.drop(['sst'], axis=1)

    # # write data
    filename = ('Tc_timeseries_K0{0}_lat{1}lon{2}_Y{3}_{4}.txt'
                .format(K0, lat, lon, YEAR_start, YEAR_end))
    file_full = os.path.join(path, filename)
    # remove (old) file, if present
    if os.path.isfile(file_full):
        if removefile:
            os.remove(file_full)
    # write/append (new) file
    file = open(file_full, 'a')
    file.write('YYYY\tMM\tDD\tTc')
    for i in range(len(Tc_data)):
        file.write('\n{0}\t{1}\t{2}\t{3}'
                   .format(Tc_data.year[i], Tc_data.month[i],
                           Tc_data.day[i], Tc_data.tc[i]))
    file.close()
    # print file details
    if path is None:
        print('File written: {0}'.format(filename))
    else:
        print('File written: {0} in {1}'.format(filename, path))

    # # output
    return


def arag_sat(pCO2, S, T, pH=8.1, Tunit='deg.C'):
    """
    The aragonite saturation state as function of the atmospheric carbon
    dioxide pressure and other properties of the (sea) water. NOTE: The pH is
    not taken into account dynamically, but is enforced on the system based on
    a well-eduacted guess.

    Parameters
    ----------
    pCO2 : numeric or array
        Atmospheric carbon dioxide pressure [atm].
    S : numeric or array
        Salinity of water [ppt].
    T : numeric or array
        Temperature of water [K].
    pH : numeric or array, optional
        Acidity of water [-].
    Tunit : string, optional
        Unit of temperature used in SST time-series.
        Options: 'K', 'deg.C'

    Returns
    -------
    omega_a : numeric or array
        Aragonite saturation state [-].
    """
    # # input checks
    # temperature units check
    Tunits = ['K', 'deg.C']
    if Tunit not in Tunits:
        raise ValueError('Invalid temperature unit. Expected one of: {0}'
                         .format(Tunits))
    # convert to Kelvin (if necessary)
    if Tunit == 'deg.C':
        T += 273.15

    # # constants of the chemical reactions
    # K0*
    lnK0 = ((-60.2409 + 93.4517 * 100 / T + 23.3585 * np.log(T / 100)) +
            (.023517 - .023656 * T / 100 + .0047036 * (T / 100) ** 2) * S)
    K0 = np.exp(lnK0)
    # K1*
    lnK1 = ((2.83655 - 2307.1266 / T - 1.5529413 * np.log(T)) +
            (-.20760841 - 4.0484 / T) * S ** .5 +
            .1130822 * S -
            .00846934 * S ** 1.5)
    K1 = np.exp(lnK1)
    # K2*
    lnK2 = ((-9.226508 - 3351.6106 / T - .2005743 * np.log(T)) +
            (-.106901773 - 23.9722 / T) * S ** .5 +
            .1130822 * S -
            .00846934 * S ** 1.5)
    K2 = np.exp(lnK2)
    # Ka*
    logKa = ((-171.945 - .077995 * T + 2909.298 / T + 71.595 * np.log10(T)) +
             (-.068393 + .0017276 * T + 88.135 / T) * S ** .5 -
             .10018 * S +
             .0059415 * S ** 1.5)
    Ka = 10 ** (logKa)

    # # H+ concentration (pH)
    h = 10 ** (-pH)

    # # chemical reactions
    # dissolved CO3
    co3 = K0 * K1 * K2 * pCO2 / (h ** 2)
    # dissolved Ca
    ca = .01028 * S / 35
    # aragonite saturation state
    omega = (ca * co3) / Ka
    return omega
