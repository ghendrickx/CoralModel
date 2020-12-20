"""
coral_model - environment

@author: Gijs G. Hendrickx
"""

import pandas as pd
import numpy as np

from utils.config_directory import DirConfig


class Processes:
    """Processes included in coral_model simulations."""
    # TODO: Include the on/off-switch for more processes:
    #  (1) hydrodynamic coupling;
    #  (2) acidity;
    #  (3) light;
    #  (4) temperature;
    #  (5) dislodgement;
    #  (6) recruitment;
    #  (7) etc.

    def __init__(self, fme=True, tme=True, pfd=True):
        """
        :param fme: flow micro-environment, defaults to True
        :param tme: thermal micro-environment, defaults to True
        :param pfd: photosynthetic flow dependency, defaults to True

        :type fme: bool, optional
        :type tme: bool, optional
        :type pfd: bool, optional
        """
        self.pfd = pfd

        if not pfd:
            if fme:
                print(
                    f'WARNING: Flow micro-environment (FME) not possible '
                    f'when photosynthetic flow dependency (PFD) is disabled.'
                )
            self.fme = False
            self.tme = False

        else:
            self.fme = fme
            if not fme:
                if tme:
                    print(
                        f'WARNING: Thermal micro-environment (TME) not possible '
                        f'when flow micro-environment is disabled.'
                    )
                self.tme = False

            else:
                self.tme = tme

        if tme:
            print('WARNING: Thermal micro-environment not fully implemented yet.')

        if not pfd:
            print('WARNING: Exclusion of photosynthetic flow dependency not fully implemented yet.')


class Constants:
    """Object containing all constants used in coral_model simulations."""

    def __init__(self, processes, lac_default=None, light_spreading_max=None,
                 turbulence_coef=None, inertia_coef=None,
                 friction_coef=None, kin_viscosity=None, therm_diff=None, spacing_ratio=None, wc_angle=None, rd=None,
                 theta=None, err=None, maxiter_k=None, maxiter_aw=None, thermal_coef=None, absorptivity=None,
                 therm_cond=None, pa_rate=None, sat_intensity_max=None, photo_max=None,
                 beta_sat_intensity=None, beta_photo=None, act_energy=None, gas_constant=None, thermal_variability=None,
                 nn=None, pfd_min=None, ucr=None, r_growth=None, r_recovery=None, r_mortality=None, r_bleaching=None,
                 calcification_const=None, arg_sat_default=None, omega0=None, kappa0=None, prop_form=None,
                 prop_plate=None, prop_plate_flow=None, prop_space=None, prop_space_light=None, prop_space_flow=None,
                 u0=None, rho_c=None, sigma_tensile=None, drag_coef=None, rho_w=None, no_larvae=None,
                 prob_settle=None, d_larvae=None):
        """
        Parameters
        ----------
        processes : Processes
            Definition of the processes that are included, specified by means
            of the Processes-object.

        > light micro-environment
        lac_default : numeric, optional
            Constant light-attenuation coefficient; is used when no time-series
            is available [m-1]. The default is 0.1.
        light_spreading_max : numeric, optional
            Maximum spreading of light as measured at the water-air interface
            [rad]. The default is 0.5*pi.

        > flow micro-environment
        turbulence_coef : float, optional
            Smagorinsky coefficient [-]. The default is 0.17.
        inertia_coef : float, optional
            Inertia coefficient [-]. The default is 1.7.
        friction_coef : float, optional
            Friction coefficient [-]. The default is 0.01.
        kin_viscosity : float, optional
            Kinematic viscosity of water [m2 s-1]. The default is 1e6.
        therm_diff : float, optional
            Thermal diffusivity of water [m2 s-1]. The default is 1e-7.
        spacing_ratio : float, optional
            Ratio of lateral over streamwise spacing of corals [-]. The default
            is 2.
        wc_angle : float, optional
            Angle between current- and wave-induced flows [rad]. The default
            is 0.
        rd : float, optional
            Velocity boundary layer wall-coordinate [-]. The default is 500.
        theta :  float, optional
            Update ratio for above-canopy flow [-]. The default is 0.5.
        err :  float, optional
            Maximum allowed relative error [-]. The default is 1e-6.
        maxiter_k :  float, optional
            Maximum number of iterations taken over the canopy layers. The
            default is 1e5.
        maxiter_aw :  float, optional
            Maximum number of iterations to solve the complex-valued wave-
            attenuation coefficient. The default is 1e5.

        > thermal micro-environment
        thermal_coef : float, optional
            Morphological thermal coefficient [-]. The default is 80.
        absorptivity : float, optional
            Absorptivity of coral [-]. The default is 0.4.
        therm_cond : float, optional
            Thermal conductivity [J m-1 s-1 K-1]. The default is 0.6089.

        > photosynthetic light dependency
        pa_rate : float, optional
            Photo-acclimation rate [d-1]. The default is 0.6.
        sat_intensity_max : float, optional
            Maximum value of the quasi steady-state for the saturation light-
            intensity [umol photons m-2 s-1]. The default is 372.32.
        photo_max : float, optional
            Maximum value of the quasi steady-state for the maximum
            photosynthetic efficiency [-]. The default is 1.
        beta_sat_intensity : float, optional
            Exponent of the quasi steady-state for the saturation light-
            intensity [-]. The default is 0.34.
        beta_photo : float, optional
            Exponent of the quasi steady-state for the maximum photosynthetic
            efficiency [-]. The default is 0.09.

        > photosynthetic thermal dependency
        act_energy : float, optional
            Activation energy [J mol-1]. The default is 6e4.
        gas_constant : float, optional
            Gas constant [J K-1 mol-1]. The default is 8.31446261815324.
        thermal_variability : float, optional
            Thermal-acclimation coefficient [-]. The default is 2.45.
        nn : float, optional
            Thermal-acclimation period [yrs]. The default is 60.

        > photosynthetic flow dependency
        pfd_min : float, optional
            Minimum photosynthetic flow dependency [-]. The default is
            0.68886964.
        ucr : float, optional
            Minimum flow velocity at which photosynthesis is not limited by
            flow [m s-1]. The default is (1) 0.17162374 if flow micro-
            environment is enabled; and (2) 0.5173... if flow micro-environment
            is disabled.

        > population states
        r_growth : float, optional
            Growth rate [d-1]. The default is 0.002.
        r_recovery : float, optional
            Recovering rate [d-1]. The default is 0.2.
        r_mortality : float, optional
            Mortality rate [d-1]. The default is 0.04.
        r_bleaching : float, optional
            Bleaching rate [d-1]. The default is 8.

        > calcification
        calcification_const : float, optional
            Calcification constant [kg m-2 d-1].. The default is 0.5.
        arg_sat_default : float, optional
            Constant aragonite saturation state (is used when no time-series of
            the parameter is available) [-]. The default is 5.
        omega0 : float, optional
            Aragonite dissolution state [-]. The default is 0.14587415.
        kappa0 : float, optional
            Modified Michaelis-Menten half-rate coefficient [-]. The default
            is 0.66236107.

        > morphological development
        prop_form : float, optional
            Overall form proportionality constant [-]. The default is 0.1.
        prop_plate : float, optional
            Overall plate proportionality constant [-]. The default is 0.5.
        prop_plate_flow : float, optional
            Flow plate proportionality constant [-]. The default is 0.1.
        prop_space : float, optional
            Overall spacing proportionality constant [-]. The default is
            0.5 / sqrt(2).
        prop_space_light : float, optional
            Light spacing proportionality constant [-]. The default is 0.1.
        prop_space_flow : float, optional
            Flow spacing proportionality constant [-]. The default is 0.1.
        u0 : float, optional
            Base-line flow velocity [m s-1]. The default is 0.2.
        rho_c : float, optional
            Density of coral [kg m-3]. The default is 1600.

        > dislodgement criterion
        sigma_tensile : float, optional
            Tensile strength of substratum [N m-2]. The default is 2e5.
        drag_coef : float, optional
            Drag coefficient [-]. The default is 1.
        rho_w : float, optional
            Density of water [kg m-3]. The default is 1025.

        > coral recruitment
        no_larvae : float, optional
            Number of larvae released during mass spawning event [-]. The
            default is 1e6.
        prob_settle : float, optional
            Probability of settlement [-]. The default is 1e-4.
        d_larvae : float, optional
            Larval diameter [m]. The default is 1e-3.

        """
        def default(x, default_value):
            """Set default value if no custom value is provided."""
            if x is None:
                return default_value
            return x

        # light micro-environment
        self.Kd0 = default(lac_default, .1)
        self.theta_max = default(light_spreading_max, .5 * np.pi)

        # flow mirco-environment
        self.Cs = default(turbulence_coef, .17)
        self.Cm = default(inertia_coef, 1.7)
        self.Cf = default(friction_coef, .01)
        self.nu = default(kin_viscosity, 1e-6)
        self.alpha = default(therm_diff, 1e-7)
        self.psi = default(spacing_ratio, 2)
        self.wcAngle = default(wc_angle, 0.)
        self.rd = default(rd, 500)
        self.numericTheta = default(theta, .5)
        self.err = default(err, 1e-3)
        self.maxiter_k = int(default(maxiter_k, 1e5))
        self.maxiter_aw = int(default(maxiter_aw, 1e5))

        # thermal micro-environment
        self.K0 = default(thermal_coef, 80.)
        self.ap = default(absorptivity, .4)
        self.k = default(therm_cond, .6089)

        # photosynthetic light dependency
        self.iota = default(pa_rate, .6)
        self.ik_max = default(sat_intensity_max, 372.32)
        self.pm_max = default(photo_max, 1.)
        self.betaI = default(beta_sat_intensity, .34)
        self.betaP = default(beta_photo, .09)

        # photosynthetic thermal dependency
        self.Ea = default(act_energy, 6e4)
        self.R = default(gas_constant, 8.31446261815324)
        self.k_var = default(thermal_variability, 2.45)
        self.nn = default(nn, 60)

        # photosynthetic flow dependency
        self.pfd_min = default(pfd_min, .68886964)
        self.ucr = default(ucr, .17162374 if processes.fme else .5173)

        # population dynamics
        self.r_growth = default(r_growth, .002)
        self.r_recovery = default(r_recovery, .2)
        self.r_mortality = default(r_mortality, .04)
        self.r_bleaching = default(r_bleaching, 8.)

        # calcification
        self.gC = default(calcification_const, .5)
        self.omegaA0 = default(arg_sat_default, 5.)
        self.omega0 = default(omega0, .14587415)
        self.kappaA = default(kappa0, .66236107)

        # morphological development
        self.prop_form = default(prop_form, .1)
        self.prop_plate = default(prop_plate, .5)
        self.prop_plate_flow = default(prop_plate_flow, .1)
        self.prop_space = default(prop_space, .5 / np.sqrt(2.))
        self.prop_space_light = default(prop_space_light, .1)
        self.prop_space_flow = default(prop_space_flow, .1)
        self.u0 = default(u0, .2)
        self.rho_c = default(rho_c, 1600.)

        # dislodgement criterion
        self.sigma_t = default(sigma_tensile, 2e5)
        self.Cd = default(drag_coef, 1.)
        self.rho_w = default(rho_w, 1025.)

        # coral recruitment
        self.no_larvae = default(no_larvae, 1e6)
        self.prob_settle = default(prob_settle, 1e-4)
        self.d_larvae = default(d_larvae, 1e-3)


class Environment:
    # TODO: Make this class robust

    _dates = None
    _light = None
    _light_attenuation = None
    _temperature = None
    _aragonite = None
    _storm_category = None

    @property
    def light(self):
        """Light-intensity in micro-mol photons per square metre-second."""
        return self._light

    @property
    def light_attenuation(self):
        """Light-attenuation coefficient in per metre."""
        return self._light_attenuation

    @property
    def temperature(self):
        """Temperature time-series in either Celsius or Kelvin."""
        return self._temperature

    @property
    def aragonite(self):
        """Aragonite saturation state."""
        return self._aragonite

    @property
    def storm_category(self):
        """Storm category time-series."""
        return self._storm_category

    @property
    def temp_kelvin(self):
        """Temperature in Kelvin."""
        if all(self.temperature.values < 100) and self.temperature is not None:
            return self.temperature + 273.15
        return self.temperature

    @property
    def temp_celsius(self):
        """Temperature in Celsius."""
        if all(self.temperature.values > 100) and self.temperature is not None:
            return self.temperature - 273.15
        return self.temperature

    @property
    def temp_mmm(self):
        monthly_mean = self.temp_kelvin.groupby([
            self.temp_kelvin.index.year, self.temp_kelvin.index.month
        ]).agg(['mean'])
        monthly_maximum_mean = monthly_mean.groupby(level=0).agg(['min', 'max'])
        monthly_maximum_mean.columns = monthly_maximum_mean.columns.droplevel([0, 1])
        return monthly_maximum_mean

    @property
    def dates(self):
        """Dates of time-series."""
        if self._dates is not None:
            d = self._dates
        elif self.light is not None:
            # TODO: Check column name of light-file
            d = self.light.reset_index().drop('light', axis=1)
        elif self.temperature is not None:
            d = self.temperature.reset_index().drop('sst', axis=1)
        else:
            msg = f'No initial data on dates provided.'
            raise ValueError(msg)
        return pd.to_datetime(d['date'])

    def set_dates(self, start_date, end_date):
        """Set dates manually, ignoring possible dates in environmental time-series.

        :param start_date: first date of time-series
        :param end_date: last date of time-series

        :type start_date: str, datetime.date
        :type end_date: str, datetime.date
        """
        dates = pd.date_range(start_date, end_date, freq='D')
        self._dates = pd.DataFrame({'date': dates})

    def set_parameter_values(self, parameter, value, pre_date=None):
        """Set the time-series data to a time-series, or a default value. In case :param value: is not iterable, the
        :param parameter: is assumed to be constant over time. In case :param value: is iterable, make sure its length
        complies with the simulation length.

        Included parameters:
            light                       :   incoming light-intensity [umol photons m-2 s-1]
            LAC / light_attenuation     :   light attenuation coefficient [m-1]
            temperature                 :   sea surface temperature [K]
            aragonite                   :   aragonite saturation state [-]
            storm                       :   storm category, annually [-]

        :param parameter: parameter to be set
        :param value: default value
        :param pre_date: time-series start before simulation dates [yrs]

        :type parameter: str
        :type value: float, list, tuple, numpy.ndarray, pandas.DataFrame
        :type pre_date: None, int, optional
        """

        def set_value(val):
            """Function to set default value."""
            if pre_date is None:
                return pd.DataFrame({parameter: val}, index=self.dates)

            dates = pd.date_range(self.dates.iloc[0] - pd.DateOffset(years=pre_date), self.dates.iloc[-1], freq='D')
            return pd.DataFrame({parameter: val}, index=dates)

        if self._dates is None:
            msg = f'No dates are defined. ' \
                f'Please, first specify the dates before setting the time-series of {parameter}; ' \
                f'or make use of the \"from_file\"-method.'
            raise TypeError(msg)

        if parameter == 'LAC':
            parameter = 'light_attenuation'

        daily_params = ('light', 'light_attenuation', 'temperature', 'aragonite')
        if parameter in daily_params:
            setattr(self, f'_{parameter}', set_value(value))
        elif parameter == 'storm':
            years = set(self.dates.dt.year)
            self._storm_category = pd.DataFrame(data=value, index=years)
        else:
            msg = f'Entered parameter ({parameter}) not included. See documentation.'
            raise ValueError(msg)

    def from_file(self, parameter, file, folder=None):
        """Read the time-series data from a file.

        Included parameters:
            light                       :   incoming light-intensity [umol photons m-2 s-1]
            LAC / light_attenuation     :   light attenuation coefficient [m-1]
            temperature                 :   sea surface temperature [K]
            aragonite                   :   aragonite saturation state [-]
            storm                       :   storm category, annually [-]

        :param parameter: parameter to be read from file
        :param file: file name, incl. file extension
        :param folder: folder directory, defaults to None

        :type parameter: str
        :type file: str
        :type folder: str, DirConfig, list, tuple, optional
        """
        # TODO: Include functionality to check file's existence
        #  > certain files are necessary: light, temperature

        def date2index(time_series):
            """Function applicable to time-series in Pandas."""
            time_series['date'] = pd.to_datetime(time_series['date'])
            time_series.set_index('date', inplace=True)

        f = DirConfig(folder).config_dir(file)

        if parameter == 'LAC':
            parameter = 'light_attenuation'

        daily_params = ('light', 'light_attenuation', 'temperature', 'aragonite')
        if parameter in daily_params:
            setattr(self, f'__{parameter}', pd.read_csv(f, sep='\t'))
            date2index(getattr(self, f'__{parameter}'))
        elif parameter == 'storm':
            self._storm_category = pd.read_csv(f, sep='\t')
            self._storm_category.set_index('year', inplace=True)
        else:
            msg = f'Entered parameter ({parameter}) not included. See documentation.'
            raise ValueError(msg)
