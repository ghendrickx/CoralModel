"""
Environmental conditions.

Author Gijs G. Hendrickx
"""
import numpy as np
import pandas as pd

from _v2._errors import DataError
from _v2.biophysics import _BasicBiophysics
from _v2.settings import Constants, Processes
from utils.config_directory import DirConfig


class EnvironmentalConditions:
    _light = None
    _light_attenuation = None
    _flow = None
    _temperature = None
    _aragonite = None

    def __init__(self, light=None, light_attenuation=None, flow=None, temperature=None, aragonite=None):
        """
        :param light: light conditions, defaults to None
        :param light_attenuation: light attenuation conditions, defaults to None
        :param flow: flow conditions, defaults to None
        :param temperature: thermal conditions in degree Kelvin, defaults to None
        :param aragonite: aragonite conditions, defaults to None

        :type light: float, int, iterable, optional
        :type light_attenuation: float, int, iterable, optional
        :type flow: float, int, iterable, optional
        :type temperature: float, int, iterable, optional
        :type aragonite: float, int, iterable, optional
        """
        self.set_conditions(
            light=light, light_attenuation=light_attenuation, flow=flow, temperature=temperature, aragonite=aragonite
        )
        self.export_conditions(self)

    @classmethod
    def set_conditions(cls, light=None, light_attenuation=None, flow=None, temperature=None, aragonite=None):
        """
        :param light: light conditions, defaults to None
        :param light_attenuation: light attenuation conditions, defaults to None
        :param flow: flow conditions, defaults to None
        :param temperature: thermal conditions, defaults to None
        :param aragonite: aragonite conditions, defaults to None

        :type light: float, int, iterable, optional
        :type light_attenuation: float, int, iterable, optional
        :type flow: float, int, iterable, optional
        :type temperature: float, int, iterable, optional
        :type aragonite: float, int, iterable, optional
        """
        cls._light = cls.fmt_conditions(light)
        cls._light_attenuation = cls.fmt_conditions(light_attenuation)
        cls._flow = cls.fmt_conditions(flow)
        cls._temperature = cls.fmt_conditions(temperature)
        cls._aragonite = cls.fmt_conditions(aragonite)

    @staticmethod
    def fmt_conditions(conditions):
        """Format environmental conditions as list/float (or None).

        :param conditions: environmental conditions

        :return: environmental conditions
        :rtype: numpy.array, float, None
        """
        if conditions is None:
            return None

        if isinstance(conditions, (float, int, np.ndarray)):
            return conditions

        if isinstance(conditions, pd.DataFrame):
            return conditions.values.flatten()

        return np.array(conditions)

    @classmethod
    def from_snippet(cls, snippet):
        """Set environmental conditions from a snippet from the environmental conditions record.

        :param snippet: snippet of environmental conditions
        :type snippet: _EnvironmentSnippet
        """
        cls(snippet.light, snippet.light_attenuation, snippet.flow, snippet.temperature, snippet.aragonite)

    @classmethod
    def from_file(
            cls, file_name, directory=None, light_col=None, light_attenuation_col=None, flow_col=None,
            temperature_col=None, aragonite_col=None, **kwargs
    ):
        """Set environmental conditions from a file.

        :param file_name: file-name
        :param directory: directory, defaults to None
        :param light_col: column name/index with light conditions, defaults to None
        :param light_attenuation_col: column name/index with light attenuation conditions, defaults to None
        :param flow_col: column name/index with flow conditions, defaults to None
        :param temperature_col: column name/index with thermal conditions, defaults to None
        :param aragonite_col: column name/index with aragonite conditions, defaults to None
        :param kwargs: key-worded arguments for reading file with `pandas.read_csv()`

        :type file_name: str
        :type directory: DirConfig, str, list, tuple, optional
        :type light_col: str, int, optional
        :type light_attenuation_col: str, int, optional
        :type flow_col: str, int, optional
        :type temperature_col: str, int, optional
        :type aragonite_col: str, int, optional
        """
        # read file
        data = pd.read_csv(DirConfig(directory).config_dir(file_name), **kwargs)

        # extract and set data

        def extract_data(col): return None if col is None else data[col]

        light = extract_data(light_col)
        light_attenuation = extract_data(light_attenuation_col)
        flow = extract_data(flow_col)
        temperature = extract_data(temperature_col)
        aragonite = extract_data(aragonite_col)

        # initiate EnvironmentalConditions
        cls(light=light, light_attenuation=light_attenuation, flow=flow, temperature=temperature, aragonite=aragonite)

    @staticmethod
    def export_conditions(env):
        """Set the defined environmental conditions to the biophysical objects."""
        _BasicBiophysics.set_environment(env)

    @property
    def light(self):
        """
        :return: light conditions
        :rtype: numpy.array, None
        """
        if self._light is None:
            self._light = Constants.get_constant('light')
        return self._light

    @property
    def light_attenuation(self):
        """
        :return: light attenuation conditions
        :rtype: numpy.array, None
        """
        if self._light_attenuation is None:
            self._light_attenuation = Constants.get_constant('light_attenuation')
        return self._light_attenuation

    @property
    def flow(self):
        """
        :return: flow conditions
        :rtype: numpy.array, None
        """
        if self._flow is None:
            self._flow = Constants.get_constant('flow')
        return self._flow

    @property
    def temperature(self):
        """
        :return: thermal conditions
        :rtype: numpy.array, None
        """
        if self._temperature is None:
            self._temperature = Constants.get_constant('temperature')
        return self._temperature

    @property
    def aragonite(self):
        """
        :return: aragonite conditions
        :rtype: numpy.array, None
        """
        if self._aragonite is None:
            self._aragonite = Constants.get_constant('aragonite')
        return self._aragonite


class Environment:
    _data = None

    @classmethod
    def get_data_set(cls):
        """Get data set with all environmental data.

        :return: environmental data set
        :rtype: pandas.DataFrame
        """
        return cls._data

    @classmethod
    def get_condition(cls, col):
        """Get a single environmental condition (or the dates).

        :param col: environmental condition key
        :type col: str

        :return: environmental data
        :rtype: pandas.Series
        """
        if col == 'date':
            return cls._dates
        return cls._data[col]

    @classmethod
    def update(cls, date_range):
        """Update environment for biophysical calculations by taking a snippet of the overall environmental time-series.

        :param date_range: snippet range of dates
        """
        snippet = _EnvironmentSnippet(cls.__call__(), date_range)
        EnvironmentalConditions.from_snippet(snippet)

    @classmethod
    def annual_update(cls, year):
        """Update environment for biophysical calculations by taking an annual snippet of the overall environmental
        time-series.

        :param year: year of snippet
        :type year: int
        """
        date_range = f'{year}-01-01', f'{year}-12-31'
        cls.update(date_range)

    @classmethod
    def _set_conditions(cls, conditions):
        """Set environmental conditions.

        :param conditions: environmental conditions
        :type conditions: float, int, list, tuple, numpy.array, pandas.DataFrame, pandas.Series

        :return: environmental conditions
        :rtype: pandas.DataFrame
        """
        if cls._dates is None:
            msg = f'No dates or temporal indications provided.'
            raise DataError(msg)

        if isinstance(conditions, (float, int)):
            return pd.DataFrame(data=conditions, index=cls._dates)

        elif isinstance(conditions, (list, tuple, np.ndarray)):
            if len(conditions) == len(cls._dates):
                return pd.DataFrame(data=conditions, index=cls._dates)
            else:
                msg = f'Length of dates and environmental conditions do not align: ' \
                    f'{len(cls._dates)} =/= {len(conditions)}'
                raise DataError(msg)

        elif isinstance(conditions, (pd.DataFrame, pd.Series)):
            if len(conditions) == len(cls._dates):
                return pd.DataFrame(data=conditions.values, index=cls._dates)
            elif hasattr(conditions, 'columns') and 'date' in conditions.columns:
                col = [c for c in conditions.columns if not c == 'date'][0]
                return pd.DataFrame(data=conditions.loc[conditions['date'] in cls._dates, col])
            else:
                msg = f'Cannot extract data from provided conditions: see \"help(Environment._set_conditions)\".'
                raise DataError(msg)

        else:
            msg = f'Cannot extract data from provided conditions: see \"help(Environment._set_conditions)\".'
            raise DataError(msg)

    @staticmethod
    def _read_file(file_name, directory=None, **kwargs):
        """Extract data from file.

        :param file_name: file name
        :param directory: directory, defaults to None

        :type file_name: str
        :type directory: str, list, tuple, DirConfig

        :return: data from file
        :rtype: pandas.DataFrame
        """
        file = DirConfig(directory).config_dir(file_name)
        return pd.read_csv(file, **kwargs)

    @classmethod
    def set_all_from_file(
            cls, file_name, directory=None, date_col=None, light_col=None, light_attenuation_col=None,
            flow_col=None, temperature_col=None, aragonite_col=None, **kwargs
    ):
        """Set all environmental data from the same file, where the environmental data is a time-series.

        :param file_name: file-name
        :param directory: directory
        :param date_col: column name/index with dates, defaults to None
        :param light_col: column name/index with light conditions, defaults to None
        :param light_attenuation_col: column name/index with light attenuation conditions, defaults to None
        :param flow_col: column name/index with flow conditions, defaults to None
        :param temperature_col: column name/index with thermal conditions, defaults to None
        :param aragonite_col: column name/index with aragonite conditions, defaults to None
        :param kwargs: key-worded arguments for reading file with `pandas.read_csv()`

        :type file_name: str
        :type directory: DirConfig, str, list, tuple, optional
        :type date_col: str, int, optional
        :type light_col: str, int, optional
        :type light_attenuation_col: str, int, optional
        :type flow_col: str, int, optional
        :type temperature_col: str, int, optional
        :type aragonite_col: str, int, optional
        """
        # read file
        data = cls._read_file(file_name=file_name, directory=directory, **kwargs)

        # TODO: auto-fill column names if none are specified:
        #  1. auto-fill column-name if none specified;
        #  2. check if auto-name is present in file: (a) if not, skip, (b) if so, use this column.
        # set dates
        cls.set_dates(data[0 if date_col is None else date_col])
        # set light
        if light_col is not None:
            cls.set_light_conditions(data[light_col])
        # set light attenuation
        if light_attenuation_col is not None:
            cls.set_light_attenuation(data[light_attenuation_col])
        # set flow
        if flow_col is not None:
            cls.set_flow_conditions(data[flow_col])
        # set temperature
        if temperature_col is not None:
            cls.set_thermal_conditions(data[temperature_col])
        # set aragonite
        if aragonite_col is not None:
            cls.set_aragonite_conditions(data[aragonite_col])

    @classmethod
    def _update_data_set(cls, col, values):
        """Update environmental data set per environmental condition.

        :param col: column name/index of environmental condition
        :param values: environmental conditions

        :type col: str, int
        :type values: float, iterable

        :return: environmental conditions
        :rtype: pandas.Series
        """
        # initialise data set
        if cls._data is None:
            cls._data = pd.DataFrame(columns=['date', 'light', 'light_attenuation', 'flow', 'temperature', 'aragonite'])

        # update data set
        cls._data[col] = np.array(values)

        # set dates as index
        if col == 'date':
            cls._data.set_index(col, inplace=True)
            return cls._data.index
        return cls._data[col]

    # dates
    _dates = None

    @classmethod
    def set_dates(cls, dates):
        """Set dates to environmental data set.

        :param dates: dates, or date range
        :type dates: iterable
        """
        if len(dates) == 2:
            dates = pd.date_range(*dates, freq='D')

        # set dates
        cls._dates = cls._update_data_set('date', dates)

    @classmethod
    def set_dates_from_file(cls, file_name, directory=None, **kwargs):
        """Set dates from a file.

        :param file_name: file-name
        :param directory: directory, defaults to None
        :param kwargs: key-worded arguments for reading file with `pandas.read_csv`

        :type file_name: str
        :type directory: DirConfig, str, list, tuple, optional
        """
        dates = cls._read_file(file_name, directory, **kwargs)
        cls.set_dates(dates)

    def _extract_dates_from_environmental_conditions(self, data):
        raise NotImplementedError

    @property
    def dates(self):
        """
        :return: dates
        :rtype: pandas.Series
        """
        return self._dates

    # light conditions
    _light = None
    _light_attenuation = None

    @classmethod
    def set_light_conditions(cls, daily_averages):
        """Set light conditions.

        :param daily_averages: daily-averaged light conditions
        :type daily_averages: float, iterable
        """
        cls._light = cls._update_data_set('light', daily_averages)

    @classmethod
    def set_light_attenuation(cls, daily_averages):
        """Set light attenuation conditions.

        :param daily_averages: daily-averaged light attenuation conditions
        :type daily_averages: float, iterable
        """
        cls._light_attenuation = cls._update_data_set('light_attenuation', daily_averages)

    @classmethod
    def set_light_from_file(cls, file_name, directory=None, **kwargs):
        """Set light conditions from a file.

        :param file_name: file-name
        :param directory: directory, defaults to None
        :param kwargs: key-worded arguments for reading file with `pandas.read_csv`

        :type file_name: str
        :type directory: DirConfig, str, list, tuple, optional
        """
        light = cls._read_file(file_name, directory, **kwargs)
        cls.set_light_conditions(light)

    @classmethod
    def set_light_attenuation_from_file(cls, file_name, directory=None, **kwargs):
        """Set light attenuation conditions from a file.

        :param file_name: file-name
        :param directory: directory, defaults to None
        :param kwargs: key-worded arguments for reading file with `pandas.read_csv`

        :type file_name: str
        :type directory: DirConfig, str, list, tuple, optional
        """
        light_attenuation = cls._read_file(file_name, directory, **kwargs)
        cls.set_light_attenuation(light_attenuation)

    @property
    def light(self):
        """
        :return: light conditions
        :rtype: pandas.Series
        """
        return self._light

    @property
    def light_attenuation(self):
        """
        :return: light attenuation conditions
        :rtype: pandas.Series
        """
        return self._light_attenuation

    # hydrodynamic conditions
    _flow = None
    _storm_category = None

    @classmethod
    def set_flow_conditions(cls, daily_averages):
        """Set flow conditions.

        :param daily_averages: daily-averaged flow conditions
        :type daily_averages: float, iterable
        """
        cls._flow = cls._update_data_set('flow', daily_averages)

    @classmethod
    def set_flow_from_file(cls, file_name, directory=None, **kwargs):
        """Set flow conditions from a file.

        :param file_name: file-name
        :param directory: directory, defaults to None
        :param kwargs: key-worded arguments for reading file with `pandas.read_csv`

        :type file_name: str
        :type directory: DirConfig, str, list, tuple, optional
        """
        flow = cls._read_file(file_name, directory, **kwargs)
        cls.set_flow_conditions(flow)

    @property
    def flow(self):
        """
        :return: flow conditions
        :rtype: pandas.Series
        """
        return self._flow

    # TODO: Determine how to store the storm conditions/categories, which are on an annual basis instead of a daily
    #  frequency.

    @classmethod
    def set_storm_conditions(cls, annual_storm):
        """Set storm conditions, annual storm category.

        :param annual_storm: storm categories
        :type annual_storm: int, iterable
        """
        cls._storm_category = annual_storm

    @classmethod
    def set_storm_from_file(cls, file_name, directory=None, **kwargs):
        """Set storm categories from a file.

        :param file_name: file-name
        :param directory: directory, defaults to None
        :param kwargs: key-worded arguments for reading file with `pandas.read_csv`

        :type file_name: str
        :type directory: DirConfig, str, list, tuple, optional
        """
        cls._storm_category = cls._read_file(file_name, directory, **kwargs)

    @property
    def storm_category(self):
        """
        :return: storm categories
        :rtype: pandas.Series
        """
        return self._storm_category

    # thermal conditions
    _temperature = None
    _temperature_kelvin = None
    _temperature_celsius = None
    _temperature_mmm = None
    __temp_conversion = 273.15
    __thermal_threshold = 100

    @classmethod
    def set_thermal_conditions(cls, daily_averages):
        """Set thermal conditions.

        :param daily_averages: daily-averaged thermal conditions
        :type daily_averages: float, iterable
        """
        cls._temperature = cls._update_data_set('temperature', daily_averages)

    @classmethod
    def set_thermal_from_file(cls, file_name, directory=None, **kwargs):
        """Set thermal conditions from a file.

        :param file_name: file-name
        :param directory: directory, defaults to None
        :param kwargs: key-worded arguments for reading file with `pandas.read_csv`

        :type file_name: str
        :type directory: DirConfig, str, list, tuple, optional
        """
        temperature = cls._read_file(file_name, directory, **kwargs)
        cls.set_thermal_conditions(temperature)

    @property
    def temperature_kelvin(self):
        """Thermal conditions expressed in degrees Kelvin, based on a thermal threshold (100):
         -  all(T) < 100    ->  T + 273.15
         -  all(T) > 100    ->  T

        :return: thermal conditions in degrees Kelvin
        :rtype: pandas.Series
        """
        if self._temperature_kelvin is None and self._temperature is not None:
            self._temperature_kelvin = self._temperature.add(
                self.__temp_conversion if all(self._temperature.values < self.__thermal_threshold) else 0
            )

        return self._temperature_kelvin

    @property
    def temperature_celsius(self):
        """Thermal conditions expressed in degrees Celsius, based on a thermal thershold (100):
         -  all(T) < 100    ->  T
         -  all(T) > 100    ->  T - 273.15

        :return: thermal conditions in degrees Celsius
        :rtype: pandas.Series
        """
        if self._temperature_celsius is None and self._temperature is not None:
            self._temperature_celsius = self._temperature.add(
                self.__temp_conversion if all(self._temperature.values > self.__thermal_threshold) else 0
            )

        return self._temperature_celsius

    @property
    def temperature(self):
        """For computations, the thermal conditions are required in degrees Kelvin. Therefore, the thermal conditions
        are expressed in degrees Kelvin by default.

        :return: thermal conditions in degrees Kelvin
        :rtype: pandas.Series
        """
        return self.temperature_kelvin

    @property
    def temperature_mmm(self):
        """The maximum and minimum monthly means (MMMs) of the thermal conditions.

        :return: MMMS of thermal conditions
        :rtype: pandas.DataFrame
        """
        if self._temperature_mmm is None and Processes.get_process('thermal_acclimation'):
            self._monthly_max_min_mean()

        return self._temperature_mmm

    def _monthly_max_min_mean(self):
        """Determine maximum and minimum monthly means (MMMs) of the thermal time-series. This data is required for the
        inclusion of the thermal acclimation.

        This method requires a sufficiently long time-series of thermal conditions to be valuable; order of decades.
        """
        monthly_mean = self.temperature_kelvin.groupby([
            self.temperature_kelvin.index.year, self.temperature_kelvin.index.month
        ]).agg(['mean'])
        monthly_max_min_mean = monthly_mean.groupby(level=0).agg(['min', 'max'])
        monthly_max_min_mean.columns = monthly_max_min_mean.columns.droplevel([0, 1])
        self._temperature_mmm = monthly_max_min_mean

    # acidic conditions
    _aragonite = None

    @classmethod
    def set_aragonite_conditions(cls, daily_averages):
        """Set aragonite conditions.

        :param daily_averages: daily-averaged aragonite conditions
        :type daily_averages: float, iterable
        """
        cls._update_data_set('aragonite', daily_averages)
        cls._aragonite = cls._data['aragonite']

    @classmethod
    def set_aragonite_from_file(cls, file_name, directory=None, **kwargs):
        """Set aragonite conditions from a file.

        :param file_name: file-name
        :param directory: directory, defaults to None
        :param kwargs: key-worded arguments for reading file with `pandas.read_csv`

        :type file_name: str
        :type directory: DirConfig, str, list, tuple, optional
        """
        cls._aragonite = cls._read_file(file_name, directory, **kwargs)

    @property
    def aragonite(self):
        """
        :return: aragonite conditions
        :rtype: pandas.Series
        """
        return self._aragonite


class _EnvironmentSnippet:

    def __init__(self, environment, date_range):
        """
        :param environment: full environmental conditions
        :param date_range: date range of snippet defined as starting and ending date: ('yyyy-mm-dd', 'yyyy-mm-dd')

        :type environment: Environment
        :type date_range: tuple, list
        """
        self._date_range = date_range
        # light conditions
        self._light = self._get_snippet(environment.light)
        self._light_attenuation = self._get_snippet(environment.light_attenuation)
        # hydrodynamic conditions
        self._flow = self._get_snippet(environment.flow)
        self._storm_category = self._get_snippet(environment.storm_category)
        # thermal conditions
        self._temperature = self._get_snippet(environment.temperature_kelvin)
        self._temperature_mmm = self._get_snippet(environment.temperature_mmm)
        # acidic conditions
        self._aragonite = self._get_snippet(environment.aragonite)

    def _get_snippet(self, conditions):
        """Extract snippet from data set, if the environmental conditions are defined.

        :param conditions: environmental conditions
        :type conditions: pandas.Series, None

        :return: snippet of environmental conditions
        :rtype: pandas.Series, None
        """
        if conditions is None:
            return None

        return conditions[self._date_range[0]:self._date_range[-1]]

    @property
    def date_range(self):
        """
        :return: date-range of snippet
        :rtype: pandas.Series
        """
        return pd.date_range(*self._date_range, freq='D')

    @property
    def light(self):
        """
        :return: light conditions of snippet
        :rtype: pandas.Series
        """
        return self._light

    @property
    def light_attenuation(self):
        """
        :return: light attenuation conditions of snippet
        :rtype: pandas.Series
        """
        return self._light_attenuation

    @property
    def flow(self):
        """
        :return: flow conditions of snippet
        :rtype: pandas.Series
        """
        return self._flow

    @property
    def storm_category(self):
        """
        :return: storm categories of snippet
        :rtype: pandas.Series
        """
        return self._storm_category

    @property
    def temperature(self):
        """
        :return: thermal conditions of snippet
        :rtype: pandas.Series
        """
        return self._temperature

    @property
    def temperature_mmm(self):
        """
        :return: maximum/minimum monthly means of thermal conditions of snippet
        :rtype: pandas.DataFrame
        """
        return self._temperature_mmm

    @property
    def aragonite(self):
        """
        :return: aragonite conditions of snippet
        :rtype: pandas.Series
        """
        return self._aragonite
