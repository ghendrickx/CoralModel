"""
Environmental conditions.

Author Gijs G. Hendrickx
"""
import numpy as np
import pandas as pd

from _v2._errors import DataError
from _v2.biophysics import _BasicBiophysics
from _v2.settings import Constants
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
    def set_conditions(cls, light, light_attenuation, flow, temperature, aragonite):
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
        return self._light

    @property
    def light_attenuation(self):
        """
        :return: light attenuation conditions
        :rtype: numpy.array, None
        """
        if self._light_attenuation is None:
            self._light_attenuation = Constants.get_constant('lac_default')
        return self._light_attenuation

    @property
    def flow(self):
        """
        :return: flow conditions
        :rtype: numpy.array, None
        """
        return self._flow

    @property
    def temperature(self):
        """
        :return: thermal conditions
        :rtype: numpy.array, None
        """
        return self._temperature

    @property
    def aragonite(self):
        """
        :return: aragonite conditions
        :rtype: numpy.array, None
        """
        return self._aragonite


class Environment:

    @classmethod
    def update(cls, date_range):
        """Update environment for biophysical calculations by taking a snippet of the overall environmental time-series.

        :param date_range: snippet range of dates
        """
        snippet = _EnvironmentSnippet(cls.__call__(), date_range)
        _BasicBiophysics.set_environment(snippet)

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
    def _from_file(file_name, directory=None, **kwargs):
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
        data = cls._from_file(file_name=file_name, directory=directory, **kwargs)

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

    # dates
    _dates = None

    @classmethod
    def set_dates(cls, dates):
        if len(dates) == 2:
            d = pd.date_range(dates[0], dates[1], freq='D')
            cls._dates = pd.DataFrame({'date': d})
        else:
            cls._dates = dates

    @classmethod
    def set_dates_from_file(cls, file_name, directory=None, **kwargs):
        cls._dates = cls._from_file(file_name, directory, **kwargs)

    def _extract_dates_from_environmental_conditions(self, data):
        raise NotImplementedError

    # light conditions
    _light = None
    _light_attenuation = None

    @classmethod
    def set_light_conditions(cls, daily_averages):
        cls._light = daily_averages

    @classmethod
    def set_light_attenuation(cls, daily_averages):
        cls._light_attenuation = daily_averages

    @classmethod
    def set_light_from_file(cls, file_name, directory=None, **kwargs):
        cls._light = cls._from_file(file_name, directory, **kwargs)

    @classmethod
    def set_light_attenuation_from_file(cls, file_name, directory=None, **kwargs):
        cls._light_attenuation = cls._from_file(file_name, directory, **kwargs)

    @property
    def light(self):
        return self._light

    @property
    def light_attenuation(self):
        return self._light_attenuation

    # hydrodynamic conditions
    _flow = None
    _storm_category = None

    @classmethod
    def set_flow_conditions(cls, flow):
        cls._flow = flow

    @classmethod
    def set_flow_from_file(cls, file_name, directory=None, **kwargs):
        cls._flow = cls._from_file(file_name, directory, **kwargs)

    @property
    def flow(self):
        return self._flow

    @classmethod
    def set_storm_conditions(cls, annual_storm):
        cls._storm_category = annual_storm

    @classmethod
    def set_storm_from_file(cls, file_name, directory=None, **kwargs):
        cls._storm_category = cls._from_file(file_name, directory, **kwargs)

    @property
    def storm_category(self):
        return self._storm_category

    # thermal conditions
    _temperature = None
    _temperature_mmm = None
    __temp_conversion = 273.15

    @classmethod
    def set_thermal_conditions(cls, daily_averages):
        cls._temperature = daily_averages

    @classmethod
    def set_thermal_from_file(cls, file_name, directory=None, **kwargs):
        cls._temperature = cls._from_file(file_name, directory, **kwargs)

    @property
    def temperature(self):
        return self._temperature

    @property
    def temperature_kelvin(self):
        if self._temperature is not None and all(self._temperature.values < 100):
            return self.temperature + self.__temp_conversion
        return self._temperature

    @property
    def temperature_celsius(self):
        if self._temperature is not None and all(self._temperature.values > 100):
            return self._temperature - self.__temp_conversion
        return self._temperature

    @property
    def temperature_mmm(self):
        if self._temperature_mmm is None:
            self._monthly_maximin_mean()
        return self._temperature_mmm

    def _monthly_maximin_mean(self):
        monthly_mean = self.temperature_kelvin.groupby([
            self.temperature_kelvin.index.year, self.temperature_kelvin.index.month
        ]).agg(['mean'])
        monthly_maximin_mean = monthly_mean.groupby(level=0).agg(['min', 'max'])
        monthly_maximin_mean.columns = monthly_maximin_mean.columns.droplevel([0, 1])
        self._temperature_mmm = monthly_maximin_mean

    # acidic conditions
    _aragonite = None

    @classmethod
    def set_aragonite_conditions(cls, daily_averages):
        cls._aragonite = daily_averages

    @classmethod
    def set_aragonite_from_file(cls, file_name, directory=None, **kwargs):
        cls._aragonite = cls._from_file(file_name, directory, **kwargs)

    @property
    def aragonite(self):
        return self._aragonite


class _EnvironmentSnippet:

    def __init__(self, environment, date_range):
        """
        :param environment: full environmental conditions
        :param date_range: date range of snippet

        :type environment: Environment
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
        return conditions[self._date_range]

    @property
    def date_range(self):
        return self._date_range

    @property
    def light(self):
        return self._light

    @property
    def light_attenuation(self):
        return self._light_attenuation

    @property
    def flow(self):
        return self._flow

    @property
    def storm_category(self):
        return self._storm_category

    @property
    def temperature(self):
        return self._temperature

    @property
    def temperature_mmm(self):
        return self._temperature_mmm

    @property
    def aragonite(self):
        return self._aragonite
