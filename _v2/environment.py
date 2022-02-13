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
        self.set_conditions(
            light=light, light_attenuation=light_attenuation, flow=flow, temperature=temperature, aragonite=aragonite
        )

    @classmethod
    def set_conditions(cls, light, light_attenuation, flow, temperature, aragonite):
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
            return

        if isinstance(conditions, (float, int, np.ndarray)):
            return conditions

        if isinstance(conditions, pd.DataFrame):
            return conditions.values.flatten()

        return np.array(conditions)

    @property
    def light(self):
        return self._light

    @property
    def light_attenuation(self):
        return Constants().lac_default if self._light_attenuation is None else self._light_attenuation

    @property
    def flow(self):
        return self._flow

    @property
    def temperature(self):
        return self._temperature

    @property
    def aragonite(self):
        return self._aragonite


class Environment:

    @classmethod
    def update(cls, date_range):
        """Update environment for biophysical calculations by taking a snippet of the overall environmental time-series.

        :param date_range: snippet range of dates
        """
        environment = _EnvironmentSnippet(cls.__call__(), date_range)
        _BasicBiophysics.set_environment(environment)

    @classmethod
    def _set_conditions(cls, conditions):
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
        file = DirConfig(directory).config_dir(file_name)
        return pd.read_csv(file, **kwargs)

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
        pass

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

    # storm conditions
    _storm_category = None

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
        if self._temperature is not None and all(self._temperature < 100):
            return self.temperature + self.__temp_conversion
        return self._temperature

    @property
    def temperature_celsius(self):
        if self._temperature is not None and all(self._temperature > 100):
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
        # storm conditions
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
