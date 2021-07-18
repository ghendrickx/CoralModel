import pandas as pd

from utils.config_directory import DirConfig


class Environment:

    def __init__(self):
        pass

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

    # light environment
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

    # thermal environment
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
