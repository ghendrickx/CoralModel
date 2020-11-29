"""
coral_model - utils

@author: Gijs G. Hendrickx
"""

import numpy as np
from netCDF4 import Dataset

from utils.config_directory import DirConfig


class SpaceTime:
    """Spacetime-object, which validates the definition of the spacetime dimensions."""

    __spacetime = None

    def __init__(self, spacetime=None):
        """
        :param spacetime: spacetime dimensions, defaults to None
        :type spacetime: None, tuple, optional
        """
        if spacetime is not None:
            self.spacetime = spacetime

    def __repr__(self):
        """Development representation."""
        return f'SpaceTime({self.__spacetime})'

    def __str__(self):
        """Print representation."""
        return str(self.spacetime)

    @property
    def spacetime(self):
        """Spacetime dimensions.

        :rtype: tuple
        """
        if self.__spacetime is None:
            return 1, 1
        return self.__spacetime

    @spacetime.setter
    def spacetime(self, space_time):
        """
        :param space_time: spacetime dimensions
        :type space_time: tuple, list, numpy.ndarray
        """
        if not isinstance(space_time, (tuple, list, np.ndarray)):
            msg = f'spacetime must be of type tuple, {type(space_time)} is given.'
            raise TypeError(msg)

        if not len(space_time) == 2:
            msg = f'spacetime must be of size 2, {len(space_time)} is given.'
            raise ValueError(msg)

        if not all(isinstance(dim, int) for dim in space_time):
            msg = f'spacetime must consist of integers only, {[type(dim) for dim in space_time]} is given.'
            raise TypeError(msg)

        self.__spacetime = tuple(space_time)

    @property
    def space(self):
        """Space dimension.

        :rtype: int
        """
        return self.spacetime[0]

    @space.setter
    def space(self, x):
        """
        :param x: space dimension
        :type x: int
        """
        self.spacetime = (x, self.time)

    @property
    def time(self):
        """Time dimension.

        :rtype: int
        """
        return self.spacetime[1]

    @time.setter
    def time(self, t):
        """
        :param t: time dimension
        :type t: int
        """
        self.spacetime = (self.space, t)


class DataReshape(SpaceTime):
    """Reshape data to create a spacetime matrix."""

    def __init__(self, spacetime=None):
        """
        :param spacetime: spacetime dimensions, defaults to None
        :type spacetime: None, tuple, optional
        """
        super().__init__(spacetime=spacetime)
    
    def variable2matrix(self, variable, dimension):
        """Transform variable to matrix.
        
        :param variable: variable to be transformed
        :param dimension: dimension of :param variable:
            
        :type variable: float, int, list, tuple, numpy.ndarray
        :type dimension: str

        :return: variable as matrix in space-time
        :rtype: numpy.ndarray
        """
        # # input check
        # dimension-type
        dimensions = ('space', 'time')
        if dimension not in dimensions:
            msg = f'{dimension} not in {dimensions}.'
            raise ValueError(msg)

        # dimension-value
        variable = self.variable2array(variable)
        self.dimension_value(variable, dimension)

        # # transformation
        if dimension == 'space':
            return np.tile(variable, (self.time, 1)).transpose()
        elif dimension == 'time':
            return np.tile(variable, (self.space, 1))

    def dimension_value(self, variable, dimension):
        """Check consistency between variable's dimensions and the defined spacetime dimensions.

        :param variable: variable to be checked
        :param dimension: dimension under consideration

        :type variable: list, tuple, numpy.ndarray
        :type dimension: str
        """
        try:
            _ = len(variable)
        except TypeError:
            variable = [variable]

        if not len(variable) == getattr(self, dimension):
            msg = f'Incorrect variable size, {len(variable)} =/= {getattr(self, dimension)}.'
            raise ValueError(msg)

    @staticmethod
    def variable2array(variable):
        """"Transform variable to numpy.array (if float or string).
        
        :param variable: variable to be transformed
        :type variable: float, int, list, numpy.ndarray

        :return: variable as array
        :rtype: numpy.ndarray
        """
        if isinstance(variable, str):
            msg = f'Variable cannot be of {type(variable)}.'
            raise NotImplementedError(msg)
        elif isinstance(variable, (float, int)):
            return np.array([float(variable)])
        elif isinstance(variable, (list, tuple)):
            return np.array(variable)
        elif isinstance(variable, np.ndarray) and not variable.shape:
            return np.array([variable])
        return variable

    def matrix2array(self, matrix, dimension, conversion=None):
        """Transform matrix to array.

        :param matrix: variable as matrix in spacetime
        :param dimension: dimension to convert matrix to
        :param conversion: how to convert the matrix to an array, defaults to None
            None    :   take the last value
            'mean'  :   take the mean value
            'max'   :   take the maximum value
            'min'   :   take the minimum value
            'sum'   :   take the summation

        :type matrix: numpy.ndarray
        :type dimension: str
        :type conversion: None, str, optional

        :return: variable as array
        :rtype: numpy.ndarray
        """
        # # input check
        # dimension-type
        dimensions = ('space', 'time')
        if dimension not in dimensions:
            msg = f'{dimension} not in {dimensions}.'
            raise ValueError(msg)

        # input as numpy.array
        matrix = np.array(matrix)

        # dimension-value
        if not matrix.shape == self.spacetime:
            if not matrix.shape[:2] == self.spacetime:
                msg = f'Matrix-shape does not correspond with spacetime-dimensions:' \
                      f'\n{matrix.shape} =/= {self.spacetime}'
                raise ValueError(msg)

        # conversion-strategy
        conversions = (None, 'mean', 'max', 'min', 'sum')
        if conversion not in conversions:
            msg = f'{conversion} not in {conversions}.'
            raise ValueError(msg)

        # # transformation
        # last position
        if conversion is None:
            if dimension == 'space':
                return matrix[:, -1]
            elif dimension == 'time':
                return matrix[-1, :]

        # conversion
        if dimension == 'space':
            return getattr(matrix, conversion)(axis=1)
        elif dimension == 'time':
            return getattr(matrix, conversion)(axis=0)


class Output:

    _file_name_map = None
    _file_name_his = None
    _folder = DirConfig()

    __map_output = None
    __his_output = None

    __map_data = None
    __his_data = None

    _idx_stations = None

    def __init__(self, xy_coordinates, first_date):
        """Generate output files of CoralModel simulation. Output files are formatted as NetCDF4-files.

        :param xy_coordinates: (x,y)-coordinates
        :param first_date: first date of simulation

        :type xy_coordinates: tuple
        :type first_date: pandas
        """
        self.xy_coordinates = xy_coordinates
        self.space = len(xy_coordinates)

        self.first_date = first_date
        self.first_year = first_date.dt.year

    def define_output(self, output_type, lme=True, fme=True, tme=True, pd=True, ps=True, calc=True, md=True):
        """Define output dictionary.

        :param output_type: mapping or history output
        :param lme: light micro-environment, defaults to True
        :param fme: flow micro-environment, defaults to True
        :param tme: thermal micro-environment, defaults to True
        :param pd: photosynthetic dependencies, defaults to True
        :param ps: population states, defaults to True
        :param calc: calcification rates, defaults to True
        :param md: morphological development, defaults to True

        :type output_type: str
        :type lme: bool, optional
        :type fme: bool, optional
        :type tme: bool, optional
        :type pd: bool, optional
        :type ps: bool, optional
        :type calc: bool, optional
        :type md: bool, optional
        """
        types = ('map', 'his')
        if output_type not in types:
            msg = f'{output_type} not in {types}.'
            raise ValueError(msg)

        setattr(self, f'__{output_type}_output', locals())

    @staticmethod
    def __file_ext(file_name):
        """Ensure NetCDF file extension.

        :param file_name: file name
        :type file_name: str
        """
        if not file_name.endswith('.nc'):
            return f'{file_name}.nc'
        return file_name

    @property
    def folder(self):
        """
        :return: output directory
        :rtype: str
        """
        return self._folder.__str__()

    @folder.setter
    def folder(self, output_folder):
        """Output folder.

        :param output_folder: output folder for output files
        :type output_folder: None, str, list, tuple
        """
        self._folder = DirConfig(home_dir=output_folder)

    @property
    def file_name_map(self):
        """File name of mapping output.

        :rtype: str
        """
        if self._file_name_map is None:
            return 'CoralModel_map.nc'
        return self._file_name_map

    @file_name_map.setter
    def file_name_map(self, file_name):
        """
        :param file_name: file name of mapping output
        :type file_name: None, str
        """
        self._file_name_map = self.__file_ext(file_name)

    @property
    def file_dir_map(self):
        """Full file directory of mapping output.

        :rtype: str
        """
        return self._folder.config_dir(self._file_name_map)

    @property
    def file_name_his(self):
        """File name of history output.

        :rtype: str
        """
        if self._file_name_his is None:
            return 'CoralModel_his.nc'
        return self._file_name_his

    @file_name_his.setter
    def file_name_his(self, file_name):
        """
        :param file_name: file name of history output
        :type file_name: str
        """
        self._file_name_his = self.__file_ext(file_name)

    @property
    def file_dir_his(self):
        """Full file directory of history output.

        :rtype: str
        """
        return self._folder.config_dir(self._file_name_his)

    def initiate_map(self, coral, parameters):
        """Initiate mapping output file in which annual output covering the whole model domain is stored.

        :param coral: coral animal
        :param parameters: parameters to be exported

        :type coral: Coral
        :type parameters: dict
        """
        if any(parameters.values()):
            self.__map_data = Dataset(self.file_name_map, 'w', format='NETCDF4')
            self.__map_data.description = 'Mapped simulation data of the CoralModel.'

            # dimensions
            self.__map_data.createDimension('time', None)
            self.__map_data.createDimension('nmesh2d_face', self.space)

            # variables
            t = self.__map_data.createVariable('time', int, ('time',))
            t.long_name = 'year'
            t.units = 'years since 0 B.C.'

            x = self.__map_data.createVariable('nmesh2d_x', 'f8', ('nmesh2d_face',))
            x.long_name = 'x-coordinate'
            x.units = 'm'

            y = self.__map_data.createVariable('nmesh2d_y', 'f8', ('nmesh2d_face',))
            y.long_name = 'y-coordinate'
            y.units = 'm'

            t[:] = self.first_year
            x[:], y[:] = self.xy_coordinates

            # initial conditions
            if parameters['lme']:
                light_set = self.__map_data.createVariable('Iz', 'f8', ('time', 'nmesh2d_face'))
                light_set.long_name = 'annual mean representative light-intensity'
                light_set.units = 'micro-mol photons m-2 s-1'
                light_set[:, :] = np.zeros(self.space)
            if parameters['fme']:
                flow_set = self.__map_data.createVariable('ucm', 'f8', ('time', 'nmesh2d_face'))
                flow_set.long_name = 'annual mean in-canopy flow'
                flow_set.units = 'm s-1'
                flow_set[:, :] = np.zeros(self.space)
            if parameters['tme']:
                temp_set = self.__map_data.createVariable('Tc', 'f8', ('time', 'nmesh2d_face'))
                temp_set.long_name = 'annual mean coral temperature'
                temp_set.units = 'K'
                temp_set[:, :] = np.zeros(self.space)

                low_temp_set = self.__map_data.createVariable('Tlo', 'f8', ('time', 'nmesh2d_face'))
                low_temp_set.long_name = 'annual mean lower thermal limit'
                low_temp_set.units = 'K'
                low_temp_set[:, :] = np.zeros(self.space)

                high_temp_set = self.__map_data.createVariable('Thi', 'f8', ('time', 'nmesh2d_face'))
                high_temp_set.long_name = 'annual mean upper thermal limit'
                high_temp_set.units = 'K'
                high_temp_set[:, :] = np.zeros(self.space)
            if parameters['pd']:
                pd_set = self.__map_data.createVariable('PD', 'f8', ('time', 'nmesh2d_face'))
                pd_set.long_name = 'annual sum photosynthetic rate'
                pd_set.units = '-'
                pd_set[:, :] = np.zeros(self.space)
            if parameters['ps']:
                pt_set = self.__map_data.createVariable('PT', 'f8', ('time', 'nmesh2d_face'))
                pt_set.long_name = 'total living coral population at the end of the year'
                pt_set.units = '-'
                pt_set[:, :] = coral.living_cover

                ph_set = self.__map_data.createVariable('PH', 'f8', ('time', 'nmesh2d_face'))
                ph_set.long_name = 'healthy coral population at the end of the year'
                ph_set.units = '-'
                ph_set[:, :] = coral.living_cover

                pr_set = self.__map_data.createVariable('PR', 'f8', ('time', 'nmesh2d_face'))
                pr_set.long_name = 'recovering coral population at the end of the year'
                pr_set.units = '-'
                pr_set[:, :] = np.zeros(self.space)

                pp_set = self.__map_data.createVariable('PP', 'f8', ('time', 'nmesh2d_face'))
                pp_set.long_name = 'pale coral population at the end of the year'
                pp_set.units = '-'
                pp_set[:, :] = np.zeros(self.space)

                pb_set = self.__map_data.createVariable('PB', 'f8', ('time', 'nmesh2d_face'))
                pb_set.long_name = 'bleached coral population at the end of the year'
                pb_set.units = '-'
                pb_set[:, :] = np.zeros(self.space)
            if parameters['calc']:
                calc_set = self.__map_data.createVariable('G', 'f8', ('time', 'nmesh2d_face'))
                calc_set.long_name = 'annual sum calcification rate'
                calc_set.units = 'kg m-2 yr-1'
                calc_set[:, :] = np.zeros(self.space)
            if parameters['md']:
                dc_set = self.__map_data.createVariable('dc', 'f8', ('time', 'nmesh2d_face'))
                dc_set.long_name = 'coral plate diameter'
                dc_set.units = 'm'
                dc_set[:, :] = coral.dc

                hc_set = self.__map_data.createVariable('hc', 'f8', ('time', 'nmesh2d_face'))
                hc_set.long_name = 'coral height'
                hc_set.units = 'm'
                hc_set[:, :] = coral.hc

                bc_set = self.__map_data.createVariable('bc', 'f8', ('time', 'nmesh2d_face'))
                bc_set.long_name = 'coral base diameter'
                bc_set.units = 'm'
                bc_set[:, :] = coral.bc

                tc_set = self.__map_data.createVariable('tc', 'f8', ('time', 'nmesh2d_face'))
                tc_set.long_name = 'coral plate thickness'
                tc_set.units = 'm'
                tc_set[:, :] = coral.tc

                ac_set = self.__map_data.createVariable('ac', 'f8', ('time', 'nmesh2d_face'))
                ac_set.long_name = 'coral axial distance'
                ac_set.units = 'm'
                ac_set[:, :] = coral.ac

                vc_set = self.__map_data.createVariable('Vc', 'f8', ('time', 'nmesh2d_face'))
                vc_set.long_name = 'coral volume'
                vc_set.units = 'm3'
                vc_set[:, :] = coral.volume
            self.__map_data.close()

    def update_map(self, coral, parameters, year):
        """Write data as annual output covering the whole model domain.

        :param coral: coral animal
        :param parameters: parameters to be exported
        :param year: simulation year

        :type coral: Coral
        :type parameters: dict
        :type year: int
        """
        if any(parameters.values()):
            self.__map_data = Dataset(self.file_dir_map, mode='a')

            i = int(year - self.first_year)
            self.__map_data['time'][i] = year
            if parameters['lme']:
                self.__map_data['Iz'][-1, :] = coral.light
            if parameters['fme']:
                self.__map_data['ucm'][-1, :] = coral.ucm
            if parameters['tme']:
                self.__map_data['Tc'][-1, :] = coral.temp[:, -1]
                self.__map_data['Tlo'][-1, :] = coral.Tlo if len(coral.Tlo) > 1 else coral.Tlo * np.ones(self.space)
                self.__map_data['Thi'][-1, :] = coral.Thi if len(coral.Thi) > 1 else coral.Thi * np.ones(self.space)
            if parameters['pd']:
                self.__map_data['PD'][-1, :] = coral.photo_rate.mean(axis=1)
            if parameters['ps']:
                self.__map_data['PT'][-1, :] = coral.pop_states[:, -1, :].sum(axis=1)
                self.__map_data['PH'][-1, :] = coral.pop_states[:, -1, 0]
                self.__map_data['PR'][-1, :] = coral.pop_states[:, -1, 1]
                self.__map_data['PP'][-1, :] = coral.pop_states[:, -1, 2]
                self.__map_data['PB'][-1, :] = coral.pop_states[:, -1, 3]
            if parameters['calc']:
                self.__map_data['calc'][-1, :] = coral.calc.sum(axis=1)
            if parameters['md']:
                self.__map_data['dc'][-1, :] = coral.dc
                self.__map_data['hc'][-1, :] = coral.hc
                self.__map_data['bc'][-1, :] = coral.bc
                self.__map_data['tc'][-1, :] = coral.tc
                self.__map_data['ac'][-1, :] = coral.ac
                self.__map_data['Vc'][-1, :] = coral.volume

            self.__map_data.close()

    @property
    def idx_stations(self):
        """Space indices based on the (x,y)-coordinates of the stations.

        :rtype: tuple
        """
        return self._idx_stations

    def set_idx_stations(self, xy_stations):
        """Determine space indices based on the (x,y)-coordinates of the stations.

        :param xy_stations: (x,y)-coordinates stations
        :type xy_stations: tuple
        """
        x, y = self.xy_coordinates
        x_station, y_station = xy_stations
        idx = np.zeros(len(xy_stations[0]))

        for s in range(len(idx)):
            idx[s] = np.argmin((x - x_station[s]) ** 2 + (y - y_station[s]) ** 2)

        self._idx_stations = idx.astype(int)

    def initiate_his(self, parameters, xy_stations):
        """Initiate history output file in which daily output at predefined locations within the model is stored.

        :param parameters: parameters to be exported
        :param xy_stations: (x,y)-coordinates of virtual stations

        :type parameters: dict
        :type xy_stations: tuple
        """
        if any(parameters.values()):
            self.__his_data = Dataset(self.file_dir_his, 'w', format='NETCDF4')
            self.__his_data.description = 'Historic simulation data of the CoralModel'

            # dimensions
            self.__his_data.createDimension('time', None)
            self.__his_data.createDimension('stations', len(self.idx_stations))

            # variables
            t = self.__his_data.createVariable('time', 'f8', ('time',))
            t.long_name = f'days since {self.first_date}'
            t.units = 'days'

            x = self.__his_data.createVariable('station_x_coordinate', 'f8', ('stations',))
            y = self.__his_data.createVariable('station_y_coordinate', 'f8', ('stations',))

            # setup data set
            self.set_idx_stations(xy_stations)
            x[:], y[:] = xy_stations

            if parameters['lme']:
                light_set = self.__his_data.createVariable('Iz', 'f8', ('time', 'stations'))
                light_set.long_name = 'representative light-intensity'
                light_set.units = 'micro-mol photons m-2 s-1'
            if parameters['fme']:
                flow_set = self.__his_data.createVariable('ucm', 'f8', ('time', 'stations'))
                flow_set.long_name = 'in-canopy flow'
                flow_set.units = 'm s-1'
            if parameters['tme']:
                temp_set = self.__his_data.createVariable('Tc', 'f8', ('time', 'stations'))
                temp_set.long_name = 'coral temperature'
                temp_set.units = 'K'

                low_temp_set = self.__his_data.createVariable('Tlo', 'f8', ('time', 'stations'))
                low_temp_set.long_name = 'lower thermal limit'
                low_temp_set.units = 'K'

                high_temp_set = self.__his_data.createVariable('Thi', 'f8', ('time', 'stations'))
                high_temp_set.long_name = 'upper thermal limit'
                high_temp_set.units = 'K'
            if parameters['pd']:
                pd_set = self.__his_data.createVariable('PD', 'f8', ('time', 'stations'))
                pd_set.long_name = 'photosynthetic rate'
                pd_set.units = '-'
            if parameters['ps']:
                pt_set = self.__his_data.createVariable('PT', 'f8', ('time', 'stations'))
                pt_set.long_name = 'total coral population'
                pt_set.units = '-'

                ph_set = self.__his_data.createVariable('PH', 'f8', ('time', 'stations'))
                ph_set.long_name = 'healthy coral population'
                ph_set.units = '-'

                pr_set = self.__his_data.createVariable('PR', 'f8', ('time', 'stations'))
                pr_set.long_name = 'recovering coral population'
                pr_set.units = '-'

                pp_set = self.__his_data.createVariable('PP', 'f8', ('time', 'stations'))
                pp_set.long_name = 'pale coral population'
                pp_set.units = '-'

                pb_set = self.__his_data.createVariable('PB', 'f8', ('time', 'stations'))
                pb_set.long_name = 'bleached coral population'
                pb_set.units = '-'
            if parameters['calc']:
                calc_set = self.__his_data.createVariable('G', 'f8', ('time', 'stations'))
                calc_set.long_name = 'calcification'
                calc_set.units = 'kg m-2 d-1'
            if parameters['md']:
                dc_set = self.__his_data.createVariable('dc', 'f8', ('time', 'stations'))
                dc_set.long_name = 'coral plate diameter'
                dc_set.units = 'm'

                hc_set = self.__his_data.createVariable('hc', 'f8', ('time', 'stations'))
                hc_set.long_name = 'coral height'
                hc_set.units = 'm'

                bc_set = self.__his_data.createVariable('bc', 'f8', ('time', 'stations'))
                bc_set.long_name = 'coral base diameter'
                bc_set.units = 'm'

                tc_set = self.__his_data.createVariable('tc', 'f8', ('time', 'stations'))
                tc_set.long_name = 'coral plate thickness'
                tc_set.units = 'm'

                ac_set = self.__his_data.createVariable('ac', 'f8', ('time', 'stations'))
                ac_set.long_name = 'coral axial distance'
                ac_set.units = 'm'

                vc_set = self.__his_data.createVariable('Vc', 'f8', ('time', 'stations'))
                vc_set.long_name = 'coral volume'
                vc_set.units = 'm3'
            self.__his_data.close()

    def update_his(self, coral, parameters, dates):
        """Write data as daily output at predefined locations within the model domain.

        :param coral: coral animal
        :param parameters: parameters to be exported
        :param dates: dates of simulation year

        :type coral: Coral
        :type parameters: dict
        :type dates: pandas
        """
        if any(parameters.values()):
            self.__his_data = Dataset(self.file_dir_his, mode='a')

            y_dates = dates.reset_index(drop=True)
            ti = (y_dates - self.first_date).dt.days.values
            self.__his_data['time'][ti] = y_dates.values
            if parameters['lme']:
                self.__his_data['Iz'][ti, :] = coral.light[self.idx_stations, :].transpose()
            if parameters['fme']:
                self.__his_data['ucm'][ti, :] = np.tile(coral.ucm, (len(y_dates), 1))[:, self.idx_stations]
            if parameters['tme']:
                self.__his_data['Tc'][ti, :] = coral.temp[self.idx_stations, :].transpose()
                if len(coral.Tlo) > 1 and len(coral.Thi) > 1:
                    self.__his_data['Tlo'][ti, :] = np.tile(coral.Tlo, (len(y_dates), 1))[:, self.idx_stations]
                    self.__his_data['Thi'][ti, :] = np.tile(coral.Thi, (len(y_dates), 1))[:, self.idx_stations]
                else:
                    self.__his_data['Tlo'][ti, :] = coral.Tlo * np.ones((len(y_dates), len(self.idx_stations)))
                    self.__his_data['Thi'][ti, :] = coral.Thi * np.ones((len(y_dates), len(self.idx_stations)))
            if parameters['pd']:
                self.__his_data['PD'][ti, :] = coral.photo_rate[self.idx_stations, :].transpose()
            if parameters['ps']:
                self.__his_data['PT'][ti, :] = coral.pop_states[self.idx_stations, :, :].sum(axis=2).transpose()
                self.__his_data['PH'][ti, :] = coral.pop_states[self.idx_stations, :, 0].transpose()
                self.__his_data['PR'][ti, :] = coral.pop_states[self.idx_stations, :, 1].transpose()
                self.__his_data['PP'][ti, :] = coral.pop_states[self.idx_stations, :, 2].transpose()
                self.__his_data['PB'][ti, :] = coral.pop_states[self.idx_stations, :, 3].transpose()
            if parameters['calc']:
                self.__his_data['G'][ti, :] = coral.calc[self.idx_stations, :].transpose()
            if parameters['md']:
                self.__his_data['dc'][ti, :] = np.tile(coral.dc, (len(y_dates), 1))[:, self.idx_stations]
                self.__his_data['hc'][ti, :] = np.tile(coral.hc, (len(y_dates), 1))[:, self.idx_stations]
                self.__his_data['bc'][ti, :] = np.tile(coral.bc, (len(y_dates), 1))[:, self.idx_stations]
                self.__his_data['tc'][ti, :] = np.tile(coral.tc, (len(y_dates), 1))[:, self.idx_stations]
                self.__his_data['ac'][ti, :] = np.tile(coral.ac, (len(y_dates), 1))[:, self.idx_stations]
                self.__his_data['Vc'][ti, :] = np.tile(coral.volume, (len(y_dates), 1))[:, self.idx_stations]

            self.__his_data.close()


def coral_only_function(coral, function, args, no_cover_value=0):
    """Only execute the function when there is coral cover.

    :param coral: coral object
    :param function: function to be executed
    :param args: input arguments of the function
    :param no_cover_value: default value in absence of coral cover

    :type coral: Coral
    :type args: tuple
    :type no_cover_value: float, optional
    """
    try:
        size = len(coral.cover)
    except TypeError:
        size = 1

    args = list(args)
    for i, arg in enumerate(args):
        if isinstance(arg, (float, int)) or (isinstance(arg, np.ndarray) and not arg.shape):
            args[i] = np.repeat(arg, size)
        elif not len(arg) == size:
            msg = f'Sizes do not match up, {len(arg)} =/= {size}.'
            raise ValueError(msg)

    output = no_cover_value * np.ones(size)
    output[coral.cover > 0] = function(*[
        arg[coral.cover > 0] for arg in args
    ])
    return output
