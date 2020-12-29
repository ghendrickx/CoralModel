"""
coral_model - loop

@author: Gijs G. Hendrickx
"""

import numpy as np
from tqdm import tqdm

from coral_model import core
from coral_model.core import Light, Flow, Temperature, Photosynthesis, PopulationStates, Calcification, Morphology, \
    Dislodgement, Recruitment
from coral_model.environment import Processes, Constants, Environment
from coral_model.hydrodynamics import Hydrodynamics, BaseHydro
from coral_model.utils import Output, DirConfig, time_series_year


class Simulation:
    """CoralModel simulation."""

    __working_dir = DirConfig()
    __input_dir = None

    output = None

    def __init__(self, environment, processes=None, constants=None, hydrodynamics=None):
        """CoralModel initiation.

        :param environment: environmental conditions
        :param processes: included processes, defaults to None
        :param constants: simulation constants, defaults to None
        :param hydrodynamics: hydrodynamic model, defaults to None

        :type environment: Environment
        :type processes: None, Processes, optional
        :type constants: None, Constants, optional
        :type hydrodynamics: None, str, Hydrodynamics, optional
        """
        self.environment = environment
        core.PROCESSES = Processes() if processes is None else processes
        core.CONSTANTS = Constants(core.PROCESSES) if constants is None else constants
        self.hydrodynamics = hydrodynamics if isinstance(hydrodynamics, Hydrodynamics) else Hydrodynamics(hydrodynamics)

    def set_coordinates(self, xy_coordinates):
        """Set (x,y)-coordinates if nt provided by hydrodynamic model.

        :param xy_coordinates: (x,y)-coordinates [m]
        :type xy_coordinates: tuple, list, numpy.ndarray
        """
        self.hydrodynamics.set_coordinates(xy_coordinates)

    def set_water_depth(self, water_depth):
        """Set water depth if not provided by hydrodynamic model.

        :param water_depth: water depth [m]
        :type water_depth: float, tuple, list, numpy.ndarray
        """
        self.hydrodynamics.set_water_depth(water_depth)

    @property
    def working_dir(self):
        """Working directory.

        :rtype: str
        """
        return self.__working_dir.__str__()

    @working_dir.setter
    def working_dir(self, directory):
        """
        :param directory: working directory
        :type directory: str, list, tuple, DirConfig
        """
        self.__working_dir = directory if isinstance(directory, DirConfig) else DirConfig(directory)

    @property
    def figures_dir(self):
        """Figures directory.

        :rtype: str
        """
        return self.__working_dir.config_dir('figures')

    @property
    def output_dir(self):
        """Output directory.

        :rtype: str
        """
        return self.__working_dir.config_dir('output')

    def set_directories(self, working_dir, input_dir=None):
        """Set directories based on working directory.

        :param working_dir: working directory
        :param input_dir: input directory, defaults to None

        :type working_dir: str, list, tuple, DirConfig
        :type input_dir: str, list, tuple, DirConfig
        """
        self.__working_dir = working_dir if isinstance(working_dir, DirConfig) else DirConfig(working_dir)
        # TODO: Not sure if input_dir is needed in the new setup; check this!
        if input_dir is None:
            self.__input_dir = self.__working_dir.config_dir('input')
        else:
            self.__input_dir = str(input_dir) if isinstance(input_dir, DirConfig) else DirConfig().config_dir(input_dir)

        self.output.folder = self.output_dir
        self.make_directories()

    def make_directories(self):
        """Create directories if not existing."""
        self.__working_dir.create_dir(self.working_dir)
        self.__working_dir.create_dir(self.output_dir)
        self.__working_dir.create_dir(self.figures_dir)

    def set_delft3d_environment(self):
        """Set directories and files of hydrodynamic mode 'Delft3D'."""
        # TODO: Set D3D-files and -directories

    def define_output(self, output_type, lme=True, fme=True, tme=True, pd=True, ps=True, calc=True, md=True):
        """Initiate output files based on requested output data.

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

        if not isinstance(self.output, Output):
            self.output = Output(self.hydrodynamics.xy_coordinates, self.environment.dates[0])

        self.output.define_output(
            output_type=output_type, lme=lme, fme=fme, tme=tme, pd=pd, ps=ps, calc=calc, md=md
        )

    def input_check(self):
        """Check input; if all required data is provided."""
        if self.environment.light is None:
            msg = f'CoralModel simulation cannot run without data on light conditions.'
            raise ValueError(msg)

        if self.environment.temperature is None:
            msg = f'CoralModel simulation cannot run without data on temperature conditions.'
            raise ValueError(msg)

        if self.environment.light_attenuation is None:
            self.environment.set_parameter_values('light_attenuation', core.CONSTANTS.Kd0)
            print(f'Light attenuation coefficient set to default: Kd = {core.CONSTANTS.Kd0} [m-1]')

        if self.environment.aragonite is None:
            self.environment.set_parameter_values('aragonite', core.CONSTANTS.omegaA0)
            print(f'Aragonite saturation state set to default: omega_a0 = {core.CONSTANTS.omegaA0} [-]')

        # TODO: if core.PROCESSES.lme: light, light_attenuation

        if core.PROCESSES.fme:
            if isinstance(self.hydrodynamics.model, BaseHydro):
                msg = f'Flow micro-environment requires the coupling with a hydrodynamic model, ' \
                    f'none is specified. See documentation.'
                raise TypeError(msg)

        if core.PROCESSES.tme:
            if isinstance(self.hydrodynamics.model, BaseHydro):
                msg = f'Thermal micro-environment requires the coupling with a hydrodynamic model, ' \
                    f'none is specified. See documentation.'
                raise TypeError(msg)

        if core.PROCESSES.pfd:
            if isinstance(self.hydrodynamics.model, BaseHydro):
                msg = f'Photosynthetic flow dependency requires the coupling with a hydrodynamic model, ' \
                    f'none is specified. See documentation.'
                raise TypeError(msg)

        # TODO: add other dependencies based on Processes if required

    def initiate(self, coral, x_range=None, y_range=None, value=None):
        """Initiate the coral distribution. The default coral distribution is a full coral cover over the whole domain.
        More complex initial conditions of the coral cover cannot be realised with this method. See the documentation on
        workarounds to achieve this anyway.

        :param coral: coral animal
        :param x_range: minimum and maximum x-coordinate, defaults to None
        :param y_range: minimum and maximum y-coordinate, defaults to None
        :param value: coral cover, defaults to None

        :type coral: Coral
        :type x_range: tuple, optional
        :type y_range: tuple, optional
        :type value: float, optional

        :return: coral animal initiated
        :rtype: Coral
        """
        self.input_check()

        self.hydrodynamics.initiate()
        core.RESHAPE.space = self.hydrodynamics.space

        self.output.initiate_his()
        self.output.initiate_map(coral)

        xy = self.hydrodynamics.xy_coordinates

        if value is None:
            value = 1

        cover = value * np.ones(core.RESHAPE.space)

        if x_range is not None:
            x_min = x_range[0] if x_range[0] is not None else min(xy[:][0])
            x_max = x_range[1] if x_range[1] is not None else max(xy[:][0])
            cover[np.logical_or(xy[:][0] <= x_min, xy[:][0] >= x_max)] = 0

        if y_range is not None:
            y_min = y_range[0] if y_range[0] is not None else min(xy[:][1])
            y_max = y_range[1] if y_range[1] is not None else max(xy[:][1])
            cover[np.logical_or(xy[:][1] <= y_min, xy[:][1] >= y_max)] = 0

        coral.initiate_spatial_morphology(cover)

        return coral

    def exec(self, coral, duration=None):
        """Execute simulation.

        :param coral: coral animal
        :param duration: simulation duration [yrs], defaults to None

        :type coral: Coral
        :type duration: int, optional
        """
        # auto-set duration based on environmental time-series
        if duration is None:
            duration = int(self.environment.dates.iloc[-1].year - self.environment.dates.iloc[0].year)
        years = range(int(self.environment.dates.iloc[0].year), int(self.environment.dates.iloc[0].year + duration))

        with tqdm(range((int(duration)))) as progress:
            for i in progress:
                # set dimensions (i.e. update time-dimension)
                core.RESHAPE.time = len(self.environment.dates.dt.year[self.environment.dates.dt.year == years[i]])

                # if-statement that encompasses all for which the hydrodynamic should be used
                progress.set_postfix(inner_loop=f'update {self.hydrodynamics.model}')
                current_vel, wave_vel, wave_per = self.hydrodynamics.update(coral, storm=False)

                # # environment
                progress.set_postfix(inner_loop='coral environment')
                # light micro-environment
                lme = Light(
                    light_in=time_series_year(self.environment.light, years[i]),
                    lac=time_series_year(self.environment.light_attenuation, years[i]),
                    depth=self.hydrodynamics.water_depth
                )
                lme.rep_light(coral)
                # flow micro-environment
                fme = Flow(current_vel, wave_vel, self.hydrodynamics.water_depth, wave_per)
                fme.velocities(coral, in_canopy=core.PROCESSES.fme)
                fme.thermal_boundary_layer(coral)
                # thermal micro-environment
                tme = Temperature(time_series_year(self.environment.temp_kelvin, years[i]))
                tme.coral_temperature(coral)

                # # physiology
                progress.set_postfix(inner_loop='coral physiology')
                # photosynthetic dependencies
                phd = Photosynthesis(
                    time_series_year(self.environment.light, years[i]),
                    first_year=True if i == 0 else False
                )
                phd.photo_rate(coral, self.environment, years[i])
                # population states
                ps = PopulationStates()
                ps.pop_states_t(coral)
                # calcification
                cr = Calcification()
                cr.calcification_rate(
                    coral, time_series_year(self.environment.aragonite, years[i])
                )
                # # morphology
                progress.set_postfix(inner_loop='coral morphology')
                # morphological development
                mor = Morphology(
                    coral.calc.sum(axis=1),
                    time_series_year(self.environment.light, years[i])
                )
                mor.update(coral)

                # # storm damage
                if self.environment.storm_category is not None:
                    if time_series_year(self.environment.storm_category, years[i]) > 0:
                        progress.set_postfix(inner_loop='storm damage')
                        # update hydrodynamic model
                        current_vel, wave_vel = self.hydrodynamics.update(coral, storm=True)
                        # storm flow environment
                        sfe = Flow(current_vel, wave_vel, None, None)
                        sfe.wave_current()
                        # storm dislodgement criterion
                        sdc = Dislodgement()
                        sdc.update(coral)

                # # recruitment
                progress.set_postfix(inner_loop='coral recruitment')
                # recruitment
                rec = Recruitment()
                rec.update(coral)

                # # export results
                progress.set_postfix(inner_loop='export results')
                # map-file
                self.output.update_map(coral, years[i])
                # his-file
                self.output.update_his(coral, self.environment.dates[self.environment.dates.dt.year == years[i]])

    def finalise(self):
        """Finalise simulation."""
        self.hydrodynamics.finalise()

# TODO: Define folder structure
#  > working directory
#  > figures directory
#  > input directory
#  > output directory
#  > etc.

# TODO: Model initiation IV: OutputFiles
#  > specify output files (i.e. define file names and directories)
#  > specify model data to be included in output files

# TODO: Model initiation V: initial conditions
#  > specify initial morphology
#  > specify initial coral cover
#  > specify carrying capacity

# TODO: Model simulation I: specify SpaceTime

# TODO: Model simulation II: hydrodynamic module
#  > update hydrodynamics
#  > extract variables

# TODO: Model simulation III: coral environment
#  > light micro-environment
#  > flow micro-environment
#  > temperature micro-environment

# TODO: Model simulation IV: coral physiology
#  > photosynthesis
#  > population states
#  > calcification

# TODO: Model simulation V: coral morphology
#  > morphological development

# TODO: Model simulation VI: storm damage
#  > set variables to hydrodynamic module
#  > update hydrodynamics and extract variables
#  > update coral storm survival

# TODO: Model simulation VII: coral recruitment
#  > update recruitment's contribution

# TODO: Model simulation VIII: return morphology
#  > set variables to hydrodynamic module

# TODO: Model simulation IX: export output
#  > write map-file
#  > write his-file

# TODO: Model finalisation


if __name__ == '__main__':
    run = Simulation(Environment(), Processes(), Constants(Processes()))
