"""
coral_model - loop

@author: Gijs G. Hendrickx
"""

import numpy as np

from coral_model import core, utils
from coral_model.environment import Processes, Constants, Environment
from coral_model.hydrodynamics import Hydrodynamics


# TODO: Write the model execution as a function to be called in "interface.py".
# TODO: Include a model execution in which all processes can be switched on and off; based on Processes. This also
#  includes the main environmental factors, etc.
from coral_model.utils import Output
from utils.config_directory import DirConfig

spacetime = (4, 10)
core.RESHAPE = utils.DataReshape(spacetime)

I0 = np.ones(10)
Kd = np.ones(10)
h = np.ones(4)


lm_env = core.Light(I0, Kd, h)

print(lm_env.I0.shape)


class Simulation:
    """CoralModel simulation."""

    __working_dir = DirConfig()
    __input_dir = None

    output = None

    def __init__(self, environment, processes, constants, hydrodynamics=None):
        """CoralModel initiation.

        :param environment: environmental conditions
        :param processes: included processes
        :param constants: simulation constants
        :param hydrodynamics: hydrodynamic model, defaults to None

        :type environment: Environment
        :type processes: Processes
        :type constants: Constants
        :type hydrodynamics: None, Hydrodynamics, optional
        """
        self.environment = environment
        core.PROCESSES = processes
        core.CONSTANTS = constants
        self.hydrodynamics = Hydrodynamics(hydrodynamics)

        self.output = Output(self.hydrodynamics.xy_coordinates, environment.dates[0])
        [self.define_output(output_type) for output_type in ('map', 'his')]

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

    def make_directories(self):
        """Create directories if not existing."""
        pass

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

        self.output.define_output(**locals())

    def initiate(self, coral):
        pass

    def exec(self, coral, duration):
        """Execute simulation.

        :param coral: coral animal
        :param duration: simulation duration [yrs]

        :type coral: Coral
        :type duration: int
        """

    def finalise(self):
        """Finalise simulation."""
        self.hydrodynamics.finalise()

# TODO: Define folder structure
#  > working directory
#  > figures directory
#  > input directory
#  > output directory
#  > etc.

# TODO: Model initiation III: Hydrodynamics
#  > define hydrodynamic module (Delft3D, Reef1D, Reef0D, None)
#  > initiate hydrodynamic module

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
    run = Simulation(Environment, Processes, Constants)
