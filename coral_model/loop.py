"""
coral_model v3 - loop

@author: Gijs G. Hendrickx
"""

import numpy as np

from coral_model import core, utils
from coral_model.environment import Processes, Constants, Environment
from coral_model.hydrodynamics import Hydrodynamics


# TODO: Write the model execution as a function to be called in "interface.py".
# TODO: Include a model execution in which all processes can be switched on and off; based on Processes. This also
#  includes the main environmental factors, etc.

spacetime = (4, 10)
core.RESHAPE = utils.DataReshape(spacetime)

I0 = np.ones(10)
Kd = np.ones(10)
h = np.ones(4)


lme = core.Light(I0, Kd, h)

print(lme.I0.shape)


class Simulation:
    """CoralModel simulation."""

    __working_dir = None
    __figures_dir = None
    __input_dir = None
    __output_dir = None

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

    def set_directories(self, working_dir, input_dir=None):
        """Set directories based on working directory.

        :param working_dir: working directory
        :param input_dir: input directory, defaults to None

        :type working_dir: str, list, tuple, DirConfig
        :type input_dir: str, list, tuple, DirConfig
        """
        self.__working_dir = working_dir
        self.__input_dir = input_dir

    def set_initial_conditions(self, coral):
        """Define coral animal(s) and set their initial conditions.

        :param coral: coral animal
        :type coral: Coral
        """
        pass

    def define_output(self, map_file, his_file):
        pass

    def exec(self):
        """Execute simulation."""
        pass

    def finalise(self):
        """Finalise simulation."""
        self.hydrodynamics.finalise()

# TODO: Define folder structure
#  > working directory
#  > figures directory
#  > input directory
#  > output directory
#  > etc.

# TODO: Model initiation I: Processes and Constants
#  > specify processes
#  > specify constants

# TODO: Model initiation II: Environment
#  > specify environmental factors (i.e. define file names and directories)

# TODO: Model initiation III: Hydrodynamics
#  > define hydrodynamic module (Delft3D, 1DReef, None)
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
