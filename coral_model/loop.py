"""
coral_model - loop

@author: Gijs G. Hendrickx
"""

import numpy as np

from coral_model import core, utils, environment, hydrodynamics


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
