"""
coral_model v3 - interface

@author: Gijs G. Hendrickx
"""
from coral_model.core import Coral
from coral_model.environment import Environment, Processes, Constants
from coral_model.hydrodynamics import Hydrodynamics
from coral_model.loop import Simulation
from coral_model.utils import DirConfig

# environment definition
environment = Environment()
environment.set_dates('01-01-2000', '01-01-2005')
environment.set_parameter_values('light', 600)
environment.set_parameter_values('temperature', 28, 10)

# processes and constants
processes = Processes(fme=False, tme=False, pfd=False)
constants = Constants(processes)

# hydrodynamic model
hydrodynamics = Hydrodynamics(mode='Delft3D')
hydrodynamics.model.working_dir = r'P:\11202744-008-vegetation-modelling\students\GijsHendrickx\models\MiniModel'
hydrodynamics.model.d3d_home = (
        'P:\\11202744-008-vegetation-modelling', 'code_1709',
        'windows', 'oss_artifacts_x64_63721', 'x64'
)
hydrodynamics.model.mdu = r'flow\FlowFM.mdu'
hydrodynamics.model.config = r'dimr_config.xml'
hydrodynamics.set_update_intervals(300, 300)
hydrodynamics.model.initiate()
print(hydrodynamics.model.settings)

# initiation
run = Simulation(environment, processes, constants, hydrodynamics)
# run.set_coordinates((0, 0))
# run.set_water_depth(10)

run.define_output('map', fme=False)
run.define_output('his', fme=False)
run.output.xy_stations = (0, 0)

run.set_directories(DirConfig(home_dir=r'P:\11202744-008-vegetation-modelling\students\GijsHendrickx\models\MiniModel'))

coral = Coral(.1, .1, .05, .05, .2)
coral = run.initiate(coral)

# simulation
run.exec(coral)

run.finalise()
