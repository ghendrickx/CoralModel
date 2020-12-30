"""
coral_model v3 - interface

@author: Gijs G. Hendrickx
"""
#from coral_model.core import Coral
#from coral_model.environment import Environment, Processes, Constants
#from coral_model.loop import Simulation
#from coral_model.utils import DirConfig

from core import Coral
from environment import Environment, Processes, Constants
from loop import Simulation
from utils import DirConfig

# environment definition
environment = Environment()
environment.set_dates('01-01-2000', '01-01-2010')
environment.set_parameter_values('light', 600)
environment.set_parameter_values('temperature', 28, 10)

# processes and constants
processes = Processes(fme=False, tme=False, pfd=False)
constants = Constants(processes)

# initiation
run = Simulation(environment, processes, constants)
run.set_coordinates((0, 0))
run.set_water_depth(10)

run.define_output('map', fme=False)
run.define_output('his', fme=False)
run.output.xy_stations = (0, 0)

run.set_directories(DirConfig(home_dir=r'P:\11202744-008-vegetation-modelling\students\GijsHendrickx\models\MiniModel3'))

coral = Coral(.1, .1, .05, .05, .2)
coral = run.initiate(coral)

# simulation
run.exec(coral)

run.finalise()
