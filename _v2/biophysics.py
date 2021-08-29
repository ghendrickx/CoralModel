import logging

import numpy as np
from scipy.optimize import newton

from _v2.coral import Coral
from _v2.settings import Constants, Processes

LOG = logging.getLogger(__name__)


class _BasicBiophysics:

    _environment = None
    _hydrodynamics = None
    _constants = Constants()
    _processes = Processes()

    def __init__(self, coral_reef):
        """Initiate biophysical process: Update coral reef.

        :param coral_reef: grid of corals, i.e. coral reef
        :type coral_reef: Grid
        """
        self.update(coral_reef)

    @classmethod
    def set_environment(cls, environment):
        """
        :param environment: environmental conditions
        :type environment: _EnvironmentSnippet
        """
        cls._environment = environment

    @classmethod
    def set_hydrodynamics(cls, hydrodynamics):
        """
        :param hydrodynamics: hydrodynamic conditions
        :type hydrodynamics: Hydrodynamics
        """
        cls._hydrodynamics = hydrodynamics

    @classmethod
    def set_constants(cls, constants):
        """
        :param constants: simulation constants
        :type constants: Constants
        """
        cls._constants = constants

    @classmethod
    def set_processes(cls, processes):
        """
        :param processes: simulation processes
        :type processes: Processes
        """
        cls._processes = processes

    def _update(self, cell):
        """Update corals.

        This method is to be overwritten by every biophysical process.

        :param cell: grid cell
        :type cell: Cell
        """

    def update(self, coral_reef):
        """Update corals.

        :param coral_reef: grid of corals, i.e. coral reef
        :type coral_reef: Grid
        """
        [self._update(cell) for cell in coral_reef.cells if cell.capacity > 0]

    @property
    def environment(self):
        """
        :return: environmental conditions
        :rtype: _EnvironmentSnippet
        """
        return self._environment

    @property
    def hydrodynamics(self):
        """
        :return: hydrodynamic conditions
        :rtype: Hydrodynamics
        """
        return self._hydrodynamics

    @property
    def constants(self):
        """
        :return: simulation constants
        :rtype: Constants
        """
        return self._constants

    @property
    def processes(self):
        """
        :return: simulation processes
        :rtype: Processes
        """
        return self._processes


class Light(_BasicBiophysics):

    def _update(self, cell):
        """Update corals: Light micro-environment.

        :param cell: grid cell
        :type cell: Cell
        """
        [self._representative_light(coral, cell.water_depth) for coral in cell.corals]

    def _representative_light(self, coral, water_depth):
        """Representative light-intensity.

        :param coral: coral
        :param water_depth: water depth

        :type coral: Coral
        :type water_depth: float
        """
        base = self._base_light(coral, water_depth)

        # light catchment per morphological section
        # > top of plate
        top = .25 * np.pi * coral.morphology.diameter ** 2 * self.environment.light * np.exp(
            -self.environment.light_attenuation * (water_depth - coral.morphology.height)
        )
        # > side of plate
        side_top = self._side_correction(coral, water_depth) * (
            np.pi * coral.morphology.diameter * self.environment.light / self.environment.light_attenuation * (
                np.exp(-self.environment.light_attenuation * (water_depth - coral.morphology.height)) -
                np.exp(-self.environment.light_attenuation * (
                        water_depth - coral.morphology.height + coral.morphology.plate_thickness
                ))
            )
        )
        # > side of base
        side_base = self._side_correction(coral, water_depth) * (
            np.pi * coral.morphology.base_diameter * self.environment.light / self.environment.light_attenuation * (
                np.exp(-self.environment.light_attenuation * (water_depth - base)) -
                np.exp(-self.environment * water_depth)
            )
        )
        # > total
        total = sum([top, side_top, side_base])

        # biomass-averaged
        biomass = self._biomass(coral, water_depth)
        light = total / biomass

        # set light micro-environment
        coral.set_characteristic('light', light)

    def _biomass(self, coral, water_depth):
        """Coral biomass, defined as light-receiving surface.

        :param coral: coral
        :param water_depth: water depth

        :type coral: Coral
        :type water_depth: float

        :return: coral biomass
        :rtype: float, iterable
        """
        base = self._base_light(coral, water_depth)
        biomass = np.pi * (
            .25 * coral.morphology.diameter ** 2 + coral.morphology.diameter * coral.morphology.plate_thickness +
            coral.morphology.base_diameter * base
        )
        coral.set_characteristic('biomass', biomass)
        return biomass

    def _base_light(self, coral, water_depth):
        """Section of coral base receiving light

        :param coral: coral
        :param water_depth: water depth

        :type coral: Coral
        :type water_depth: float

        :return: lighted base section of coral
        :rtype: float, iterable
        """
        # spreading of light
        theta = self._light_spreading(coral, water_depth)
        # coral base section
        base = coral.morphology.height - coral.morphology.plate_thickness - (
            (coral.morphology.diameter - coral.morphology.base_diameter) / (2 * np.tan(.5 * theta))
        )
        # no negative lengths
        return np.max([base, np.zeros_like(base)], axis=0)

    def _side_correction(self, coral, water_depth):
        """Correction of the light-intensity on the sides of the coral object.

        :param coral: coral
        :param water_depth: water depth

        :type coral: Coral
        :type water_depth: float

        :return: correction factor
        :rtype: float, iterable
        """
        # spreading of light
        theta = self._light_spreading(coral, water_depth)
        # correction factor
        return np.sin(.5 * theta)

    def _light_spreading(self, coral, water_depth):
        """Spreading of light based on water depth and coral morphology.

        :param coral: coral
        :param water_depth: water depth

        :type coral: Coral
        :type water_depth: float

        :return: spreading of light
        :rtype: float, iterable
        """
        return self.constants.theta_max * np.exp(-self.environment.light_attenuation * (
                water_depth - coral.morphology.height + coral.morphology.plate_thickness
        ))


class Flow(_BasicBiophysics):

    def _update(self, cell):
        """Update corals: Flow micro-environment.

        :param cell: grid cell
        :type cell: Cell
        """
        [self._execute_flow(coral, cell.water_depth) for coral in cell.corals]

    def _execute_flow(self, coral, water_depth):
        """Execution of Flow-object.
        
        :param coral: coral
        :param water_depth: water depth

        :type coral: Coral
        :type water_depth: float
        """
        self._velocities(coral, water_depth)
        self._thermal_boundary_layer(coral)

    def _velocities(self, coral, water_depth):
        """In-canopy flow velocities, and depth-averaged flow velocities.

        :param coral: coral
        :param water_depth: water depth

        :type coral: Coral
        :type water_depth: float
        """
        if self.processes.photosynthetic_flow_dependency:
            if self.processes.flow_micro_environment:
                wave_attenuation = self._wave_attenuation(
                    coral.morphology.representative_diameter, coral.morphology.height, coral.morphology.distance,
                    self.hydrodynamics.wave_velocity, self.hydrodynamics.wave_period, water_depth, 'wave'
                )
                current_attenuation = self._wave_attenuation(
                    coral.morphology.representative_diameter, coral.morphology.height, coral.morphology.distance,
                    self.hydrodynamics.current_velocity, 1e3, water_depth, 'current'
                )
            else:
                wave_attenuation, current_attenuation = 1, 1

            coral.set_characteristic('in_canopy_flow', self._wave_current(wave_attenuation, current_attenuation))
            coral.set_characteristic('overall_flow', self._wave_current())
        else:
            coral.set_characteristic('in_canopy_flow', 9999)

    def _wave_current(self, wave_attenuation=1, current_attenuation=1):
        """Wave-current interaction.

        :param wave_attenuation: wave-attenuation coefficient, defaults to 1
        :param current_attenuation: current-attenuation coefficient, defaults to 1

        :type wave_attenuation: float, iterable, optional
        :type current_attenuation: float, iterable, optional

        :return: flow velocity due to wave-current interaction
        :rtype: float, iterable
        """
        wave_in_canopy = wave_attenuation * self.hydrodynamics.wave_velocity
        current_in_canopy = current_attenuation * self.hydrodynamics.current_velocity
        return np.sqrt(
            wave_in_canopy ** 2 + current_in_canopy ** 2 +
            2 * wave_in_canopy * current_in_canopy * np.cos(self.constants.angle)
        )

    def _wave_attenuation(self, diameter, height, distance, velocity, period, depth, attenuation):
        """Wave-attenuation coefficient.

        :param diameter: representative coral diameter [m]
        :param height: coral height [m]
        :param distance: axial distance [m]
        :param velocity: flow velocity [m s-1]
        :param period: wave period [s]
        :param depth: water depth [m]
        :param attenuation: type of wave-attenuation coefficient [-]

        :type diameter: float
        :type height: float
        :type distance: float
        :type velocity: float
        :type depth: float
        :type depth: float
        :type attenuation: str
        """
        assert attenuation not in ('wave', 'current')

        above_motion = 1
        shear_length = 1
        drag_length = 1
        lambda_planar = 1

        def function(beta):
            """Complex-valued function to be solved.

            :param beta: complex-valued wave-attenuation coefficient
            :type beta: complex

            :return: function
            :rtype: complex
            """
            # components
            shear = (8 * above_motion) / (3 * np.pi * shear_length) * (abs(1 - beta) * (1 - beta))
            drag = (8 * above_motion) / (3 * np.pi * drag_length) * (abs(beta) * beta)
            inertia = 1j * beta * self.constants.inertia * lambda_planar / (1 - lambda_planar)
            # combined
            return 1j * (beta - 1) - shear + drag + inertia

        def derivative(beta):
            """Complex-valued derivative of above complex-valued function.

            :param beta: complex-valued wave-attenuation coefficient
            :type beta: complex

            :return: derivative
            :rtype: complex
            """
            # components
            shear = ((1 - beta) ** 2 / abs(1 - beta) - abs(1 - beta)) / shear_length
            drag = (beta ** 2 / abs(beta) + beta) / drag_length
            inertia = 1j * self.constants.inertia * lambda_planar / (1 - lambda_planar)
            # combined
            return 1j + (8 * above_motion) / (3 * np.pi) * (-shear + drag) + inertia

        # parameter definitions: geometric parameters
        planar_area = .25 * np.pi * diameter ** 2
        frontal_area = diameter * height
        total_area = .5 * distance ** 2
        lambda_planar = planar_area / total_area
        lambda_frontal = frontal_area / total_area
        shear_length = height / (self.constants.smagorinsky ** 2)

        # calculations
        alpha = 1
        if depth > height:
            # initial iteration values
            above_flow = velocity
            drag_coefficient = 1
            # iteration
            for k in range(int(self.constants.max_iter_canopy)):
                drag_length = 2 * height * (1 - lambda_planar) / (drag_coefficient * lambda_frontal)
                above_motion = above_flow * period / (2 * np.pi)

                if attenuation == 'wave':
                    # noinspection PyTypeChecker
                    alpha = abs(newton(
                        function, x0=complex(.1, .1), fprime=derivative, maxiter=self.constants.max_iter_attenuation)
                    )
                elif attenuation == 'current':
                    x = drag_length / shear_length * (height / (depth - height) + 1)
                    alpha = (x - np.sqrt(x)) / (x - 1)
                else:
                    raise ValueError

                porous_flow = alpha * above_flow
                constricted_flow = (1 - lambda_planar) / (1 - np.sqrt(
                    4 * lambda_planar / (self.constants.spacing_ratio * np.pi)
                )) * porous_flow
                reynolds = constricted_flow * diameter / self.constants.viscosity
                new_drag = 1 + 10 * reynolds ** (-2 / 3)

                if abs((new_drag - drag_coefficient) / new_drag) <= self.constants.error:
                    break
                else:
                    drag_coefficient = float(new_drag)
                    above_flow = abs(
                        (1 - self.constants.numeric_theta) * above_flow +
                        self.constants.numeric_theta * (depth * velocity - height * porous_flow) / (depth - height)
                    )

                if k == self.constants.max_iter_canopy:
                    LOG.warning(f'Maximum number of iterations reached\t:\t{self.constants.max_iter_canopy}')

        return alpha

    def _thermal_boundary_layer(self, coral):
        """Thermal boundary layer.

        :param coral: coral
        :type coral: Coral
        """
        if self.processes.photosynthetic_flow_dependency and self.processes.thermal_micro_environment:
            vbl = self._velocity_boundary_layer(coral)
            tbl = vbl * ((self.constants.absorptivity / self.constants.viscosity) ** (1 / 3))
            coral.set_characteristic('thermal_boundary_layer', tbl)

    def _velocity_boundary_layer(self, coral):
        """Velocity boundary layer.

        :param coral: coral
        :type coral: Coral

        :return: velocity boundary layer
        :rtype: float
        """
        return self.constants.wall_coordinate * self.constants.viscosity / (
            np.sqrt(self.constants.friction) * coral.get_characteristic('in_canopy_flow')
        )


class Temperature(_BasicBiophysics):
    pass


class Photosynthesis(_BasicBiophysics):
    pass


class PopulationStates(_BasicBiophysics):
    pass


class Calcification(_BasicBiophysics):
    pass


class Morphology(_BasicBiophysics):
    pass


class Dislodgement(_BasicBiophysics):
    pass


class Recruitment(_BasicBiophysics):
    pass
