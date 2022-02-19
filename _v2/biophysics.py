"""
Biophysical relations.

Author: Gijs G. Hendrickx
"""
import logging

import numpy as np
from scipy.optimize import newton

from _v2.coral import Coral, _CoralState, _CoralMorphology
from _v2.settings import Constants, Processes

LOG = logging.getLogger(__name__)


class _BasicBiophysics:
    _environment = None
    _hydrodynamics = None
    _constants = Constants()
    _processes = Processes()

    _essential_data = None

    def __init__(self, coral_reef):
        """Initiate biophysical process: Update coral reef.

        :param coral_reef: grid of corals, i.e. coral reef
        :type coral_reef: Grid
        """
        self._verify_essentials()
        self.update(coral_reef)

    def _verify_essentials(self):
        """Verify if all essential information is available for the biophysical process to execute."""
        if self.environment is None:
            msg = f'No environmental conditions defined.'
            raise ValueError(msg)

        if self._essential_data is not None and getattr(self.environment, self._essential_data) is None:
            msg = f'Essential environmental data missing: \"{self._essential_data}\".'
            raise ValueError(msg)

    @classmethod
    def set_environment(cls, environment):
        """
        :param environment: environmental conditions
        :type environment: EnvironmentalConditions
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
    _essential_data = 'light'

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
                np.exp(-self.environment.light_attenuation * water_depth)
            )
        )
        # > total
        total = sum([top, side_top, side_base])

        # biomass-averaged
        biomass = self._biomass(coral, water_depth)
        light = total / biomass

        # set light micro-environment
        coral.vars.light = light

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
        coral.vars.biomass = biomass
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
    _essential_data = 'flow'

    def _update(self, cell):
        """Update corals: Flow micro-environment.

        :param cell: grid cell
        :type cell: Cell
        """
        icf = self._velocities(_CoralMorphology.cell_representative(cell), cell.water_depth)
        [coral.vars.set_variables(in_canopy_flow=icf) for coral in cell.corals]

        [self._thermal_boundary_layer(coral) for coral in cell.corals]
        cell.flow_velocity = self._wave_current()

    def _velocities(self, morphology, water_depth):
        """In-canopy flow velocities, and depth-averaged flow velocities.

        :param morphology: cell-representative morphology
        :param water_depth: water depth

        :type morphology: _CoralMorphology
        :type water_depth: float

        :return: (in-canopy) flow velocity
        :rtype: float
        """
        if self.processes.photosynthetic_flow_dependency:
            if self.processes.flow_micro_environment:
                wave_attenuation = self._wave_attenuation(
                    morphology.representative_diameter, morphology.height, morphology.distance,
                    self.hydrodynamics.wave_velocity, self.hydrodynamics.wave_period, water_depth, 'wave'
                )
                current_attenuation = self._wave_attenuation(
                    morphology.representative_diameter, morphology.height, morphology.distance,
                    self.hydrodynamics.current_velocity, 1e3, water_depth, 'current'
                )
            else:
                wave_attenuation, current_attenuation = 1, 1

            return self._wave_current(wave_attenuation, current_attenuation)
        return 0

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
        assert attenuation in ('wave', 'current')

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
            coral.vars.tbl = tbl

    def _velocity_boundary_layer(self, coral):
        """Velocity boundary layer.

        :param coral: coral
        :type coral: Coral

        :return: velocity boundary layer
        :rtype: float
        """
        return self.constants.wall_coordinate * self.constants.viscosity / (
            np.sqrt(self.constants.friction) * coral.vars.in_canopy_flow
        )


class Temperature(_BasicBiophysics):
    _essential_data = 'temperature'

    def _update(self, cell):
        """Update corals: Thermal micro-environment

        :param cell: grid cell
        :type cell: Cell
        """
        [self._coral_temperature(coral) for coral in cell.corals]

    def _coral_temperature(self, coral):
        """Coral temperature.

        :param coral: coral
        :type coral: Coral
        """
        if self.processes.thermal_micro_environment:
            add_temperature = coral.vars.tbl * self.constants.absorptivity / (
                self.constants.thermal_conductivity * self.constants.thermal_morphology
            ) * coral.vars.light
            coral_temperature = self.environment.temperature + add_temperature
        else:
            coral_temperature = self.environment.temperature

        coral.vars.temperature = coral_temperature


class Photosynthesis(_BasicBiophysics):

    def __init__(self, coral_reef, year=None):
        """If :param year: is set to None, thermal limits must be provided manually.

        :param coral_reef: grid of corals, i.e. coral reef
        :param year: year of simulation, defaults to None

        :type coral_reef: Grid
        :type year: int, optional
        """
        self._year = year
        super().__init__(coral_reef)

    def _update(self, cell):
        """Update corals: Photosynthetic dependencies.

        :param cell: grid cell
        :type cell: Cell
        """
        [self._photosynthetic_rate(coral) for coral in cell.corals]

    def _photosynthetic_rate(self, coral,):
        """Photosynthetic efficiency.

        :param coral: coral
        :type coral: Coral
        """
        # photosynthetic dependencies
        pld = 1 if coral.vars.light is None else self._light_dependency(coral, 'qss')
        pfd = 1 if coral.vars.in_canopy_flow is None else self._flow_dependency(coral)
        ptd = 1 if coral.vars.temperature is None else self._thermal_dependency(coral)

        # combined
        coral.vars.photosynthesis = pld * pfd * ptd

    def _light_dependency(self, coral, output):
        """Photosynthetic light dependency.

        :param coral: coral
        :param output: output definition

        :type coral: Coral
        :type output: str
        """

        def photo_acclimation(coral_light, light, x_old, param):
            """Photo-acclimation.

            :param coral_light: representative light conditions
            :param light: light conditions
            :param x_old: value at previous time step
            :param param: parameter

            :type coral_light: float
            :type light: float
            :type x_old: float
            :type param: str
            """
            assert param in ('Ik', 'Pmax')
            assert output in ('qss', 'new')

            # parameter definitions
            x_max = self.constants.max_saturation if param == 'Ik' else self.constants.max_photosynthesis
            x_pow = self.constants.exp_saturation if param == 'Ik' else self.constants.exp_max_photosynthesis

            # calculations
            xs = x_max * (coral_light / light) ** x_pow
            if output == 'qss':
                return xs
            elif output == 'new':
                return xs + (x_old - xs) * np.exp(-self.constants.photo_acc_rate)

        if output == 'qss':
            saturation = photo_acclimation(
                coral.vars.light, self.environment.light, 0, 'Ik'
            )
            max_photosynthesis = photo_acclimation(
                coral.vars.light, self.environment.light, 0, 'Pmax'
            )
        else:
            raise NotImplementedError

        # calculations
        return max_photosynthesis * (
            np.tanh(coral.vars.light / saturation) - np.tanh(.01 * self.environment.light / saturation)
        )

    def _thermal_dependency(self, coral):
        """Photosynthetic thermal dependency.

        :param coral: coral
        :type coral: Coral
        """

        def thermal_acclimation():
            """Thermal acclimation."""
            if self._year is None:
                msg = f'Thermal acclimation requires a definition of the year: year = {self._year}'
                raise ValueError(msg)

            if self.processes.thermal_micro_environment:
                raise NotImplementedError
            else:
                mmm = self.environment.temperature_mmm[np.logical_and(
                    self.environment.temperature_mmm.index < self._year,
                    self.environment.temperature_mmm.index >= self._year - int(
                        self.constants.thermal_acclimation_period / coral.constants.species_constant
                    )
                )]
                m_min, m_max = mmm.mean(axis=0)
                s_min, s_max = mmm.std(axis=0)

            coral.vars.lower_limit = m_min - self.constants.thermal_variability * s_min
            coral.vars.upper_limit = m_max + self.constants.thermal_variability * s_max

        def adapted_temperature(delta_temp):
            """Adapted temperature response."""

            def specialisation():
                """Specialisation term."""
                return 4e-4 * np.exp(-.33 * (delta_temp - 10))

            relative_temperature = coral.vars.temperature - coral.vars.lower_limit
            response = -relative_temperature * (relative_temperature ** 2 - delta_temp ** 2)
            critical = coral.vars.lower_limit - (1 / np.sqrt(3)) * delta_temp

            if self.processes.thermal_micro_environment:
                pass
            else:
                response[coral.vars.temperature <= critical] = -2 / (3 * np.sqrt(3)) * delta_temp ** 3

            return response * specialisation()

        def thermal_envelope(optimal):
            """Thermal envelope."""
            return np.exp((self.constants.activation_energy / self.constants.gas_constant) * (1 / 300 - 1 / optimal))

        # parameter definitions
        if self.processes.thermal_acclimation:
            thermal_acclimation()
        diff_temp = coral.vars.upper_limit - coral.vars.lower_limit
        opt_temp = coral.vars.lower_limit + (1 / np.sqrt(3)) * diff_temp

        # calculations
        return adapted_temperature(diff_temp) * thermal_envelope(opt_temp)

    def _flow_dependency(self, coral):
        """Photosynthetic flow dependency.

        :param coral: coral
        :type coral: Coral
        """
        if self.processes.photosynthetic_flow_dependency:
            return self.constants.min_photosynthetic_flow_dependency + (
                    1 - self.constants.min_photosynthetic_flow_dependency
            ) * np.tanh(2 * coral.vars.in_canopy_flow / self.constants.invariant_flow_velocity)
        return 1


class PopulationStates(_BasicBiophysics):

    def _update(self, cell):
        """Update corals: Population states

        :param cell: grid cell
        :type cell: Cell
        """
        [self._population_states(coral, cell.capacity) for coral in cell.corals]

    def _population_states(self, coral, capacity):
        """Population dynamics: temporal iteration.

        :param coral: coral
        :param capacity: carrying capacity

        :type coral: Coral
        :type capacity: float
        """
        # set initial coral state
        coral.states.last_reset()
        # append new coral states
        [coral.states.append(
            self._population_dynamics(ps, coral.states[i], coral.constants.species_constant, capacity)
        ) for i, ps in enumerate(coral.vars.photosynthesis)]
        # remove initial coral state
        coral.states.pop_state(index=0)

    def _population_dynamics(self, photosynthesis, p0, species_constant, capacity, dt=1):
        """Population dynamics: spatial iteration.

        :param photosynthesis: photosynthetic rate
        :param p0: coral states at time t - 1
        :param capacity: carrying capacity

        :type photosynthesis: float
        :type p0: _CoralState
        :type capacity: float
        """
        # growing conditions
        if photosynthesis > 0:
            # bleached population
            bleached = p0.bleached / (1 + dt * (
                8 * self.constants.recovery_rate * photosynthesis / species_constant +
                self.constants.mortality_rate * species_constant
            ))
            # pale population
            pale = (p0.pale + bleached * (
                8 * dt * self.constants.recovery_rate * photosynthesis / species_constant
            )) / (1 + dt * self.constants.recovery_rate * photosynthesis * species_constant)
            # recovered population
            recovered = (
                p0.recovered + dt * self.constants.recovery_rate * photosynthesis * species_constant * pale
            ) / (1 + .5 * dt * self.constants.recovery_rate * photosynthesis * species_constant)
            # healthy population
            a = dt * self.constants.growth_rate * photosynthesis * species_constant / capacity
            b = 1 - dt * self.constants.growth_rate * photosynthesis * species_constant * (
                1 - sum([recovered, pale, bleached]) / capacity
            )
            c = -(p0.healthy + .5 * dt * self.constants.recovery_rate * photosynthesis * species_constant * recovered)
            healthy = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

        # bleaching conditions
        else:
            # healthy population
            healthy = p0.healthy / (1 - dt * self.constants.bleaching_rate * photosynthesis * species_constant)
            # recovered population
            recovered = p0.recovered / (1 - dt * self.constants.bleaching_rate * photosynthesis * species_constant)
            # pale population
            pale = (p0.pale - dt * self.constants.bleaching_rate * photosynthesis * species_constant * (
                healthy + recovered
            )) / (1 - .5 * dt * self.constants.bleaching_rate * photosynthesis * species_constant)
            # bleached population
            bleached = (
                p0.bleached - .5 * dt * self.constants.bleaching_rate * photosynthesis * species_constant * pale
            ) / (1 - .25 * dt * self.constants.bleaching_rate * photosynthesis * species_constant)

        return _CoralState(healthy=healthy, recovered=recovered, pale=pale, bleached=bleached)


class Calcification(_BasicBiophysics):

    def _update(self, cell):
        """Update corals: Calcification

        :param cell: grid cell
        :type cell: Cell
        """
        [self._calcification_rate(coral) for coral in cell.corals]

    def _calcification_rate(self, coral):
        """Calcification rate.

        :param coral: coral
        :type coral: Coral
        """
        if self.environment.aragonite is None:
            aragonite_dependency = 1
        else:
            aragonite_dependency = (self.environment.aragonite - self.constants.dissolution_saturation) / (
                self.constants.half_rate + self.environment.aragonite - self.constants.dissolution_saturation
            )

        calcification = self.constants.calcification_constant * coral.constants.species_constant * \
            np.array([s.healthy for s in coral.states]) * aragonite_dependency * coral.vars.photosynthesis

        coral.vars.calcification = calcification


class Morphology(_BasicBiophysics):

    def _update(self, cell):
        """Update corals: Morphology.

        :param cell: grid cell
        :type cell: Cell
        """
        [self._new_morphology(coral) for coral in cell.corals]

    def _new_morphology(self, coral):
        """

        :param coral: coral
        :type coral: Coral
        """
        # updated ratios
        ratios = [self._ratio_update(coral, ratio) for ratio in ('form', 'plate', 'spacing')]

        # updated volume
        volume = coral.morphology.volume + self._increased_volume(coral)

        # updated morphology
        coral.morphology.update(volume, *ratios)

    def _ratio_update(self, coral, ratio):
        """Update morphological ratios.

        :param coral: coral
        :param ratio: ratio type

        :type coral: Coral
        :type ratio: str
        """
        assert ratio in ('form', 'plate', 'spacing')

        def r_new(r_old, r_opt):
            """New morphological ratio based on mass balance.

            :param r_old: previous morphological ratio
            :param r_opt: optimal morphological ratio

            :type r_old: float
            :type r_opt: float

            :return: updated morphological ratio
            :rtype: float
            """
            return (coral.morphology.volume * r_old + self._increased_volume(coral) * r_opt) / \
                (coral.morphology.volume + self._increased_volume(coral))

        # optimal ratios
        opt_ratio = getattr(self, f'_optimal_{ratio}_ratio')(coral)

        # update morphological ratio
        return r_new(getattr(coral.morphology, f'{ratio}_ratio'), opt_ratio)

    def _increased_volume(self, coral, dt=1):
        """Increase in volume after :param dt: years.

        :param coral: coral
        :param dt: time step in years, defaults to 1

        :type coral: Coral
        :type dt: float

        :return: increased coral volume
        :rtype: float
        """
        return .5 * coral.morphology.distance ** 2 * sum(coral.vars.calcification) * dt / \
            self.constants.coral_density * np.mean(coral.vars.biomass)

    def _optimal_form_ratio(self, coral):
        """Optimal form ratio; height : (plate) diameter.

        :param coral: coral
        :type coral: Coral

        :return: optimal form ratio
        :rtype: float
        """
        in_canopy_flow = coral.vars.in_canopy_flow
        return self.constants.proportionality_form * np.mean(coral.vars.light) / np.mean(self.environment.light) * \
            (self.constants.fitting_flow_velocity / (in_canopy_flow if in_canopy_flow > 0 else 1e-6))

    def _optimal_plate_ratio(self, coral):
        """Optimal plate ratio; base diameter : (plate) diameter.

        :param coral: coral
        :type coral: Coral

        :return: optimal plate ratio
        :rtype: float
        """
        return self.constants.proportionality_plate * (1 + np.tanh(
            self.constants.proportionality_plate_flow * (
                coral.vars.in_canopy_flow - self.constants.fitting_flow_velocity
            ) / self.constants.fitting_flow_velocity
        ))

    def _optimal_spacing_ratio(self, coral):
        """Optimal spacing ratio; plate diameter : axial distance.

        :param coral: coral
        :type coral: Coral

        :return: optimal spacing ratio
        :rtype: float
        """
        return self.constants.proportionality_space * (
            1 - np.tanh(
                self.constants.proportionality_space_light * np.mean(coral.vars.light / np.mean(self.environment.light))
            )
        ) * (1 + np.tanh(
            self.constants.proportionality_space_flow * (
                coral.vars.in_canopy_flow - self.constants.fitting_flow_velocity
            ) / self.constants.fitting_flow_velocity
        ))


class Dislodgement(_BasicBiophysics):

    def __new__(cls, *args, **kwargs):
        raise NotImplementedError

    def _update(self, cell):
        """Update corals: Dislodgement.

        :param cell: grid cell
        :type cell: Cell
        """
        [self._storm_impact(coral, cell.flow_velocity) for coral in cell.corals]

    def _storm_impact(self, coral, flow):
        """Update coral population states and morphology due to storm event.

        :param coral: coral
        :param flow: flow velocity

        :type coral: Coral
        :type flow: float
        """
        # survival rate
        survival = self._partial_dislodgement(coral, flow)
        # update population states
        [
            setattr(coral.states[-1], state, survival * getattr(coral.states[-1], state))
            for state in ('healthy', 'recovered', 'pale', 'bleached')
        ]
        # update morphology
        coral.morphology.update(survival * coral.morphology.volume)

    def _partial_dislodgement(self, coral, flow):
        """Percentage surviving storm event.

        :param coral: coral
        :param flow: flow velocity

        :type coral: Coral
        :type flow: float

        :return: surviving fraction
        :rtype: float
        """
        if self._dislodgement_criterion(coral, flow):
            return self._dislodgement_mechanical_threshold(coral, flow) / self._canopy_shape_factor(coral)
        return 1

    def _dislodgement_criterion(self, coral, flow):
        """Potential dislodgement of corals.

        :param coral: coral
        :param flow: flow velocity

        :type coral: Coral
        :type flow: float

        :return: coral dislodges
        :rtype: bool
        """
        return self._dislodgement_mechanical_threshold(coral, flow) <= self._canopy_shape_factor(coral)

    def _dislodgement_mechanical_threshold(self, coral, flow):
        """Dislodgement Mechanical Threshold.

        :param coral: coral
        :param flow: flow velocity

        :type coral: Coral
        :type flow: float

        :return: dislodgement mechanical threshold
        :rtype: float
        """
        return self.constants.tensile_stress / (
                self.constants.water_density * self.constants.drag_coefficient * flow
        )

    @staticmethod
    def _canopy_shape_factor(coral):
        """Canopy Shape Factor.

        :param coral: coral
        :type coral: Coral

        :return: canopy shape factor
        :rtype: float
        """
        # arms of moment
        arm_top = coral.morphology.height - .5 * coral.morphology.plate_thickness
        arm_bottom = .5 * (coral.morphology.height - coral.morphology.plate_thickness)
        # area of moment
        area_top = coral.morphology.distance * coral.morphology.plate_thickness
        area_bottom = coral.morphology.base_diameter * (coral.morphology.height - coral.morphology.plate_thickness)
        # integral
        integral = arm_top * area_top + arm_bottom * area_bottom
        # colony shape factor
        return 16 / (np.pi * coral.morphology.base_diameter ** 3) * integral


class Recruitment(_BasicBiophysics):

    def __new__(cls, *args, **kwargs):
        raise NotImplementedError

    def _update(self, cell):
        """Update corals: Coral recruitment.

        :param cell: grid cell
        :type cell: Cell
        """
        [self._coral_recruitment(coral, cell.living_cover, cell.capacity) for coral in cell.corals]

    def _coral_recruitment(self, coral, living_cover, capacity):
        """Update coral population states and morphology due to spawning event.

        :param coral: coral
        :param living_cover: living coral cover, all coral species in grid cell included
        :param capacity: carrying capacity

        :type coral: Coral
        :type living_cover: float
        :type capacity: float
        """
        [self._spawning(coral, living_cover, capacity, param) for param in ('P', 'V')]

    def _spawning(self, coral, living_cover, capacity, param):
        """Contribution to coral growth due to mass spawning

        :param coral: coral
        :param living_cover: living coral cover, all coral species in grid cell included
        :param capacity: carrying capacity
        :param param: determination of spawning contribution

        :type coral: Coral
        :type living_cover: float
        :type capacity: float
        :type param: str
        """
        assert param in ('P', 'V')

        # potential
        power = 2 if param == 'P' else 3
        potential = self.constants.settle_probability * self.constants.larvae_spawned * \
            self.constants.larval_diameter ** power
        # healthy population
        averaged_healthy_population = np.mean([s.healthy for s in coral.states])
        # recruitment
        recruited = potential * averaged_healthy_population * (1 - living_cover / capacity)
        # update coral
        if param == 'P':
            # update population states
            coral.states[-1].healthy += recruited
        elif param == 'V':
            # update morphology
            coral.morphology.update(coral.morphology.volume + recruited)
