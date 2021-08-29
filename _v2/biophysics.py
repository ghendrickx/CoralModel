import logging

import numpy as np
from scipy.optimize import newton

from _v2.coral import Coral, _CoralStates
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
    def e(self):
        """
        :return: environmental conditions
        :rtype: _EnvironmentSnippet
        """
        return self._environment

    @property
    def h(self):
        """
        :return: hydrodynamic conditions
        :rtype: Hydrodynamics
        """
        return self._hydrodynamics

    @property
    def c(self):
        """
        :return: simulation constants
        :rtype: Constants
        """
        return self._constants

    @property
    def p(self):
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
        top = .25 * np.pi * coral.morphology.diameter ** 2 * self.e.light * np.exp(
            -self.e.light_attenuation * (water_depth - coral.morphology.height)
        )
        # > side of plate
        side_top = self._side_correction(coral, water_depth) * (
                np.pi * coral.morphology.diameter * self.e.light / self.e.light_attenuation * (
                np.exp(-self.e.light_attenuation * (water_depth - coral.morphology.height)) -
                np.exp(-self.e.light_attenuation * (
                        water_depth - coral.morphology.height + coral.morphology.plate_thickness
                ))
            )
        )
        # > side of base
        side_base = self._side_correction(coral, water_depth) * (
                np.pi * coral.morphology.base_diameter * self.e.light / self.e.light_attenuation * (
                np.exp(-self.e.light_attenuation * (water_depth - base)) -
                np.exp(-self.e * water_depth)
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
        return self.c.theta_max * np.exp(-self.e.light_attenuation * (
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
        if self.p.photosynthetic_flow_dependency:
            if self.p.flow_micro_environment:
                wave_attenuation = self._wave_attenuation(
                    coral.morphology.representative_diameter, coral.morphology.height, coral.morphology.distance,
                    self.h.wave_velocity, self.h.wave_period, water_depth, 'wave'
                )
                current_attenuation = self._wave_attenuation(
                    coral.morphology.representative_diameter, coral.morphology.height, coral.morphology.distance,
                    self.h.current_velocity, 1e3, water_depth, 'current'
                )
            else:
                wave_attenuation, current_attenuation = 1, 1

            coral.set_characteristic('in_canopy_flow', self._wave_current(wave_attenuation, current_attenuation))
        else:
            coral.set_characteristic('in_canopy_flow', 9999)

        coral.set_characteristic('overall_flow', self._wave_current())

    def _wave_current(self, wave_attenuation=1, current_attenuation=1):
        """Wave-current interaction.

        :param wave_attenuation: wave-attenuation coefficient, defaults to 1
        :param current_attenuation: current-attenuation coefficient, defaults to 1

        :type wave_attenuation: float, iterable, optional
        :type current_attenuation: float, iterable, optional

        :return: flow velocity due to wave-current interaction
        :rtype: float, iterable
        """
        wave_in_canopy = wave_attenuation * self.h.wave_velocity
        current_in_canopy = current_attenuation * self.h.current_velocity
        return np.sqrt(
            wave_in_canopy ** 2 + current_in_canopy ** 2 +
            2 * wave_in_canopy * current_in_canopy * np.cos(self.c.angle)
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
            inertia = 1j * beta * self.c.inertia * lambda_planar / (1 - lambda_planar)
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
            inertia = 1j * self.c.inertia * lambda_planar / (1 - lambda_planar)
            # combined
            return 1j + (8 * above_motion) / (3 * np.pi) * (-shear + drag) + inertia

        # parameter definitions: geometric parameters
        planar_area = .25 * np.pi * diameter ** 2
        frontal_area = diameter * height
        total_area = .5 * distance ** 2
        lambda_planar = planar_area / total_area
        lambda_frontal = frontal_area / total_area
        shear_length = height / (self.c.smagorinsky ** 2)

        # calculations
        alpha = 1
        if depth > height:
            # initial iteration values
            above_flow = velocity
            drag_coefficient = 1
            # iteration
            for k in range(int(self.c.max_iter_canopy)):
                drag_length = 2 * height * (1 - lambda_planar) / (drag_coefficient * lambda_frontal)
                above_motion = above_flow * period / (2 * np.pi)

                if attenuation == 'wave':
                    # noinspection PyTypeChecker
                    alpha = abs(newton(
                        function, x0=complex(.1, .1), fprime=derivative, maxiter=self.c.max_iter_attenuation)
                    )
                elif attenuation == 'current':
                    x = drag_length / shear_length * (height / (depth - height) + 1)
                    alpha = (x - np.sqrt(x)) / (x - 1)
                else:
                    raise ValueError

                porous_flow = alpha * above_flow
                constricted_flow = (1 - lambda_planar) / (1 - np.sqrt(
                    4 * lambda_planar / (self.c.spacing_ratio * np.pi)
                )) * porous_flow
                reynolds = constricted_flow * diameter / self.c.viscosity
                new_drag = 1 + 10 * reynolds ** (-2 / 3)

                if abs((new_drag - drag_coefficient) / new_drag) <= self.c.error:
                    break
                else:
                    drag_coefficient = float(new_drag)
                    above_flow = abs(
                        (1 - self.c.numeric_theta) * above_flow +
                        self.c.numeric_theta * (depth * velocity - height * porous_flow) / (depth - height)
                    )

                if k == self.c.max_iter_canopy:
                    LOG.warning(f'Maximum number of iterations reached\t:\t{self.c.max_iter_canopy}')

        return alpha

    def _thermal_boundary_layer(self, coral):
        """Thermal boundary layer.

        :param coral: coral
        :type coral: Coral
        """
        if self.p.photosynthetic_flow_dependency and self.p.thermal_micro_environment:
            vbl = self._velocity_boundary_layer(coral)
            tbl = vbl * ((self.c.absorptivity / self.c.viscosity) ** (1 / 3))
            coral.set_characteristic('thermal_boundary_layer', tbl)

    def _velocity_boundary_layer(self, coral):
        """Velocity boundary layer.

        :param coral: coral
        :type coral: Coral

        :return: velocity boundary layer
        :rtype: float
        """
        return self.c.wall_coordinate * self.c.viscosity / (
                np.sqrt(self.c.friction) * coral.get_characteristic('in_canopy_flow')
        )


class Temperature(_BasicBiophysics):

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
        if self.p.thermal_micro_environment:
            add_temperature = coral.get_characteristic('thermal_boundary_layer') * self.c.absorptivity / (
                    self.c.thermal_conductivity * self.c.thermal_morphology
            ) * coral.get_characteristic('light')
            coral_temperature = self.e.temperature + add_temperature
        else:
            coral_temperature = self.e.temperature

        coral.set_characteristic('temperature', coral_temperature)


class Photosynthesis(_BasicBiophysics):

    def __init__(self, coral_reef, year):
        """
        :param coral_reef: grid of corals, i.e. coral reef
        :param year: year of simulation

        :type coral_reef: Grid
        :type year: int
        """
        self._year = year
        super().__init__(coral_reef)

    def _update(self, cell):
        """Update corals: Photosynthetic dependencies.

        :param cell: grid cell
        :type cell: Cell
        """
        [self._photosynthetic_rate(coral, self._year) for coral in cell.corals]

    def _photosynthetic_rate(self, coral, year):
        """Photosynthetic efficiency.

        """
        # photosynthetic dependencies
        pld = self._light_dependency(coral, 'qss')
        pfd = self._thermal_dependency(coral, year)
        ptd = self._flow_dependency(coral)

        # combined
        coral.set_characteristic('photosynthesis', pld * pfd * ptd)

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
            x_max = self.c.max_saturation if param == 'Ik' else self.c.max_photosynthesis
            x_pow = self.c.exp_saturation if param == 'Ik' else self.c.exp_max_photosynthesis

            # calculations
            xs = x_max * (coral_light / light) ** x_pow
            if output == 'qss':
                return xs
            elif output == 'new':
                return xs + (x_old - xs) * np.exp(-self.c.photo_acc_rate)

        if output == 'qss':
            saturation = photo_acclimation(
                coral.get_characteristic('light'), self.e.light, 0, 'Ik'
            )
            max_photosynthesis = photo_acclimation(
                coral.get_characteristic('light'), self.e.light, 0, 'Pmax'
            )
        else:
            raise NotImplementedError

        # calculations
        return max_photosynthesis * (
            np.tanh(coral.get_characteristic('light') / saturation) - np.tanh(.01 * self.e.light / saturation)
        )

    def _thermal_dependency(self, coral, year):
        """Photosynthetic thermal dependency.

        :param coral: coral
        :param year: year of simulation

        :type coral: Coral
        :type year: int
        """

        delta_temp = 1

        def thermal_acclimation():
            """Thermal acclimation."""
            if self.p.thermal_micro_environment:
                raise NotImplementedError
            else:
                mmm = self.e.temperature_mmm[np.logical_and(
                    self.e.temperature_mmm.index < year,
                    self.e.temperature_mmm.index >= year - int(
                        self.c.thermal_acclimation_period / coral.constants.species_constant
                    )
                )]
                m_min, m_max = mmm.mean(axis=0)
                s_min, s_max = mmm.std(axis=0)

            coral.set_characteristic('lower_limit', m_min - self.c.thermal_variability * s_min)
            coral.set_characteristic('upper_limit', m_max + self.c.thermal_variability * s_max)

        def adapted_temperature():
            """Adapted temperature response."""

            def specialisation():
                """Specialisation term."""
                return 4e-4 * np.exp(-.33 * delta_temp - 10)

            relative_temperature = coral.get_characteristic('temperature') - coral.get_characteristic('lower_limit')
            response = -relative_temperature * (relative_temperature ** 2 - delta_temp ** 2)
            critical = coral.get_characteristic('lower_limit') - (1 / np.sqrt(3)) * delta_temp

            if self.p.thermal_micro_environment:
                pass
            else:
                response[coral.get_characteristic('temperature') <= critical] = -2 / (3 * np.sqrt(3)) * delta_temp ** 3

            return response * specialisation()

        def thermal_envelope():
            """Thermal envelope."""
            return np.exp((self.c.activation_energy / self.c.gas_constant) * (1 / 300 - 1 / optimal))

        # parameter definitions
        thermal_acclimation()
        delta_temp = coral.get_characteristic('upper_limit') - coral.get_characteristic('lower_limit')
        optimal = coral.get_characteristic('lower_limit') + (1 / np.sqrt(3)) * delta_temp

        # calculations
        return adapted_temperature() * thermal_envelope()

    def _flow_dependency(self, coral):
        """Photosynthetic flow dependency.

        :param coral: coral
        :type coral: Coral
        """
        if self.p.photosynthetic_flow_dependency:
            return self.c.min_photosynthetic_flow_dependency + (
                    1 - self.c.min_photosynthetic_flow_dependency
            ) * np.tanh(2 * coral.get_characteristic('in_canopy_flow') / self.c.invariant_flow_velocity)
        return 1


class PopulationStates(_BasicBiophysics):

    def _update(self, cell):
        """Update corals: Population states

        :param cell: grid cell
        :type cell: Cell
        """

    def _population_states(self, coral, capacity):
        """Population dynamics: temporal iteration.

        :param coral: coral
        :param capacity: carrying capacity

        :type coral: Coral
        :type capacity: float
        """
        # set initial coral states
        coral.states = [coral.states[-1]]
        # append new coral states
        [coral.states.append(
            self._population_dynamics(ps, coral.states[i], coral.constants.species_constant, capacity)
        ) for i, ps in coral.get_characteristic('photosynthesis')]
        # remove initial coral states
        del coral.states[0]

    def _population_dynamics(self, photosynthesis, p0, species_constant, capacity, dt=1):
        """Population dynamics: spatial iteration.

        :param photosynthesis: photosynthetic rate
        :param p0: coral states at time t - 1
        :param capacity: carrying capacity

        :type photosynthesis: float
        :type p0: _CoralStates
        :type capacity: float
        """
        p = _CoralStates()

        # growing conditions
        if photosynthesis:
            # bleached population
            p.bleached = p0.bleached / (1 + dt * (
                    8 * self.c.recovery_rate * photosynthesis / species_constant +
                    self.c.mortality_rate * species_constant
            ))
            # pale population
            p.pale = (p0.pale + p.bleached * (
                    8 * dt * self.c.recovery_rate * photosynthesis / species_constant
            )) / (1 + dt * self.c.recovery_rate * photosynthesis * species_constant)
            # recovered population
            p.recovered = (
                                  p0.recovered + dt * self.c.recovery_rate * photosynthesis * species_constant * p.pale
            ) / (1 + .5 * dt * self.c.recovery_rate * photosynthesis * species_constant)
            # healthy population
            a = dt * self.c.growth_rate * photosynthesis * species_constant / capacity
            b = 1 - dt * self.c.growth_rate * photosynthesis * species_constant * (
                1 - sum([p.recovered, p.pale, p.bleached]) / capacity
            )
            c = -(p0.healthy + .5 * dt * self.c.recovery_rate * photosynthesis * species_constant * p.recovered)
            p.healthy = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

        # bleaching conditions
        else:
            # healthy population
            p.healthy = p0.healthy / (1 - dt * self.c.bleaching_rate * photosynthesis * species_constant)
            # recovered population
            p.recovered = p0.recovered / (1 - dt * self.c.bleaching_rate * photosynthesis * species_constant)
            # pale population
            p.pale = (p0.pale - dt * self.c.bleaching_rate * photosynthesis * species_constant * (
                p.healthy + p.recovered
            )) / (1 - .5 * dt * self.c.bleaching_rate * photosynthesis * species_constant)
            # bleached population
            p.bleached = (
                    p0.bleached - .5 * dt * self.c.bleaching_rate * photosynthesis * species_constant * p.pale
            ) / (1 - .25 * dt * self.c.bleaching_rate * photosynthesis * species_constant)

        return p


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
        aragonite_dependency = (self.e.aragonite - self.c.dissolution_saturation) / (
                self.c.half_rate + self.e.aragonite - self.c.dissolution_saturation
        )
        calcification = self.c.calcification_constant * coral.constants.species_constant * \
            [cs.healthy for cs in coral.states] * aragonite_dependency * coral.get_characteristic('photosynthesis')

        coral.set_characteristic('calcification', calcification)


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
        return .5 * coral.morphology.distance ** 2 * sum(coral.get_characteristic('calcification')) * dt / \
            self.c.coral_density * np.mean(coral.get_characteristic('biomass'))

    def _optimal_form_ratio(self, coral):
        """Optimal form ratio; height : (plate) diameter.

        :param coral: coral
        :type coral: Coral

        :return: optimal form ratio
        :rtype: float
        """
        in_canopy_flow = coral.get_characteristic('in_canopy_flow')
        return self.c.proportionality_form * np.mean(coral.get_characteristic('light')) / np.mean(self.e.light) * \
            (self.c.fitting_flow_velocity / (in_canopy_flow if in_canopy_flow > 0 else 1e-6))

    def _optimal_plate_ratio(self, coral):
        """Optimal plate ratio; base diameter : (plate) diameter.

        :param coral: coral
        :type coral: Coral

        :return: optimal plate ratio
        :rtype: float
        """
        return self.c.proportionality_plate * (1 + np.tanh(
            self.c.proportionality_plate_flow * (
                    coral.get_characteristic('in_canopy_flow') - self.c.fitting_flow_velocity
            ) / self.c.fitting_flow_velocity
        ))

    def _optimal_spacing_ratio(self, coral):
        """Optimal spacing ratio; plate diameter : axial distance.

        :param coral: coral
        :type coral: Coral

        :return: optimal spacing ratio
        :rtype: float
        """
        return self.c.proportionality_space * (
            1 - np.tanh(
                self.c.proportionality_space_light * np.mean(coral.get_characteristic('light') / np.mean(self.e.light))
            )
        ) * (1 + np.tanh(
            self.c.proportionality_space_flow * (
                    coral.get_characteristic('in_canopy_flow') - self.c.fitting_flow_velocity
            ) / self.c.fitting_flow_velocity
        ))


class Dislodgement(_BasicBiophysics):

    def _update(self, cell):
        """Update corals: Dislodgement.

        :param cell: grid cell
        :type cell: Cell
        """
        [self._storm_impact(coral) for coral in cell.corals]

    def _storm_impact(self, coral):
        """Update coral population states and morphology due to storm event.

        :param coral: coral
        :type coral: Coral
        """
        # survival rate
        survival = self._partial_dislodgement(coral)
        # update population states
        [
            setattr(coral.states[-1], state, survival * getattr(coral.states[-1], state))
            for state in ('healthy', 'recovered', 'pale', 'bleached')
        ]
        # update morphology
        coral.morphology.update(survival * coral.morphology.volume)

    def _partial_dislodgement(self, coral):
        """Percentage surviving storm event.

        :param coral: coral
        :type coral: Coral

        :return: surviving fraction
        :rtype: float
        """
        if self._dislodgement_criterion(coral):
            return self._dislodgement_mechanical_threshold(coral) / self._canopy_shape_factor(coral)
        return 1

    def _dislodgement_criterion(self, coral):
        """Potential dislodgement of corals.

        :param coral: coral
        :type coral: Coral

        :return: coral dislodges
        :rtype: bool
        """
        return self._dislodgement_mechanical_threshold(coral) <= self._canopy_shape_factor(coral)

    def _dislodgement_mechanical_threshold(self, coral):
        """Dislodgement Mechanical Threshold.

        :param coral: coral
        :type coral: Coral

        :return: dislodgement mechanical threshold
        :rtype: float
        """
        return self.c.tensile_stress / (
                self.c.water_density * self.c.drag_coefficient * coral.get_characteristic('overall_flow')
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

    def _update(self, cell):
        """Update corals: Coral recruitment.

        :param cell: grid cell
        :type cell: Cell
        """
        [self._coral_recruitment(coral, cell.capacity) for coral in cell.corals]

    def _coral_recruitment(self, coral, capacity):
        """Update coral population states and morphology due to spawning event.

        :param coral: coral
        :param capacity: carrying capacity

        :type coral: Coral
        :type capacity: float
        """
        [self._spawning(coral, capacity, param) for param in ('P', 'V')]

    def _spawning(self, coral, capacity, param):
        """Contribution to coral growth due to mass spawning

        :param coral: coral
        :param capacity: carrying capacity
        :param param: determination of spawning contribution

        :type coral: Coral
        :type capacity: float
        :type param: str
        """
        assert param in ('P', 'V')

        # potential
        power = 2 if param == 'P' else 3
        potential = self.c.settle_probability * self.c.larvae_spawned * self.c.larval_diameter ** power
        # healthy population
        averaged_healthy_population = np.mean([s.healthy for s in coral.states])
        # living cover
        living_cover = coral.states[-1].sum
        # recruitment
        recruited = potential * averaged_healthy_population * (1 - living_cover / capacity)
        # update coral
        if param == 'P':
            # update population states
            coral.states[-1].healthy += recruited
        elif param == 'V':
            # update morphology
            coral.morphology.update(coral.morphology.volume + recruited)
