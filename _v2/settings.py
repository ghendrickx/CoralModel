"""
Constants and included processes.

Author: Gijs G. Hendrickx
"""
import numpy as np


class Processes:

    @classmethod
    def set_process(cls, **kwargs):
        """Define single process to be included in the modelling of corals/coral reefs."""
        [setattr(cls, f'_{k}', v) for k, v in kwargs.items() if hasattr(cls, k)]

    @classmethod
    def set_processes(
            cls, flow_micro_environment=False, thermal_micro_environment=False, photosynthetic_flow_dependency=False,
            photo_acclimation=False, thermal_acclimation=False
    ):
        """Define processes to include in the modelling of corals.

        :param flow_micro_environment: include flow micro environment, defaults to False
        :param thermal_micro_environment: include thermal micro environment, defaults to False
        :param photosynthetic_flow_dependency: include photosynthetic flow dependency, defaults to False
        :param photo_acclimation: include photo-acclimation as differential equation, defaults to False
        :param thermal_acclimation: include thermal-acclimation dynamically, defaults to False

        :type flow_micro_environment: bool, optional
        :type thermal_micro_environment: bool, optional
        :type photosynthetic_flow_dependency: bool, optional
        :type photo_acclimation: bool, optional
        :type thermal_acclimation: bool, optional
        """
        cls._flow_micro_environment = flow_micro_environment
        cls._thermal_micro_environment = thermal_micro_environment
        cls._photosynthetic_flow_dependency = photosynthetic_flow_dependency
        cls._photo_acclimation = photo_acclimation
        cls._thermal_acclimation = thermal_acclimation

    @property
    def flow_micro_environment(self):
        """
        :return: include flow micro-environment
        :rtype: bool
        """
        return self._flow_micro_environment

    @property
    def thermal_micro_environment(self):
        """
        :return: include thermal micro-environment
        :rtype: bool
        """
        return self._thermal_micro_environment

    @property
    def photosynthetic_flow_dependency(self):
        """
        :return: include photosynthetic flow dependency
        :rtype: bool
        """
        return self._photosynthetic_flow_dependency

    @property
    def photo_acclimation(self):
        """
        :return: include photo-acclimation
        :rtype: bool
        """
        return self._photo_acclimation

    @property
    def thermal_acclimation(self):
        """
        :return: include thermal acclimation
        :rtype: bool
        """
        return self._thermal_acclimation


class Constants:

    @classmethod
    def set_constant(cls, constant, value):
        if hasattr(cls, f'_{constant}'):
            setattr(cls, f'_{constant}', value)
        else:
            msg = f'There is no constant named \"{constant}\".'
            raise ValueError(msg)

    # light micro-environment
    _lac_default = None
    _theta_max = None

    @classmethod
    def set_light_micro_environment(cls, lac_default=None, theta_max=None):
        cls._lac_default = lac_default
        cls._theta_max = theta_max

    @property
    def lac_default(self):
        """Default light attenuation coefficient [m-1]."""
        return .1 if self._lac_default is None else self._lac_default

    @property
    def theta_max(self):
        return .5 * np.pi if self._theta_max is None else self._theta_max

    # flow micro-environment
    _smagorisnky = None
    _inertia = None
    _friction = None
    _viscosity = None
    _thermal_diffusivity = None
    _spacing_ratio = None
    _angle = None
    _wall_coordinate = None
    _numeric_theta = None
    _error = None
    _max_iter_canopy = None
    _max_iter_attenuation = None

    @classmethod
    def set_flow_micro_environment(cls, smagorinsky=None, inertia=None, friction=None, viscosity=None,
                                   thermal_diffusivity=None, spacing_ratio=None, angle=None, wall_coordinate=None,
                                   numeric_theta=None, error=None, max_iter_canopy=None, max_iter_attenuation=None):
        cls._smagorisnky = smagorinsky
        cls._inertia = inertia
        cls._friction = friction
        cls._viscosity = viscosity
        cls._thermal_diffusivity = thermal_diffusivity
        cls._spacing_ratio = spacing_ratio
        cls._angle = angle
        cls._wall_coordinate = wall_coordinate
        cls._numeric_theta = numeric_theta
        cls._error = error
        cls._max_iter_canopy = max_iter_canopy
        cls._max_iter_attenuation = max_iter_attenuation

    @property
    def smagorinsky(self):
        return .17 if self._smagorisnky is None else self._smagorisnky

    @property
    def inertia(self):
        return 1.7 if self._inertia is None else self._inertia

    @property
    def friction(self):
        return .01 if self._friction is None else self._friction

    @property
    def viscosity(self):
        return 1e-6 if self._viscosity is None else self._viscosity

    @property
    def thermal_diffusivity(self):
        return 1e-7 if self._thermal_diffusivity is None else self._thermal_diffusivity

    @property
    def spacing_ratio(self):
        return 2 if self._spacing_ratio is None else self._spacing_ratio

    @property
    def angle(self):
        return 0 if self._angle is None else self._angle

    @property
    def wall_coordinate(self):
        return 500 if self._wall_coordinate is None else self._wall_coordinate

    @property
    def numeric_theta(self):
        return .5 if self._numeric_theta is None else self._numeric_theta

    @property
    def error(self):
        return 1e-3 if self._error is None else self._error

    @property
    def max_iter_canopy(self):
        return 1e5 if self._max_iter_canopy is None else self._max_iter_canopy

    @property
    def max_iter_attenuation(self):
        return 1e5 if self._max_iter_attenuation is None else self._max_iter_attenuation

    # thermal micro-environment
    _thermal_morphology = None
    _absorptivity = None
    _thermal_conductivity = None

    @classmethod
    def set_thermal_micro_environment(cls, thermal_morphology=None, absorptivity=None, thermal_conductivity=None):
        cls._thermal_morphology = thermal_morphology
        cls._absorptivity = absorptivity
        cls._thermal_conductivity = thermal_conductivity

    @property
    def thermal_morphology(self):
        return 80 if self._thermal_morphology is None else self._thermal_morphology

    @property
    def absorptivity(self):
        return .4 if self._absorptivity is None else self._absorptivity

    @property
    def thermal_conductivity(self):
        return .6089 if self._thermal_conductivity is None else self._thermal_conductivity

    # photosynthetic light dependency
    _photo_acc_rate = None
    _max_saturation = None
    _max_photosynthesis = None
    _exp_saturation = None
    _exp_max_photosynthesis = None

    @classmethod
    def set_photosynthetic_light_dependency(cls, photo_acc_rate=None, max_satuation=None, max_photosynthesis=None,
                                            exp_saturation=None, exp_max_photosynthesis=None):
        cls._photo_acc_rate = photo_acc_rate
        cls._max_saturation = max_satuation
        cls._max_photosynthesis = max_photosynthesis
        cls._exp_saturation = exp_saturation
        cls._exp_max_photosynthesis = exp_max_photosynthesis

    @property
    def photo_acc_rate(self):
        return .6 if self._photo_acc_rate is None else self._photo_acc_rate

    @property
    def max_saturation(self):
        return 372.32 if self._max_saturation is None else self._max_saturation

    @property
    def max_photosynthesis(self):
        return 1 if self._max_photosynthesis is None else self._max_photosynthesis

    @property
    def exp_saturation(self):
        return .34 if self._exp_saturation is None else self._exp_saturation

    @property
    def exp_max_photosynthesis(self):
        return .09 if self._exp_max_photosynthesis is None else self._exp_max_photosynthesis

    # photosynthetic thermal dependency
    _activation_energy = None
    _gas_constant = None
    _thermal_variability = None
    _thermal_acclimation_period = None

    @classmethod
    def set_photosynthetic_thermal_dependency(cls, activation_energy=None, gas_constant=None, thermal_variability=None,
                                              thermal_acclimation_period=None):
        cls._activation_energy = activation_energy
        cls._gas_constant = gas_constant
        cls._thermal_variability = thermal_variability
        cls._thermal_acclimation_period = thermal_acclimation_period

    @property
    def activation_energy(self):
        return 6e4 if self._activation_energy is None else self._activation_energy

    @property
    def gas_constant(self):
        return 8.31446261815324 if self._gas_constant is None else self._gas_constant

    @property
    def thermal_variability(self):
        return 2.45 if self._thermal_variability is None else self._thermal_variability

    @property
    def thermal_acclimation_period(self):
        return 60 if self._thermal_acclimation_period is None else self._thermal_acclimation_period

    # photosynthetic flow dependency
    _min_photosynthetic_flow_dependency = None
    _invariant_flow_velocity = None

    @classmethod
    def set_photosynthetic_flow_dependency(cls, min_photosynthetic_flow_dependency=None, invariant_flow_velocity=None):
        cls._min_photosynthetic_flow_dependency = min_photosynthetic_flow_dependency
        cls._invariant_flow_velocity = invariant_flow_velocity

    @property
    def min_photosynthetic_flow_dependency(self):
        return .68886964 if self._min_photosynthetic_flow_dependency is None \
            else self._min_photosynthetic_flow_dependency

    @property
    def invariant_flow_velocity(self):
        default = .17162374 if Processes().flow_micro_environment else .5173
        return default if self._invariant_flow_velocity is None else self._invariant_flow_velocity

    # population dynamics
    _growth_rate = None
    _recovery_rate = None
    _mortality_rate = None
    _bleaching_rate = None

    @classmethod
    def set_population_dynamics(cls, growth_rate=None, recovery_rate=None, mortality_rate=None, bleaching_rate=None):
        cls._growth_rate = growth_rate
        cls._recovery_rate = recovery_rate
        cls._mortality_rate = mortality_rate
        cls._bleaching_rate = bleaching_rate

    @property
    def growth_rate(self):
        return .002 if self._growth_rate is None else self._growth_rate

    @property
    def recovery_rate(self):
        return .2 if self._recovery_rate is None else self._recovery_rate

    @property
    def mortality_rate(self):
        return .04 if self._mortality_rate is None else self._mortality_rate

    @property
    def bleaching_rate(self):
        return 8 if self._bleaching_rate is None else self._bleaching_rate

    # calcification
    _calcification_constant = None
    _aragonite_saturation = None
    _dissolution_saturation = None
    _half_rate = None

    @classmethod
    def set_calcification(cls, calcification_constant=None, aragonite_saturation=None, dissolution_saturation=None,
                          half_rate=None):
        cls._calcification_constant = calcification_constant
        cls._aragonite_saturation = aragonite_saturation
        cls._dissolution_saturation = dissolution_saturation
        cls._half_rate = half_rate

    @property
    def calcification_constant(self):
        return .5 if self._calcification_constant is None else self._calcification_constant

    @property
    def aragonite_saturation(self):
        return 5 if self._aragonite_saturation is None else self._aragonite_saturation

    @property
    def dissolution_saturation(self):
        return .14587415 if self._dissolution_saturation is None else self._dissolution_saturation

    @property
    def half_rate(self):
        return .66236107 if self._half_rate is None else self._half_rate

    # morphological development
    _proportionality_form = None
    _proportionality_plate = None
    _proportionality_plate_flow = None
    _proportionality_space = None
    _proportionality_space_light = None
    _proportionality_space_flow = None
    _fitting_flow_velocity = None
    _coral_density = None

    @classmethod
    def set_morphological_development(cls, proportionality_form=None, proportionality_plate=None,
                                      proportionality_plate_flow=None, proportionality_space=None,
                                      proportionality_space_light=None, proportionality_space_flow=None,
                                      fitting_flow_velocity=None, coral_density=None):
        if proportionality_space is not None and proportionality_space > .5 / np.sqrt(2):
            msg = f'Space proportionality constant (Xs) must be smaller than 0.5/sqrt(2) : ' \
                f'{proportionality_space} > {.5 / np.sqrt(2)}'
            raise ValueError(msg)

        cls._proportionality_form = proportionality_form
        cls._proportionality_plate = proportionality_plate
        cls._proportionality_plate_flow = proportionality_plate_flow
        cls._proportionality_space = proportionality_space
        cls._proportionality_space_light = proportionality_space_light
        cls._proportionality_space_flow = proportionality_space_flow
        cls._fitting_flow_velocity = fitting_flow_velocity
        cls._coral_density = coral_density

    @property
    def proportionality_form(self):
        return .1 if self._proportionality_form is None else self._proportionality_form

    @property
    def proportionality_plate(self):
        return .5 if self._proportionality_plate is None else self._proportionality_plate

    @property
    def proportionality_plate_flow(self):
        return .1 if self._proportionality_plate_flow is None else self._proportionality_plate_flow

    @property
    def proportionality_space(self):
        return .5 / np.sqrt(2) if self._proportionality_space is None else self._proportionality_space

    @property
    def proportionality_space_light(self):
        return .1 if self._proportionality_space_light is None else self._proportionality_space_light

    @property
    def proportionality_space_flow(self):
        return .1 if self._proportionality_space_flow is None else self._proportionality_space_flow

    @property
    def fitting_flow_velocity(self):
        return .2 if self._fitting_flow_velocity is None else self._fitting_flow_velocity

    @property
    def coral_density(self):
        return 1600 if self._coral_density is None else self._coral_density

    # coral dislodgement
    _tensile_stress = None
    _drag_coefficient = None
    _water_density = None

    @classmethod
    def set_coral_dislodgement(cls, tensile_stress=None, drag_coefficient=None, water_density=None):
        cls._tensile_stress = tensile_stress
        cls._drag_coefficient = drag_coefficient
        cls._water_density = water_density

    @property
    def tensile_stress(self):
        return 2e5 if self._tensile_stress is None else self._tensile_stress

    @property
    def drag_coefficient(self):
        return 1 if self._drag_coefficient is None else self._drag_coefficient

    @property
    def water_density(self):
        return 1025 if self._water_density is None else self._water_density

    # coral recruitment
    _larvae_spawned = None
    _settle_probability = None
    _larval_diameter = None

    @classmethod
    def set_coral_recruitment(cls, larvae_spawned=None, settle_probability=None, larval_diameter=None):
        cls._larvae_spawned = larvae_spawned
        cls._settle_probability = settle_probability
        cls._larval_diameter = larval_diameter

    @property
    def larvae_spawned(self):
        return 1e6 if self._larvae_spawned is None else self._larvae_spawned

    @property
    def settle_probability(self):
        return 1e-4 if self._settle_probability is None else self._settle_probability

    @property
    def larval_diameter(self):
        return 1e-3 if self._larval_diameter is None else self._larval_diameter


if __name__ == '__main__':
    c = Constants()
    print(c.invariant_flow_velocity)
    Processes.set_processes(flow_micro_environment=False)
    print(c.invariant_flow_velocity)
    Constants.set_photosynthetic_flow_dependency(invariant_flow_velocity=.1)
    print(c.invariant_flow_velocity)
    Constants.set_constant('invariant_flow_velocity', 10)
    print(c.invariant_flow_velocity)
    Constants.set_constant('invariant_flow_velocity', None)
    print(c.invariant_flow_velocity)
    Processes.set_processes(flow_micro_environment=True)
    print(c.invariant_flow_velocity)
