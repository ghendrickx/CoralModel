import numpy as np


class Constants:

    @classmethod
    def set_constant(cls, constant, value):
        if hasattr(cls, constant):
            setattr(cls, f'_{constant}', value)
        else:
            msg = f'There is no constant named \"{constant}\".'
            raise ValueError(msg)

    # light micro-environment
    _lac_default = None
    _theta_max = None

    @classmethod
    def set_light_micro_environment(cls, lac_default=None, theta_max=None):
        [setattr(cls, f'_{key}', value) for key, value in locals().items()]
        # cls._lac_default = lac_default
        # cls._theta_max = theta_max

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


if __name__ == '__main__':
    c = Constants()
    print(c.lac_default)
    Constants.set_light_micro_environment(lac_default=.2)
    print(c.lac_default)
    c.set_light_micro_environment(lac_default=.3)
    print(c.lac_default)
    Constants.set_constant('lac_default', .4)
    print(c.lac_default)
    c.set_light_micro_environment(lac_default=.5)
    print(c.lac_default)
