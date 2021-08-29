import numpy as np

from _v2.coral import Coral
from _v2.settings import Constants, Processes


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
        pass

    def _velocities(self):
        pass


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
