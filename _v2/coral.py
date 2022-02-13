"""
Coral definition.

Author: Gijs G. Hendrickx
"""
import logging

import numpy as np

LOG = logging.getLogger(__name__)


class _CoralConstants:

    _constants = ('species_constant',)

    def __init__(self):
        """Initiate with default values."""
        self.species_constant = 1

    def __repr__(self):
        """Representation."""
        return f'_CoralConstants()'

    def __str__(self):
        """String-representation."""
        return str({key: getattr(self, key) for key in self._constants})

    def set_constants(self, **kwargs):
        """Set coral constants, different from default values.

        :param kwargs: coral constants
        :type kwargs: float, None
        """
        [self._def(key, value) for key, value in kwargs.items() if key in self._constants]

    def _def(self, cst, value):
        """Overwrite default with defined value, unless None-type is provided as value.

        :param cst: coral constant
        :param value: defined value

        :type cst: str
        :type value: float, None
        """
        setattr(self, cst, value) if value is not None else None


class _CoralVariables:

    def __init__(
            self, light=None, biomass=None, in_canopy_flow=None, tbl=None, temperature=None,
            lower_limit=None, upper_limit=None, photosynthesis=None, calcification=None
    ):
        """Coral variables: Values that will change along the simulation, but which are coral-specific.

        :param light: representative light-intensity
        :param biomass: coral biomass
        :param in_canopy_flow: in-canopy flow
        :param tbl: thermal boundary layer
        :param temperature: coral temperature
        :param lower_limit: photosynthetic thermal dependency, lower limit
        :param upper_limit: photosynthetic thermal dependency, upper limit
        :param photosynthesis: photosynthetic rate
        :param calcification: calcification rate
        """
        self.light = light
        self.biomass = biomass
        self.in_canopy_flow = in_canopy_flow
        self.tbl = tbl
        self.temperature = temperature
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.photosynthesis = photosynthesis
        self.calcification = calcification

    def __repr__(self):
        """Object-representation."""
        return f'_CoralVariables(**kwargs)'

    def __str__(self):
        """String-representation."""
        return f'_CoralVariables'


class _CoralState:

    def __init__(self, healthy, recovered, pale, bleached):
        """
        :param healthy: healthy coral cover
        :param recovered: recovered coral cover
        :param pale: pale coral cover
        :param bleached: bleached coral cover

        :type healthy: float
        :type recovered: float
        :type pale: float
        :type bleached: float
        """
        self.healthy = healthy
        self.recovered = recovered
        self.pale = pale
        self.bleached = bleached

    def __repr__(self):
        """Object-representation."""
        return f'_CoralState(healthy={self.healthy}, recovered={self.recovered}, pale={self.pale}, ' \
            f'bleached={self.bleached})'

    def __str__(self):
        """String-representation."""
        return f'Coral state: healthy = {self.healthy:.4f}; recovered = {self.recovered:.4f}; pale = {self.pale:.4f};' \
            f' bleached = {self.bleached:.4f}'

    @property
    def sum(self):
        """
        :return: total coral cover
        :rtype: float
        """
        return sum([self.healthy, self.recovered, self.pale, self.bleached])


class _CoralStates:

    def __init__(self, healthy=1, recovered=0, pale=0, bleached=0):
        """Initiate _CoralStates by setting an initial _CoralState.

        :param healthy: healthy coral cover, defaults to 1
        :param recovered: recovered coral cover, defaults to 0
        :param pale: pale coral cover, defaults to 0
        :param bleached: bleached coral cover, defaults to 0

        :type healthy: float, optional
        :type recovered: float, optional
        :type pale: float, optional
        :type bleached: float, optional
        """
        assert all(0 <= s <= 1 for s in (healthy, recovered, pale, bleached))
        self._states = [_CoralState(healthy, recovered, pale, bleached)]

    def __repr__(self):
        """Object-representation."""
        return f'_CoralStates(**kwargs)'

    def __str__(self):
        """String-representation."""
        return f'Coral-states: healthy = {self.healthy}; recovered = {self.recovered}; pale = {self.pale}; ' \
            f'bleached = {self.bleached}'

    def __len__(self):
        """Object-length."""
        return len(self._states)

    def append(self, state):
        """Append coral-state to list of coral-states.

        :param state: coral-state
        :type state: _CoralState
        """
        self._states.append(state)

    def extend(self, states):
        """Append multiple coral-states to list at once.

        :param states: coral-states
        :type states: list[_CoralState]
        """
        self._states.extend(states)

    def add_state(self, healthy, recovered, pale, bleached):
        """Add coral-state by defining its coral cover fractions.

        :param healthy: healthy coral cover
        :param recovered: recovered coral cover
        :param pale: pale coral cover
        :param bleached: bleached coral cover

        :type healthy: float
        :type recovered: float
        :type pale: float
        :type bleached: float
        """
        self.append(_CoralState(healthy, recovered, pale, bleached))

    def mod_state(self, idx, healthy, recovered, pale, bleached):
        """Modify coral-state by index of the list of coral-states.

        :param idx: list-index
        :param healthy: healthy coral cover
        :param recovered: recovered coral cover
        :param pale: pale coral cover
        :param bleached: bleached coral cover

        :type idx: int
        :type healthy: float
        :type recovered: float
        :type pale: float
        :type bleached: float
        """
        self._states[idx] = _CoralState(healthy, recovered, pale, bleached)

    def full_reset(self, healthy=1, recovered=0, pale=0, bleached=0):
        """Fully reset the coral-states by re-initiating the object.

        :param healthy: healthy coral cover, defaults to 1
        :param recovered: recovered coral cover, defaults to 0
        :param pale: pale coral cover, defaults to 0
        :param bleached: bleached coral cover, defaults to 0

        :type healthy: float, optional
        :type recovered: float, optional
        :type pale: float, optional
        :type bleached: float, optional
        """
        self.__init__(healthy, recovered, pale, bleached)

    def last_reset(self):
        """Reset the coral-states by keeping the last coral-state of the list."""
        self._states = self._states[-1]

    @property
    def states(self):
        """
        :return: coral-states
        :rtype: list
        """
        return self._states

    @property
    def healthy(self):
        """
        :return: healthy coral cover, all states
        :rtype: list
        """
        return [s.healthy for s in self._states]

    @property
    def recovered(self):
        """
        :return: recovered coral cover, all states
        :rtype: list
        """
        return [s.recovered for s in self._states]

    @property
    def pale(self):
        """
        :return: pale coral cover, all states
        :rtype: list
        """
        return [s.pale for s in self._states]

    @property
    def bleached(self):
        """
        :return: bleached coral cover, all states
        :rtype: list
        """
        return [s.bleached for s in self._states]

    @property
    def sum(self):
        """
        :return: total coral cover, all states
        :rtype: list
        """
        return [s.sum for s in self._states]


class _CoralMorphology:

    def __init__(self, diameter, height, distance, base_diameter=None, plate_thickness=None):
        """
        :param diameter: coral diameter
        :param height: coral height
        :param distance: coral distance
        :param base_diameter: coral base diameter, defaults to None
        :param plate_thickness: coral plate thickness, defaults to None

        :type diameter: float
        :type height: float
        :type distance: float
        :type base_diameter: float, optional
        :type plate_thickness: float, optional
        """
        self.diameter = diameter
        self.height = height
        self.base_diameter = diameter if base_diameter is None else base_diameter
        self.plate_thickness = height if plate_thickness is None else plate_thickness
        self.distance = distance

    def __repr__(self):
        """Object-representation."""
        return f'_CoralMorphology(diameter={self.diameter}, height={self.height}, distance={self.distance}, ' \
            f'base_diameter={self.base_diameter}, plate_thickness={self.plate_thickness})'

    def __str__(self):
        """String-representation."""
        return f'Coral morphology'

    def update(self, volume, rf=None, rp=None, rs=None):
        """Update coral morphology based on the coral's volume and its morphological ratios.

        :param volume: updated coral volume
        :param rf: updated form ratio, defaults to None
        :param rp: updated plate ratio, defaults to None
        :param rs: updated spacing ratio, defaults to None

        :type volume: float
        :type rf: float, optional
        :type rp: float, optional
        :type rs: float, optional
        """
        # default ratios: current ratios
        rf = self.form_ratio if rf is None else rf
        rp = self.plate_ratio if rp is None else rp
        rs = self.spacing_ratio if rs is None else rs

        def vc2dc():
            """Coral volume to coral plate diameter."""
            return ((4. * volume) / (np.pi * rf * rp * (1. + rp - rp ** 2))) ** (1. / 3.)

        def vc2hc():
            """Coral volume to coral height."""
            return ((4. * volume * rf ** 2) / (np.pi * rp * (1. + rp - rp ** 2))) ** (1. / 3.)

        def vc2bc():
            """Coral volume > diameter of the base."""
            return ((4. * volume * rp ** 2) / (np.pi * rf * (1. + rp - rp ** 2))) ** (1. / 3.)

        def vc2tc():
            """Coral volume > thickness of the plate."""
            return ((4. * volume * rf ** 2 * rp ** 2) / (np.pi * (1. + rp - rp ** 2))) ** (1. / 3.)

        def vc2ac():
            """Coral volume > axial distance."""
            return (1. / rs) * ((4. * volume) / (np.pi * rf * rp * (1. + rp - rp ** 2))) ** (1. / 3.)

        # update morphology
        self.diameter = vc2dc()
        self.height = vc2hc()
        self.base_diameter = vc2bc()
        self.plate_thickness = vc2tc()
        self.distance = vc2ac()

    @property
    def representative_diameter(self):
        """Diameter representing the combined effect of the base and plate diameters.

        :return: representative diameter
        :rtype: float
        """
        return (self.base_diameter * (self.height - self.plate_thickness) + self.diameter * self.plate_thickness) / \
            self.height

    @property
    def form_ratio(self):
        """
        :return: coral morphology form ratio
        :rtype: float
        """
        return self.height / self.diameter

    @property
    def plate_ratio(self):
        """
        :return: coral morphology plate ratio
        :rtype: float
        """
        return self.base_diameter / self.diameter

    @property
    def spacing_ratio(self):
        """
        :return: coral morphology spacing ratio
        :rtype: float
        """
        return self.diameter / self.distance

    @property
    def volume(self):
        """
        :return: coral volume
        :rtype: float
        """
        return .25 * np.pi * (
                (self.height - self.plate_thickness) * self.base_diameter ** 2 +
                self.plate_thickness * self.diameter ** 2
        )


class _CoralEnvironment:

    def __init__(self):
        """Coral environment."""
        self.light = None
        self.temperature = None
        self.flow = None

    def __repr__(self):
        """Object-representation."""
        return f'_CoralEnvironment()'

    def __str__(self):
        """String-representation."""
        return f'Coral environment'


class CoralSpecies:

    _coral_species = set()
    _re_initiate = False

    def __init__(self, diameter, height, distance, base_diameter=None, plate_thickness=None, name=None):
        """
        :param diameter: coral diameter
        :param height: coral height
        :param distance: coral distance
        :param base_diameter: coral base diameter, defaults to None
        :param plate_thickness: coral plate thickness, defaults to None
        :param name: name of coral-type, defaults to None

        :type diameter: float
        :type height: float
        :type distance: float
        :type base_diameter: float, optional
        :type plate_thickness: float, optional
        :type name: str, optional
        """
        # initiate CoralSpecies
        self.name = name
        self._constants = _CoralConstants()
        self._initial_morphology = _CoralMorphology(diameter, height, distance, base_diameter, plate_thickness)
        # constant coral variables
        self._variables = None
        # store CoralSpecies
        self.add_species(self)

    def __repr__(self):
        """Object-representation."""
        return f'CoralSpecies(diameter={self._initial_morphology.diameter}, height={self._initial_morphology.height},' \
            f' distance={self._initial_morphology.distance}, base_diameter={self._initial_morphology.base_diameter}, ' \
            f'plate_thickness={self._initial_morphology.plate_thickness}, name={self.name})'

    def __str__(self):
        """String-representation."""
        return f'Coral species: {self.name}'

    @classmethod
    def add_species(cls, coral_species):
        """Add CoralSpecies to collection of species.

        :param coral_species: coral species
        :type coral_species: CoralSpecies
        """
        # auto-name species
        if coral_species.name is None:
            coral_species.name = f'Coral-{len(cls._coral_species)}'
        # store species
        cls._coral_species.add(coral_species)
        # inclusion check
        if _CoralCollection.is_initiated():
            LOG.critical(
                f'CoralCollection already initiated: {coral_species.name} is not included in that initiated instance.'
            )
            cls._re_initiate = True

    @classmethod
    def re_initiate(cls):
        """Re-initiate CoralCollections due to addition of more CoralSpecies after CoralCollections have been initiated,
        which results in the exclusion of the newly added CoralSpecies from the model.

        :return: re-initiation required
        :rtype: bool
        """
        return cls._re_initiate

    @classmethod
    def get_species(cls):
        """Get all defined CoralSpecies.

        :return: coral species
        :rtype: set
        """
        return cls._coral_species

    @property
    def constants(self):
        """
        :return: coral constants
        :rtype: _CoralConstants
        """
        return self._constants

    def set_constants(self, species_constant=None):
        """Set coral constants, different from default values.

        :param species_constant: coral species constant, defaults to None
        :type species_constant: float, optional
        """
        kwargs = {key: value for key, value in locals().items() if not value == self}
        self._constants.set_constants(**kwargs)

    @property
    def initial_morphology(self):
        """
        :return: initial morphology
        :rtype: _CoralMorphology
        """
        return self._initial_morphology

    @property
    def variables(self):
        """
        :return: default coral variables for given coral species
        :rtype: _CoralVariables
        """
        return self._variables

    def set_variables(self, **kwargs):
        """
        :param kwargs: species-constant coral variables
        :type kwargs: iterable, float
        """
        self._variables = _CoralVariables(**kwargs)


class Coral:

    def __init__(self, species):
        """
        :param species: coral species
        :type species: CoralSpecies
        """
        self._species = species
        self._constants = species.constants
        self._vars = _CoralVariables()
        self._states = _CoralStates()
        self._morphology = species.initial_morphology

    def __repr__(self):
        """Object-representation."""
        return f'Coral({self._species})'

    def __str__(self):
        """String-representation."""
        return f'Coral: {self._species}'

    @property
    def constants(self):
        """
        :return: coral constants
        :rtype: _CoralConstants
        """
        return self._constants

    @property
    def vars(self):
        """
        :return: coral variables
        :rtype: _CoralVariables
        """
        return self._vars

    @vars.setter
    def vars(self, variables):
        """
        :param variables: coral variables
        :type variables: _CoralVariables
        """
        self._vars = variables

    @property
    def states(self):
        """
        :return: coral population states
        :rtype: _CoralStates
        """
        return self._states

    @states.setter
    def states(self, coral_states):
        """
        :param coral_states: coral population states
        :type coral_states: _CoralStates
        """
        self._states = coral_states

    @property
    def morphology(self):
        """
        :return: coral morphology
        :rtype: _CoralMorphology
        """
        return self._morphology


class _CoralCollection:

    _initiated = False

    def __init__(self):
        self.is_initiated()
        self._corals = {species: Coral(species) for species in CoralSpecies.get_species()}

    def __repr__(self):
        """Object-representation."""
        return f'_CoralCollection()'

    def __str__(self):
        """String-representation."""
        return f'Coral collection of {len(self)} corals'

    def __len__(self):
        """Object-length."""
        return len(self._corals)

    @classmethod
    def is_initiated(cls):
        """Communicate if CoralCollection is already initiated, to raise critical warning when new CoralSpecies are
        defined after a CoralCollection has been initiated.
        """
        cls._initiated = True

    @property
    def corals(self):
        """
        :return: corals
        :rtype: dict
        """
        return self._corals

    @property
    def coral_list(self):
        """
        :return: corals
        :rtype: list
        """
        return list(self._corals.values())


if __name__ == '__main__':
    cs = CoralSpecies(.1, .1, .1, .1, .1, name='special coral')
    c = Coral(cs)
    print(c.__dict__)
