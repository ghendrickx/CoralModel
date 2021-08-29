import logging

LOG = logging.getLogger(__name__)


class _CoralConstants:

    _constants = ('species_constant',)

    def __init__(self):
        """Initiate with default values."""
        self.species_constant = 1.

    def __repr__(self):
        """Representation."""
        return f'_CoralConstants'

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


class _CoralStates:

    def __init__(self):
        self._healthy = 1
        self._recovered = 0
        self._pale = 0
        self._bleached = 0

    @staticmethod
    def _set_state(state):
        """Set coral state value, which must be in range [0, 1].

        :param state: coral state value
        :type state: float
        """
        assert 0 < state < 1
        return state

    @property
    def healthy(self):
        """
        :return: healthy coral cover
        :rtype: float
        """
        return self._healthy

    @healthy.setter
    def healthy(self, state):
        """
        :param state: healthy coral cover
        :type state: float
        """
        self._healthy = self._set_state(state)

    @property
    def recovering(self):
        """
        :return: recovering coral cover
        :rtype: float
        """
        return self._recovered

    @recovering.setter
    def recovering(self, state):
        """
        :param state: recovering coral cover
        :type state: float
        """
        self._recovered = self._set_state(state)

    @property
    def pale(self):
        """
        :return: pale coral cover
        :rtype: float
        """
        return self._pale

    @pale.setter
    def pale(self, state):
        """
        :param state: pale coral cover
        :type state: float
        """
        self._pale = self._set_state(state)

    @property
    def bleaching(self):
        """
        :return: bleaching coral cover
        :rtype: float
        """
        return self._bleached

    @bleaching.setter
    def bleaching(self, state):
        """
        :param state: bleaching coral cover
        :type state: float
        """
        self._bleached = self._set_state(state)

    @property
    def all(self):
        """
        :return: total coral cover
        :rtype: float
        """
        return sum([self.healthy, self.recovering, self.pale, self.bleaching])


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


class _CoralEnvironment:

    def __init__(self):
        self.light = None
        self.temperature = None
        self.flow = None


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
        # store CoralSpecies
        self.add_species(self)

    def __repr__(self):
        """Representation."""
        return f'Coral(name={self.name})'

    def __str__(self):
        """String-representation."""
        return f'Coral: {self.name}'

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


class Coral:

    def __init__(self, species):
        """
        :param species: coral species
        :type species: CoralSpecies
        """
        self._constants = species.constants
        self._states = _CoralStates()
        self._morphology = species.initial_morphology

    @property
    def constants(self):
        """
        :return: coral constants
        :rtype: _CoralConstants
        """
        return self._constants

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
        assert isinstance(coral_states, _CoralStates)
        self._states = coral_states

    @property
    def morphology(self):
        """
        :return: coral morphology
        :rtype: _CoralMorphology
        """
        return self._morphology

    def set_characteristic(self, characteristic, value):
        """Set non-essential coral characteristic.

        :param characteristic: characteristic's name
        :param value: characteristic's value

        :type characteristic: str
        :type value: float, iterable, bool
        """
        if not hasattr(self, characteristic):
            setattr(self, characteristic, value)

    def get_characteristic(self, characteristic):
        """Get non-essential coral characteristic.

        :param characteristic: characteristic's name
        :type characteristic: str

        :return: characteristic's value
        :rtype: float, iterable, bool
        """
        if hasattr(self, characteristic):
            return getattr(self, characteristic)


class _CoralCollection:

    _initiated = False

    def __init__(self):
        self.is_initiated()
        self._corals = {species: Coral(species) for species in CoralSpecies.get_species()}

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
