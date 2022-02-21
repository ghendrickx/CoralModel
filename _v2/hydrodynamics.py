"""
Hydrodynamic model.

Author: Gijs G. Hendrickx
"""
import logging
import sys

import numpy as np

from _v2._errors import InitialisationError
from _v2.grid import Grid

LOG = logging.getLogger(__name__)

GRAVITY = 9.81


class Hydrodynamics:
    __modes = (None, '0D', '1D', '2D')
    _model = None

    _grid = None
    _init_grid = True  # (re-)initiate grid

    def __init__(self, mode=None):
        """
        :param mode: mode of hydrodynamic model, defaults to None
        :type mode: str, optional
        """
        self._set_model(mode)

    @classmethod
    def _set_model(cls, mode):
        """Set hydrodynamic model

        :param mode: mode of hydrodynamic model
        :type mode: str, None

        :return: hydrodynamic model
        :rtype: _Base
        """
        if mode not in cls.__modes:
            raise ValueError

        model_cls = '_Base' if mode is None else f'Reef{mode}'

        if cls._model is not None:
            print(f'Hydrodynamic model already defined: {cls._model}')
            if input(f'Overwrite with {model_cls}? [y/n]') == 'n':
                return

        cls._model = getattr(sys.modules[__name__], model_cls)()

    def update(self, storm=False):
        """Update hydrodynamic model.

        :param storm: storm hydrodynamics, defaults to False
        :type storm: bool, optional
        """
        # TODO: Update hydrodynamic conditions
        self._model.update(self.grid, storm=storm)

    @property
    def model(self):
        """
        :return: hydrodynamic model
        :rtype: _Base
        """
        return self._model

    @property
    def x_coordinates(self):
        """
        :return: x-coordinate(s)
        :rtype: None, float, iterable
        """
        return self.model.x

    @property
    def y_coordinates(self):
        """
        :return: y-coordinate(s)
        :rtype: None, float, iterable
        """
        return self.model.y

    @property
    def grid(self):
        """
        :return: (hydrodynamic) grid
        :rtype: Grid
        """
        if self._init_grid:
            self._grid = Grid()
            self._init_grid = False
        return self._grid

    @property
    def current_velocity(self):
        """
        :return: current flow velocity
        :rtype: float, iterable
        """
        return self.model.current_velocity

    @property
    def wave_velocity(self):
        """
        :return: wave flow velocity
        :rtype: float, iterable
        """
        return self.model.wave_velocity

    @property
    def wave_period(self):
        """
        :return: wave period
        :rtype: float, iterable
        """
        return self.model.wave_period


class _Base:
    update_interval = None
    update_interval_storm = None

    _initialised = False

    def __init__(self, calculations=False):
        """
        :param calculations: execute calculations, i.e. execute update-method, defaults to False
        :type calculations: bool, optional
        """
        self._calc = calculations

        self._current_velocity = None
        self._wave_velocity = None
        self._wave_period = None

        self._storm_wave_velocity = None

    def __str__(self):
        """String-representation."""
        return self.__class__.__name__

    @property
    def x(self):
        """
        :return: x-coordinate(s)
        :rtype: float, iterable
        """
        return

    @property
    def y(self):
        """
        :return: y-coordinate(s)
        :rtype: float, iterable
        """
        return

    @property
    def current_velocity(self):
        """
        :return: current flow velocity
        :rtype: float, iterable
        """
        return self._current_velocity

    @property
    def wave_velocity(self):
        """
        :return: wave velocity
        :rtype: float, iterable
        """
        return self._wave_velocity

    @property
    def wave_period(self):
        """
        :return: wave period
        :rtype: float, iterable
        """
        return self._wave_period

    @property
    def storm_wave_velocity(self):
        """
        :return: wave velocity, storm conditions
        :rtype: float, iterable
        """
        return self._storm_wave_velocity

    def _initialise(self):
        """Initialise hydrodynamic model."""

    def initialise(self):
        """Initialise hydrodynamic model."""
        self._initialise()
        self._initialised = True

    def initialize(self):
        """Initialise hyodrynamic model; American-spelling."""
        self.initialise()

    def update(self, grid, storm=False):
        """Update hydrodynamic model: grid-level.

        :param grid: grid
        :param storm: storm conditions, defaults to False

        :type grid: Grid
        :type storm: bool, optional
        """
        if self._calc:
            [self._update(cell, storm) for cell in grid.cells]

    def _update(self, cell, storm=False):
        """Update hydrodynamic model: cell-level.

        :param cell: grid cell
        :param storm: storm conditions, defaults to False

        :type cell: Cell
        :type storm: bool, optional
        """

    def finalise(self):
        """Finalise hydrodynamic model."""

    def finalize(self):
        """Finalise hydrodynamic model; American-spelling."""
        self.finalise()


class Reef0D(_Base):
    update_interval = None
    update_interval_storm = None

    def __init__(self, tidal_amplitude, wave_height, wave_period, storm_wave_height, storm_wave_period):
        super().__init__(calculations=True)

        self._tidal_amplitude = tidal_amplitude
        self._wave_height = wave_height
        self._wave_period = wave_period

        self._storm_wave_height = storm_wave_height
        self._storm_wave_period = storm_wave_period

    def _set_tidal_velocity(self, cell):
        self._current_velocity = 1 / np.sqrt(2) * np.sqrt(GRAVITY / cell.water_depth) * self._tidal_amplitude

    def _set_wave_velocity(self, cell):
        # TODO: Verify the validness of this approximation!
        self._wave_velocity = np.sqrt(GRAVITY / cell.water_depth) * self._wave_height

    def _set_storm_wave_velocity(self, cell):
        # TODO: Verify the validness of this approximation!
        self._storm_wave_velocity = np.sqrt(GRAVITY / cell.water_depth) * self._storm_wave_height

    def _initialise(self):
        """Initialise 0D-hydrodynamic model."""
        if Grid.get_size() > 1:
            Grid.reset()
            LOG.warning('Grid is reset and will be set to a 0D-grid: (0, 0).')

            # re-initiate grid
            Grid(x=0, y=0)

    def _update(self, cell, storm=False):
        """Update 0D-hydrodynamic model.

        :param cell: grid cell
        :param storm: storm conditions, defaults to False

        :type cell: Cell
        :type storm: bool, optional
        """
        # determine tidal flow velocity
        self._set_tidal_velocity(cell)

        # determine relevant wave velocity
        self._set_storm_wave_velocity(cell) if storm else self._set_wave_velocity(cell)

        # return relevant hydrodynamic data
        if storm:
            return self.current_velocity, self.storm_wave_velocity
        return self.current_velocity, self.wave_velocity, self.wave_period


class Reef1D(_Base):
    update_interval = None
    update_interval_storm = None

    def __new__(cls, *args, **kwargs):
        raise NotImplementedError

    def __init__(self):
        super().__init__(calculations=True)

    @property
    def x(self):
        return 0, 0

    @property
    def y(self):
        return 0

    def _update(self, cell, storm=False):
        """Update hydrodynamic model.

        :param cell: grid cell
        :param storm: storm conditions, defaults to False

        :type cell: Cell
        :type storm: bool, optional
        """
        # TODO: Define _update()-method
        if storm:
            return None, None
        return None, None, None


class Reef2D(_Base):
    update_interval = None
    update_interval_storm = None

    _wrapper = None
    _dfm = None
    _dimr = None

    _x = None
    _y = None
    _n_internal_cells = None

    def __new__(cls, *args, **kwargs):
        raise NotImplementedError

    def __init__(self):
        super().__init__(calculations=True)

        self._import_wrapper()

    @classmethod
    def _import_wrapper(cls):
        """Import BMI-wrapper."""
        try:
            import bmi.wrapper
        except ImportError:
            LOG.critical('BMI-wrapper cannot be imported.')
        else:
            cls._wrapper = bmi.wrapper

    @property
    def dfm(self):
        """
        :return: DFM-model
        :rtype: BMIWrapper
        """
        return self._dfm

    @classmethod
    def set_dfm(cls, engine, config_file):
        """Set DFM-model.

        :param engine: DFM *.dll-file
        :param config_file: configuration file (*.mdu)

        :type engine: str
        :type config_file: str
        """
        if cls._wrapper is None:
            cls._import_wrapper()

        cls._dfm = cls._wrapper.BMIWrapper(engine=engine, configfile=config_file)

    @property
    def dimr(self):
        """
        :return: DIMR-model
        :rtype: BMIWrapper
        """
        return self._dimr

    @classmethod
    def set_dimr(cls, engine, config_file):
        """Set DIMR-model.

        :param engine: DIMR *.dll-file
        :param config_file: configuration file (*.xml)

        :type engine: str
        :type config_file: str
        """
        if cls._wrapper is None:
            cls._import_wrapper()

        cls._dimr = cls._wrapper.BMIWrapper(engine=engine, configfile=config_file)

    @property
    def _model(self):
        """
        :return: highest level model defined: 1. DIMR, 2. DFM.
        :rtype: BMIWrapper
        """
        return self._dfm if self._dimr is None else self._dimr

    @property
    def x(self):
        """
        :return: x-coordinates
        :rtype: numpy.array
        """
        if self._x is None and self._initialised:
            self._x = self.dfm.get_var('xzw')
        return self._x

    @property
    def y(self):
        """
        :return: y-coordinates
        :rtype: numpy.array
        """
        if self._y is None and self._initialised:
            self._y = self.dfm.get_var('yzw')
        return self._y

    @property
    def n_internal_cells(self):
        """
        :return: number of non-boundary boxes, i.e. within-domain boxes
        :rtype: int
        """
        if self._n_internal_cells is None and self._initialised:
            self._n_internal_cells = self.dfm.get_var('ndxi')
        return self._n_internal_cells

    def _initialise(self):
        """Initiate DFM/DIMR."""
        if self._model is None:
            msg = f'No model set, thus nothing to initialise: Set a DFM- (and DIMR-) model.'
            raise InitialisationError(msg)

        self._model.initialize()

        if Grid.get_size() > 0:
            Grid.reset()
            LOG.critical('Grid is reset and will be drawn from DFM-model.')

        # re-initialise grid
        Grid(x=self.x, y=self.y)

    def _update(self, cell, storm=False):
        """Update DFM.

        :param cell: grid cell
        :param storm: storm conditions, defaults to False

        :type cell: Cell
        :type storm: bool, optional
        """
        # TODO: Define _update()-method
        if storm:
            return None, None
        return None, None, None

    def finalise(self):
        """Finalise DFM."""
        self._model.finalize()
