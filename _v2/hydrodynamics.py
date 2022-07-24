"""
Hydrodynamic model.

Author: Gijs G. Hendrickx
"""
import logging
import sys

import numpy as np
import scipy.optimize as opt

from _v2._errors import InitialisationError, DataError
from _v2.grid import Grid

LOG = logging.getLogger(__name__)

_GRAVITY = 9.81


class Hydrodynamics:
    __modes = (None, '0D', '1D', '2D')
    _model = None

    _grid = None
    _grid_initialised = False

    def __init__(self, mode=None, init_grid=True):
        """
        :param mode: mode of hydrodynamic model, defaults to None
        :param init_grid: initialise grid, defaults to True

        :type mode: str, optional
        :type init_grid: bool, optional
        """
        self._set_model(mode)
        if init_grid:
            self.initialise_grid()

    @classmethod
    def _set_model(cls, mode):
        """Set hydrodynamic model

        :param mode: mode of hydrodynamic model
        :type mode: str, None

        :return: hydrodynamic model
        :rtype: _Base
        """
        if mode not in cls.__modes:
            msg = f'Unknown mode for hydrodynamic model ({mode}); choose one of {cls.__modes}.'
            raise ValueError(msg)

        model_cls = '_Base' if mode is None else f'Reef{mode}'

        if cls._model is not None:
            print(f'Hydrodynamic model already defined: {cls._model}')
            if input(f'Overwrite with {model_cls}? [y/n]') == 'y':
                cls._model = getattr(sys.modules[__name__], model_cls)()

    def initialise(self):
        """Initialise hydrodynamic model."""
        self._model.initialise()

    def initialize(self):
        """Initialise hydrodynamic model; American-spelling."""
        self.initialise()

    def update(self, storm=False):
        """Update hydrodynamic model.

        :param storm: storm hydrodynamics, defaults to False
        :type storm: bool, optional
        """
        # TODO: Update hydrodynamic conditions
        self._model.update(self.grid, storm=storm)

    def finalise(self):
        """Finalise hydrodynamic model."""
        self._model.finalise()

    def finalize(self):
        """Finalise hydrodynamic model; American-spelling."""
        self.finalise()

    @property
    def model(self):
        """
        :return: hydrodynamic model
        :rtype: _Base
        """
        return self._model

    def set_model_environment(
            self, tidal_range, tidal_period, wave_height, wave_period, storm_wave_height, storm_wave_period
    ):
        """Set environmental conditions for hydrodynamic model.

        :param tidal_range: tidal range
        :param tidal_period: tidal period
        :param wave_height: (significant) wave height
        :param wave_period: (peak) wave period
        :param storm_wave_height: (significant) storm wave height
        :param storm_wave_period: (peak) storm wave period

        :type tidal_range: float
        :type tidal_period: float
        :type wave_height: float
        :type wave_period: float
        :type storm_wave_height: float
        :type storm_wave_period: float
        """
        if self.model is None:
            # no hydrodynamic model defined: raise error
            msg = 'No hydrodynamic model defined.'
            raise InitialisationError(msg)

        elif isinstance(self._model, Reef0D):
            # set the environmental conditions: Reef0D
            self._model.set_environment(
                tidal_range=tidal_range, tidal_period=tidal_period, wave_height=wave_height, wave_period=wave_period,
                storm_wave_height=storm_wave_height, storm_wave_period=storm_wave_period
            )

        else:
            # defined hydrodynamic model does not have such a method: raise error
            msg = f'The chosen hydrodynamic model ({self.model}) does not require/allow setting its environment.'
            raise NotImplementedError(msg)

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
        # initialise grid if not done yet
        if not self._grid_initialised:
            self.initialise_grid()

        # return (initialised) grid
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

    def initialise_grid(self):
        """Initialise grid."""
        # initialise Grid-object
        grid = Grid()

        # define grid based on hydrodynamic model
        if self.model._grid_from_hydrodynamic_model:
            grid.reset()
            grid.grid_from_xy(self.model.x, self.model.y)

        # set grid to _grid-property
        self._grid = grid
        # grid is initialised
        self._grid_initialised = True


class _Base:
    update_interval = None
    update_interval_storm = None

    _initialised = False
    _grid_from_hydrodynamic_model = False

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
        """Initialise hydrodynamic model; American-spelling."""
        self.initialise()

    def update(self, grid, storm=False):
        """Update hydrodynamic model: grid-level.

        :param grid: grid
        :param storm: storm conditions, defaults to False

        :type grid: Grid
        :type storm: bool, optional
        """
        if not self._initialised:
            msg = f'Hydrodynamic model is not yet initialised. Please, do so before updating the hydrodynamic output.'
            raise InitialisationError(msg)

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

    def __init__(self):
        super().__init__(calculations=True)

        self._tidal_range = None
        self._tidal_period = None
        self._wave_height = None
        self._wave_period = None
        self._storm_wave_height = None
        self._storm_wave_period = None

    def set_environment(self, tidal_range, tidal_period, wave_height, wave_period, storm_wave_height, storm_wave_period):
        """Set environmental conditions for hydrodynamic model.

        :param tidal_range: tidal range
        :param tidal_period: tidal period
        :param wave_height: (significant) wave height
        :param wave_period: (peak) wave period
        :param storm_wave_height: (significant) storm wave height
        :param storm_wave_period: (peak) storm wave period

        :type tidal_range: float
        :type tidal_period: float
        :type wave_height: float
        :type wave_period: float
        :type storm_wave_height: float
        :type storm_wave_period: float
        """
        # set base environmental conditions
        self.set_base_conditions(
            tidal_range=tidal_range, tidal_period=tidal_period, wave_height=wave_height, wave_period=wave_period
        )
        # set storm environmental conditions
        self.set_storm_conditions(
            storm_wave_height=storm_wave_height, storm_wave_period=storm_wave_period
        )

    def set_base_conditions(self, tidal_range, tidal_period, wave_height, wave_period):
        """Set "base" environmental conditions (i.e. excluding storm conditions) for hydrodynamic model.

        :param tidal_range: tidal range
        :param tidal_period: tidal period
        :param wave_height: (significant) wave height
        :param wave_period: (peak) wave period

        :type tidal_range: float
        :type tidal_period: float
        :type wave_height: float
        :type wave_period: float
        """
        self._tidal_range = tidal_range
        self._tidal_period = tidal_period
        self._wave_height = wave_height
        self._wave_period = wave_period

    def set_storm_conditions(self, storm_wave_height, storm_wave_period):
        """Set storm environmental conditions for hydrodynamic model.

        :param storm_wave_height: (significant) storm wave height
        :param storm_wave_period: (peak) storm wave period

        :type storm_wave_height: float
        :type storm_wave_period: float
        """
        self._storm_wave_height = storm_wave_height
        self._storm_wave_period = storm_wave_period

    @staticmethod
    def _estimate_wave_velocity(wave_height, wave_period, water_depth):
        """Estimate wave velocity based on the wave height and water depth.

        :param wave_height: wave height
        :param water_depth: water depth

        :type wave_height: float, iterable
        :type water_depth: float

        :return: wave velocity estimation
        :rtype: float, iterable
        """
        # tidal frequency
        frequency = 2 * np.pi / wave_period

        def dispersion(k):
            """Dispersion relation."""
            return _GRAVITY * k * np.tanh(k * water_depth) - (frequency ** 2)

        # solve for wave number
        wave_number = opt.newton(dispersion, x0=frequency / np.sqrt(_GRAVITY * water_depth))

        # depth-averaged, tidal-averaged horizontal velocity
        return (frequency * wave_height) / (wave_number * water_depth * np.pi)

    def _set_tidal_velocity(self, cell):
        """Estimate tidal velocity.

        :param cell: grid cell
        :type cell: Cell
        """
        if self._tidal_range is None or self._tidal_period is None:
            msg = f'Tidal data is missing: tidal range = {self._tidal_range}; tidal period = {self._tidal_period}.'
            raise DataError(msg)

        self._current_velocity = self._estimate_wave_velocity(
            self._tidal_range, self._tidal_period, cell.water_depth
        )

    def _set_wave_velocity(self, cell):
        """Estimate wave velocity.

        :param cell: grid cell
        :type cell: Cell
        """
        if self._wave_height is None or self._wave_period is None:
            msg = f'Wave data is missing: ' \
                f'wave height = {self._wave_height}; wave period = {self._wave_period}.'
            raise DataError(msg)

        self._wave_velocity = self._estimate_wave_velocity(
            self._wave_height, self._wave_period, cell.water_depth
        )

    def _set_storm_wave_velocity(self, cell):
        """Estimate storm wave velocity.

        :param cell: grid cell
        :type cell: Cell
        """
        if self._storm_wave_height is None or self._storm_wave_period is None:
            msg = f'Storm data is missing: ' \
                f'wave height = {self._storm_wave_height}; wave period = {self._storm_wave_period}'
            raise DataError(msg)

        self._storm_wave_velocity = self._estimate_wave_velocity(
            self._storm_wave_height, self._storm_wave_period, cell.water_depth
        )

    def _initialise(self):
        """Initialise 0D-hydrodynamic model."""
        if not Grid.get_size() == 1:
            # message
            if Grid.get_size() > 1:
                LOG.warning('Grid is reset and will be set to a 0D-grid: (0, 0).')
            else:
                LOG.info('Grid is initiated as a 0D-grid: (0, 0).')

            # reset grid
            Grid.reset()

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

    _grid_from_hydrodynamic_model = True

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
