"""
Grid settings.

Author: Gijs G. Hendrickx
"""
import logging

import numpy as np

from _v2.coral import _CoralCollection, CoralSpecies

LOG = logging.getLogger(__name__)


class _CellVariables:

    def __init__(self, capacity=None, water_depth=None, velocity=None, **kwargs):
        """
        :param capacity: carrying capacity of cell, defaults to None
        :param water_depth: water depth at cell, defaults to None
        :param velocity: depth-averaged flow velocity at cell, defaults to None
        :param kwargs: non-essential cell characteristics

        :type capacity: float, optional
        :type water_depth: float, optional
        :type velocity: float, optional
        """
        self.capacity = 1 if capacity is None else capacity
        self.water_depth = water_depth
        self.flow_velocity = velocity
        [setattr(self, key, value) for key, value in kwargs]


class Cell:
    _cells = dict()

    def __new__(cls, x, y, **kwargs):
        """
        :param x: x-coordinate
        :param y: y-coordinate
        :param kwargs: key-worded arguments: cell characteristics

        :type x: float
        :type y: float
        """
        # use predefined cell with same (x, y)-coordinates
        if (x, y) in cls._cells:
            return cls._cells[(x, y)]

        # create new cell
        cell = super().__new__(cls)
        cell._already_initiated = False
        cls._cells[(x, y)] = cell
        return cell

    def __init__(self, x, y, **kwargs):
        """
        :param x: x-coordinate
        :param y: y-coordinate
        :param kwargs: key-worded arguments: cell characteristics

        :type x: float
        :type y: float
        """
        if not self._already_initiated:
            self._x = x
            self._y = y
            self._coral_collection = _CoralCollection()
            self._variables = _CellVariables(**kwargs)
            self._already_initiated = True

    def __str__(self):
        """String-representation of Cell."""
        return f'Cell at {self.coordinates}'

    def __repr__(self):
        """Object-representation of Cell."""
        return f'Cell({self._x}, {self._y})'

    @property
    def x(self):
        """
        :return: x-coordinate
        :rtype: float
        """
        return self._x

    @property
    def y(self):
        """
        :return: y-coordinate
        :rtype: float
        """
        return self._y

    @property
    def coordinates(self):
        """
        :return: (x, y)-coordinates
        :rtype: tuple
        """
        return self._x, self._y

    @property
    def corals(self):
        """
        :return: coral collection in cell
        :rtype: list
        """
        return self._coral_collection.coral_list

    @property
    def coral_details(self):
        """
        :return: coral collection in cell
        :rtype: dict
        """
        return self._coral_collection.corals

    @corals.setter
    def corals(self, collection):
        """
        :param collection: coral collection
        :type collection: _CoralCollection

        :raises TypeError: if collection is not CoralCollection
        """
        if isinstance(collection, _CoralCollection):
            self._coral_collection = collection
        else:
            raise TypeError

    def reset_corals(self):
        """Reset CoralCollection."""
        self._coral_collection = _CoralCollection()

    @property
    def capacity(self):
        """
        :return: carrying capacity of cell
        :rtype: float
        """
        return self._variables.capacity

    @capacity.setter
    def capacity(self, carrying_capacity):
        """
        :param carrying_capacity: carrying capacity of cell
        :type carrying_capacity: float

        :raises ValueError: if value is not in range [0, 1]
        """
        if not 0 < carrying_capacity < 1:
            msg = f'Carrying capacity is a value in the range [0, 1]; {carrying_capacity} is given.'
            raise ValueError(msg)

        self._variables.capacity = carrying_capacity

    @property
    def living_cover(self):
        """
        :return: living coral cover, all coral species in grid cell included
        :rtype: float
        """
        return sum(coral.sum for coral in self.corals)

    @property
    def water_depth(self):
        """
        :return: water depth at cell location
        :rtype: float
        """
        return self._variables.water_depth

    @water_depth.setter
    def water_depth(self, depth):
        """
        :param depth: water depth at cell location
        :type depth: float
        """
        self._variables.water_depth = depth

    @property
    def flow_velocity(self):
        """
        :return: depth-averaged flow velocity
        :rtype: float
        """
        return self._variables.flow_velocity

    @flow_velocity.setter
    def flow_velocity(self, velocity):
        """
        :param velocity: depth-averaged flow velocity
        :type velocity: float
        """
        self._variables.flow_velocity = velocity

    def get_var(self, variable):
        """Get cell characteristic.

        :param variable: cell characteristic
        :type variable: str
        """
        if hasattr(self._variables, variable):
            return getattr(self._variables, variable)
        LOG.warning(f'{self} does not have the characteristic \"{variable}\".')


class Grid:
    _cells = set()

    def __init__(self, x=None, y=None):
        """
        :param x: x-coordinate(s), defaults to None
        :param y: y-coordinate(s), defaults to None

        :type x: float, int, iterable, optional
        :type y: float, int, iterable, optional
        """
        if not (x is None and y is None):
            self.grid_from_xy(0 if x is None else x, 0 if y is None else y)

    @classmethod
    def grid_from_xy(cls, x, y):
        """Create grid from x- and/or y-coordinates.

        :param x: x-coordinate(s)
        :param y: y-coordinate(s)

        :type x: float, int, iterable
        :type y: float, int, iterable
        """
        def size(_v):
            """
            :param _v: variable
            :return: variable length
            """
            try:
                return len(_v)
            except TypeError:
                return 1

        # x,y-sizes
        x_len = size(x)
        y_len = size(y)
        if (x_len > 1 and y_len > 1) and not x_len == y_len:
            raise ValueError

        # x,y as arrays
        x_array = x if x_len > 1 else [x] * y_len
        y_array = y if y_len > 1 else [y] * x_len

        # add cells
        [cls.add_cell(xx, yy) for xx, yy in zip(x_array, y_array)]

    @property
    def cells(self):
        """
        :return: cells included in the grid
        :rtype: set
        """
        return self.get_cells()

    @classmethod
    def get_cells(cls):
        """Get grid cells."""
        return cls._cells

    @property
    def number_of_cells(self):
        """
        :return: number of cells included in the grid
        :rtype: int
        """
        return self.get_size()

    @property
    def size(self):
        """
        :return: size of grid (i.e. number of cells)
        :rtype: int
        """
        return self.get_size()

    @classmethod
    def get_size(cls):
        """Get size of grid, i.e. number of cells."""
        return len(cls._cells)

    @staticmethod
    def _create_array(range_, spacing, edge='round'):
        """Create array of equally spaced coordinates.

        :param range_: range of the array
        :param spacing: spacing between coordinates
        :param edge: edge-case method, defaults to 'round'

        :type range_: iterable
        :type spacing: float
        :type edge: str, optional
        """
        different_max = False

        dist = max(range_) - min(range_)

        if dist % spacing:
            LOG.warning(
                f'Range cannot be equally subdivided. Edge-case method \"{edge}\" used:'
                f'\n\trange  \t:\t{range_}'
                f'\n\tspacing\t:\t{spacing}'
            )
            different_max = True

        edges = ('round', 'above', 'below')
        if edge == 'round':
            n = round(dist / spacing)
        elif edge == 'above':
            n = dist // spacing + (1 if different_max else 0)
        elif edge == 'below':
            n = dist // spacing
        else:
            msg = f'Unknown edge-case method: choose one of {edges}.'
            raise ValueError(msg)

        max_ = min(range_) + n * spacing
        if different_max:
            LOG.warning(f'Modified range\t:\t{min(range_), max_}')

        return np.linspace(min(range_), max_, int(n + 1))

    @classmethod
    def add_transect(cls, x_range, spacing, edge='round'):
        """Add transect of cells, where all y-coordinates are set equal to 0.

        :param x_range: range of x-coordinates
        :param spacing: spacing between cells
        :param edge: edge-case method, defaults to 'round'

        :type x_range: iterable
        :type spacing: float
        :type edge: str, optional
        """
        # create array of x-coordinates
        array = cls._create_array(x_range, spacing, edge)
        # create cells at x-coordinates
        [cls._cells.add(Cell(x, 0)) for x in array]

    @classmethod
    def add_square(cls, xy_range, spacing, edge='round'):
        """Add square grid of cells.

        :param xy_range: range of x- and y-coordinates
        :param spacing: spacing between cells
        :param edge: edge-case method, defaults to 'round'

        :type xy_range: iterable
        :type spacing: float
        :type edge: str, optional
        """
        cls.add_rectangle(xy_range, xy_range, spacing, edge)

    @classmethod
    def add_rectangle(cls, x_range, y_range, spacing, edge='round'):
        """Add rectangular grid of cells.

        :param x_range: range of x-coordinates
        :param y_range: range of y-coordinates
        :param spacing: spacing between cells
        :param edge: edge-case method, defaults to 'round'

        :type x_range: iterable
        :type y_range: iterable
        :type spacing: float
        :type edge: str, optional
        """
        # create arrays of x- and y-coordinates
        x_array = cls._create_array(x_range, spacing, edge)
        y_array = cls._create_array(y_range, spacing, edge)
        # create cells at (x,y)-coordinates
        [cls._cells.add(Cell(x, y)) for x in x_array for y in y_array]

    @classmethod
    def add_cell(cls, x, y, **kwargs):
        """Add individual cell to grid.

        :param x: x-coordinate
        :param y: y-coordinate
        :param kwargs: non-essential cell characteristics

        :type x: float
        :type y: float
        """
        cls._cells.add(Cell(x, y, **kwargs))

    @classmethod
    def reset(cls):
        """Reset grid."""
        cls._cells = set()

    @classmethod
    def reset_corals(cls):
        """Reset CoralCollections of all Cells."""
        if CoralSpecies.re_initiate():
            [cell.reset_corals() for cell in cls._cells]
