from _v2.hydrodynamics import Hydrodynamics


class Simulation:
    """CoralModel simulation."""

    _grid = None
    _hydrodynamics = None

    def __init__(self, hydrodynamics=None):
        """
        :param hydrodynamics: hydrodynamic model
        :type hydrodynamics: Hydrodynamics, str
        """
        self._hydrodynamics = hydrodynamics if isinstance(hydrodynamics, Hydrodynamics) else Hydrodynamics(hydrodynamics)

    @property
    def grid(self):
        """
        :return: model grid
        :rtype: Grid
        """
        if self._grid is None:
            self._extract_grid()
        return self._grid

    @classmethod
    def set_grid(cls, grid):
        """Set a manually-generated grid. If none is specified, the grid is extracted from the hydrodynamic model.

        :param grid: grid
        :type grid: Grid
        """
        cls._grid = grid

    def _extract_grid(self):
        """Extract grid from hydrodynamic model."""
        self.set_grid(self._hydrodynamics.grid)
