"""
Simulation framework.

Author: Gijs G. Hendrickx
"""
from _v2.biophysics import _BasicBiophysics, Light, Flow, Temperature, Photosynthesis, PopulationStates, \
    Calcification, Morphology, Dislodgement, Recruitment
from _v2.environment import Environment
from _v2.grid import Grid
from _v2.hydrodynamics import Hydrodynamics


class Simulation:
    """CoralModel simulation."""
    _grid = None
    _hydrodynamics = None
    _processes = [
        Light,
        Flow,
        Temperature,
        Photosynthesis,
        PopulationStates,
        Calcification,
        Morphology,
        Dislodgement,
        Recruitment,
    ]

    def __init__(self, hydrodynamics=None):
        """
        :param hydrodynamics: hydrodynamic model
        :type hydrodynamics: Hydrodynamics, str
        """
        self._hydrodynamics = self._get_hydrodynamics(hydrodynamics)

    @staticmethod
    def _get_hydrodynamics(hydrodynamics):
        """Get hydrodynamic model definition as an Hydrodynamics-object.

        :param hydrodynamics: hydrodynamic model
        :type hydrodynamics: Hydrodynamics, str, None

        :return: hydrodynamic model
        :rtype: Hydrodynamics
        """
        return hydrodynamics if isinstance(hydrodynamics, Hydrodynamics) else Hydrodynamics(hydrodynamics)

    @property
    def grid(self):
        """
        :return: model grid
        :rtype: Grid
        """
        # auto-set grid if none is defined
        if self._grid is None:
            self._extract_grid()

        # return grid
        return self._grid

    @classmethod
    def set_grid(cls, grid):
        """Set a manually-generated grid. If none is specified, the grid is extracted from the hydrodynamic model.

        :param grid: grid
        :type grid: Grid
        """
        # verify if grid-instance is of type Grid
        if not isinstance(grid, Grid):
            msg = f'Grid must be defined as a Grid-object; {type(grid)} provided.'
            raise TypeError(msg)

        # set grid-object
        cls._grid = grid

    def _extract_grid(self):
        """Extract grid from hydrodynamic model."""
        self.set_grid(self._hydrodynamics.grid)

    def set_processes(self, processes):
        """Set biophysical processes to include in the simulation.

        :param processes: list of processes
        :type processes: list[_BasicBiophysics]
        """
        if all(isinstance(process, _BasicBiophysics) for process in processes):
            self._processes = processes
        else:
            msg = f'Not all provided processes are of type \"_BasicBiophysics\": ' \
                f'{[type(process) for process in processes]}'
            raise TypeError(msg)

    def update(self, year):
        # update environmental conditions
        Environment.annual_update(year)

        # update biophysical processes
        for process in self._processes:
            process(self.grid)
