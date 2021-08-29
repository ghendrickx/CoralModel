class Hydrodynamics:

    def __init__(self):
        self._current_velocity = None
        self._wave_velocity = None
        self._wave_period = None

    def update(self):
        # TODO: Update hydrodynamic conditions
        pass

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
        :return: wave flow velocity
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
