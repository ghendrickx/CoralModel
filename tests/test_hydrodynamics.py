import unittest

import numpy

from coral_model.hydrodynamics import Hydrodynamics


class TestHydrodynamics(unittest.TestCase):

    def test_init(self):
        modes = (None, 'Reef0D', 'Reef1D', 'Delft3D')
        nir_modes = ('Reef1D', 'Delft3D')

        # implemented modes
        for mode in modes:
            if mode not in nir_modes:
                _ = Hydrodynamics(mode=mode)

        # not implemented modes
        for mode in nir_modes:
            with self.assertRaises(NotImplementedError):
                _ = Hydrodynamics(mode=mode)

    def test_input_check01(self):
        model = Hydrodynamics(mode=None)
        with self.assertRaises(ValueError):
            model.input_check()

    def test_input_check02(self):
        model = Hydrodynamics(mode=None)
        model.set_coordinates((0, 0))
        with self.assertRaises(ValueError) as context:
            model.input_check()

        self.assertTrue('Water depth has to be provided' in str(context.exception))

    def test_input_check03(self):
        model = Hydrodynamics(mode=None)
        model.set_water_depth(10)
        with self.assertRaises(ValueError) as context:
            model.input_check()

        self.assertTrue('(x,y)-coordinates have to be provided' in str(context.exception))

    def test_input_check04(self):
        model = Hydrodynamics(mode=None)
        model.set_coordinates((0, 0))
        model.set_water_depth(10)
        model.input_check()

    def test_input_check11(self):
        model = Hydrodynamics(mode='Reef0D')
        with self.assertRaises(ValueError):
            model.input_check()

    def test_input_check12(self):
        model = Hydrodynamics(mode='Reef0D')
        model.set_coordinates((0, 0))
        with self.assertRaises(ValueError) as context:
            model.input_check()

        self.assertTrue('Water depth has to be provided' in str(context.exception))

    def test_input_check13(self):
        model = Hydrodynamics(mode='Reef0D')
        model.set_water_depth(10)
        model.input_check()

    def test_input_check14(self):
        model = Hydrodynamics(mode='Reef0D')
        model.set_coordinates((0, 0))
        model.set_water_depth(10)
        model.input_check()

    # TODO: test input_check Reef1D
    # TODO: test input_check Delft3D

    def test_coordinates01(self):
        model = Hydrodynamics(mode=None)
        model.set_coordinates((0, 0))
        answer = numpy.array([[0, 0]])
        self.assertEqual(model.xy_coordinates.all(), answer.all())

    def test_coordinates02(self):
        model = Hydrodynamics(mode=None)
        model.set_coordinates([0, 0])
        answer = numpy.array([[0, 0]])
        self.assertEqual(model.xy_coordinates.all(), answer.all())

    def test_coordinates03(self):
        model = Hydrodynamics(mode=None)
        model.set_coordinates(numpy.array([0, 0]))
        answer = numpy.array([[0, 0]])
        self.assertEqual(model.xy_coordinates.all(), answer.all())

    def test_coordinates04(self):
        model = Hydrodynamics(mode=None)
        model.set_coordinates([(0, 0), (0, 1)])
        answer = numpy.array([[0, 0], [0, 1]])
        self.assertEqual(model.xy_coordinates.all(), answer.all())

    def test_coordinates05(self):
        model = Hydrodynamics(mode=None)
        model.set_coordinates(((0, 0), (0, 1)))
        answer = numpy.array([[0, 0], [0, 1]])
        self.assertEqual(model.xy_coordinates.all(), answer.all())

    def test_coordinates06(self):
        model = Hydrodynamics(mode=None)
        model.set_coordinates(numpy.array([[0, 0], [0, 1]]))
        answer = numpy.array([[0, 0], [0, 1]])
        self.assertEqual(model.xy_coordinates.all(), answer.all())

    def test_coordinates11(self):
        model = Hydrodynamics(mode='Reef0D')
        model.set_coordinates((0, 0))
        answer = numpy.array([[0, 0]])
        self.assertEqual(model.xy_coordinates.all(), answer.all())

    def test_coordinates12(self):
        model = Hydrodynamics(mode='Reef0D')
        model.set_coordinates([0, 0])
        answer = numpy.array([[0, 0]])
        self.assertEqual(model.xy_coordinates.all(), answer.all())

    def test_coordinates13(self):
        model = Hydrodynamics(mode='Reef0D')
        model.set_coordinates(numpy.array([0, 0]))
        answer = numpy.array([[0, 0]])
        self.assertEqual(model.xy_coordinates.all(), answer.all())

    def test_coordinates14(self):
        model = Hydrodynamics(mode='Reef0D')
        model.set_coordinates([(0, 0), (0, 1)])
        answer = numpy.array([[0, 0], [0, 1]])
        self.assertEqual(model.xy_coordinates.all(), answer.all())

    def test_coordinates15(self):
        model = Hydrodynamics(mode='Reef0D')
        model.set_coordinates(((0, 0), (0, 1)))
        answer = numpy.array([[0, 0], [0, 1]])
        self.assertEqual(model.xy_coordinates.all(), answer.all())

    def test_coordinates16(self):
        model = Hydrodynamics(mode='Reef0D')
        model.set_coordinates(numpy.array([[0, 0], [0, 1]]))
        answer = numpy.array([[0, 0], [0, 1]])
        self.assertEqual(model.xy_coordinates.all(), answer.all())
