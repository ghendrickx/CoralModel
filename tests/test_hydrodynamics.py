import unittest

import numpy

from coral_model.hydrodynamics import Hydrodynamics


class TestHydrodynamics(unittest.TestCase):

    def test_init(self):
        modes = (None, 'Reef0D', 'Reef1D', 'Delft3D')

        for mode in modes:
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

    def test_input_check31(self):
        model = Hydrodynamics(mode='Delft3D')
        with self.assertRaises(ValueError) as context:
            model.input_check_interval('update_interval')

        self.assertTrue('update_interval undefined' in str(context.exception))

    def test_input_check32(self):
        model = Hydrodynamics(mode='Delft3D')
        with self.assertRaises(ValueError) as context:
            model.input_check_interval('update_interval_storm')

        self.assertTrue('update_interval_storm undefined' in str(context.exception))

    def test_input_check33(self):
        model = Hydrodynamics(mode='Delft3D')
        model.set_update_intervals(10, 20)

        intervals = ('update_interval', 'update_interval_storm')
        [model.input_check_interval(interval) for interval in intervals]

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

    # TODO: test set_coordinates Reef1D
    # TODO: test set_coordinates Delft3D

    def test_water_depth01(self):
        model = Hydrodynamics(mode=None)
        model.set_water_depth(10)
        answer = numpy.array([10])
        self.assertEqual(model.water_depth, answer)

    def test_water_depth02(self):
        model = Hydrodynamics(mode=None)
        model.set_water_depth((10,))
        answer = numpy.array([10])
        self.assertEqual(model.water_depth, answer)

    def test_water_depth03(self):
        model = Hydrodynamics(mode=None)
        model.set_water_depth([10])
        answer = numpy.array([10])
        self.assertEqual(model.water_depth, answer)

    def test_water_depth04(self):
        model = Hydrodynamics(mode=None)
        model.set_water_depth(numpy.array([10]))
        answer = numpy.array([10])
        self.assertEqual(model.water_depth, answer)

    def test_water_depth05(self):
        model = Hydrodynamics(mode=None)
        model.set_water_depth((10, 10))
        answer = numpy.array([10, 10])
        self.assertEqual(model.water_depth.all(), answer.all())

    def test_water_depth06(self):
        model = Hydrodynamics(mode=None)
        model.set_water_depth([10, 10])
        answer = numpy.array([10, 10])
        self.assertEqual(model.water_depth.all(), answer.all())

    def test_water_depth07(self):
        model = Hydrodynamics(mode=None)
        model.set_water_depth(numpy.array([10, 10]))
        answer = numpy.array([10, 10])
        self.assertEqual(model.water_depth.all(), answer.all())

    def test_water_depth11(self):
        model = Hydrodynamics(mode='Reef0D')
        model.set_water_depth(10)
        answer = numpy.array([10])
        self.assertEqual(model.water_depth, answer)

    def test_water_depth12(self):
        model = Hydrodynamics(mode='Reef0D')
        model.set_water_depth((10,))
        answer = numpy.array([10])
        self.assertEqual(model.water_depth, answer)

    def test_water_depth13(self):
        model = Hydrodynamics(mode='Reef0D')
        model.set_water_depth([10])
        answer = numpy.array([10])
        self.assertEqual(model.water_depth, answer)

    def test_water_depth14(self):
        model = Hydrodynamics(mode='Reef0D')
        model.set_water_depth(numpy.array([10]))
        answer = numpy.array([10])
        self.assertEqual(model.water_depth, answer)

    def test_water_depth15(self):
        model = Hydrodynamics(mode='Reef0D')
        model.set_water_depth((10, 10))
        answer = numpy.array([10, 10])
        self.assertEqual(model.water_depth.all(), answer.all())

    def test_water_depth16(self):
        model = Hydrodynamics(mode='Reef0D')
        model.set_water_depth([10, 10])
        answer = numpy.array([10, 10])
        self.assertEqual(model.water_depth.all(), answer.all())

    def test_water_depth17(self):
        model = Hydrodynamics(mode='Reef0D')
        model.set_water_depth(numpy.array([10, 10]))
        answer = numpy.array([10, 10])
        self.assertEqual(model.water_depth.all(), answer.all())
