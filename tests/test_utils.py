import os
import unittest

import numpy

from coral_model.utils import SpaceTime, DataReshape, DirConfig


class TestSpaceTime(unittest.TestCase):

    def test_default(self):
        spacetime = SpaceTime().spacetime
        self.assertEqual(len(spacetime), 2)
        self.assertIsInstance(spacetime, tuple)

    def test_default_input(self):
        spacetime = SpaceTime(None).spacetime
        self.assertEqual(len(spacetime), 2)
        self.assertIsInstance(spacetime, tuple)

    def test_global_raise_type_error(self):
        with self.assertRaises(TypeError):
            SpaceTime(int(1))
        with self.assertRaises(TypeError):
            SpaceTime(float(1))
        with self.assertRaises(TypeError):
            SpaceTime(str(1))

    def test_global_not_raise_type_error(self):
        SpaceTime((1, 1))
        SpaceTime([1, 1])

    def test_size_error(self):
        with self.assertRaises(ValueError):
            SpaceTime((1,))
        with self.assertRaises(ValueError):
            SpaceTime((1, 1, 1))

    def test_local_raise_type_error(self):
        with self.assertRaises(TypeError):
            SpaceTime((float(1), 1))

    def test_return_type(self):
        self.assertIsInstance(SpaceTime((1, 1)).spacetime, tuple)
        self.assertIsInstance(SpaceTime([1, 1]).spacetime, tuple)


# noinspection PyTypeChecker
class TestDataReshape(unittest.TestCase):

    def test_default_spacetime(self):
        reshape = DataReshape()
        spacetime = SpaceTime().spacetime
        self.assertIsInstance(reshape.spacetime, tuple)
        self.assertEqual(reshape.spacetime, spacetime)

    def test_default_space(self):
        reshape = DataReshape()
        spacetime = SpaceTime().spacetime
        self.assertIsInstance(reshape.space, int)
        self.assertEqual(reshape.space, spacetime[0])

    def test_default_time(self):
        reshape = DataReshape()
        spacetime = SpaceTime().spacetime
        self.assertIsInstance(reshape.time, int)
        self.assertEqual(reshape.time, spacetime[1])

    def test_set_spacetime_raise_type_error(self):
        with self.assertRaises(TypeError):
            DataReshape(int(1))
        with self.assertRaises(TypeError):
            DataReshape(float(1))
        with self.assertRaises(TypeError):
            DataReshape(str(1))

    def test_set_spacetime_not_raise_error(self):
        DataReshape((1, 1))
        DataReshape([1, 1])

    def test_variable2array(self):
        self.assertIsInstance(DataReshape.variable2array(float(1)), numpy.ndarray)
        self.assertIsInstance(DataReshape.variable2array(int(1)), numpy.ndarray)
        with self.assertRaises(NotImplementedError):
            DataReshape.variable2array(str(1))
        self.assertIsInstance(DataReshape.variable2array((1, 1)), numpy.ndarray)
        self.assertIsInstance(DataReshape.variable2array([1, 1]), numpy.ndarray)

    def test_variable2matrix_shape_space1(self):
        reshape = DataReshape((4, 1))
        var = [0, 1, 2, 4]
        matrix = reshape.variable2matrix(var, 'space')
        self.assertEqual(matrix.shape, (4, 1))

    def test_variable2matrix_shape_space2(self):
        reshape = DataReshape((1, 5))
        var = 0
        matrix = reshape.variable2matrix(var, 'space')
        self.assertEqual(matrix.shape, (1, 5))

    def test_variable2matrix_shape_space3(self):
        reshape = DataReshape((4, 5))
        var = [0, 1, 2, 4]
        matrix = reshape.variable2matrix(var, 'space')
        self.assertEqual(matrix.shape, (4, 5))

    def test_variable2matrix_shape_time1(self):
        reshape = DataReshape((4, 1))
        var = 0
        matrix = reshape.variable2matrix(var, 'time')
        self.assertEqual(matrix.shape, (4, 1))

    def test_variable2matrix_shape_time2(self):
        reshape = DataReshape((1, 5))
        var = [0, 1, 2, 4, 8]
        matrix = reshape.variable2matrix(var, 'time')
        self.assertEqual(matrix.shape, (1, 5))

    def test_variable2matrix_shape_time3(self):
        reshape = DataReshape((4, 5))
        var = [0, 1, 2, 4, 8]
        matrix = reshape.variable2matrix(var, 'time')
        self.assertEqual(matrix.shape, (4, 5))

    def test_variable2matrix_value_space(self):
        reshape = DataReshape((4, 5))
        var = [0, 1, 2, 4]
        matrix = reshape.variable2matrix(var, 'space')
        answer = numpy.array([
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [4, 4, 4, 4, 4]
        ])
        self.assertListEqual(matrix.tolist(), answer.tolist())

    def test_variable2matrix_value_time(self):
        reshape = DataReshape((4, 5))
        var = [0, 1, 2, 4, 8]
        matrix = reshape.variable2matrix(var, 'time')
        answer = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8]
        ])
        self.assertListEqual(matrix.tolist(), answer.tolist())

    def test_raise_error_space(self):
        reshape = DataReshape((4, 5))
        var = [0, 1, 2, 4, 8]
        with self.assertRaises(ValueError):
            reshape.variable2matrix(var, 'space')

    def test_raise_error_time(self):
        reshape = DataReshape((4, 5))
        var = [0, 1, 2, 4]
        with self.assertRaises(ValueError):
            reshape.variable2matrix(var, 'time')

    def test_matrix2array_space_last(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'space', None)
        answer = [8, 8, 8, 8]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_space_mean(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'space', 'mean')
        answer = [3, 3, 3, 3]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_space_max(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'space', 'max')
        answer = [8, 8, 8, 8]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_space_min(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'space', 'min')
        answer = [0, 0, 0, 0]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_space_sum(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'space', 'sum')
        answer = [15, 15, 15, 15]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_time_last(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'time', None)
        answer = [0, 1, 2, 4, 8]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_time_mean(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'time', 'mean')
        answer = [0, 1, 2, 4, 8]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_time_max(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'time', 'max')
        answer = [0, 1, 2, 4, 8]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_time_min(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'time', 'min')
        answer = [0, 1, 2, 4, 8]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)

    def test_matrix2array_time_sum(self):
        reshape = DataReshape((4, 5))
        var = numpy.array([
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
            [0, 1, 2, 4, 8],
        ])
        result = reshape.matrix2array(var, 'time', 'sum')
        answer = [0, 4, 8, 16, 32]
        for i, val in enumerate(answer):
            self.assertEqual(result[i], val)


class TestDirConfig(unittest.TestCase):

    def test_default(self):
        folder = DirConfig()
        self.assertEqual(folder.__repr__(), 'DirConfig(home_dir=None)')
        self.assertEqual(folder.__str__(), os.getcwd())

    def test_home_dir1(self):
        folder = DirConfig(home_dir=r'folder1\folder2')
        answer = f'{os.getcwd()}\\folder1\\folder2'
        self.assertEqual(folder.__str__(), answer)

    def test_home_dir2(self):
        folder = DirConfig(home_dir=r'C:\folder')
        answer = r'C:\folder'
        self.assertEqual(folder.__str__(), answer)

    def test_home_dir3(self):
        folder = DirConfig(home_dir=['folder1', 'folder2'])
        answer = f'{os.getcwd()}\\folder1\\folder2'
        self.assertEqual(folder.__str__(), answer)

    def test_home_dir4(self):
        folder = DirConfig(home_dir=('C:', 'folder'))
        answer = r'C:\folder'
        self.assertEqual(folder.__str__(), answer)

    def test_home_dir5(self):
        folder = DirConfig(home_dir=[r'C:\folder1', 'folder2'])
        answer = r'C:\folder1\folder2'
        self.assertEqual(folder.__str__(), answer)

    def test_home_dir6(self):
        folder = DirConfig(home_dir=DirConfig(r'C:\folder'))
        answer = r'C:\folder'
        self.assertEqual(folder.__str__(), answer)

    def test_config_dir1(self):
        folder = DirConfig().config_dir(r'folder1\folder2')
        answer = f'{os.getcwd()}\\folder1\\folder2'
        self.assertEqual(folder, answer)

    def test_config_dir2(self):
        folder = DirConfig().config_dir(r'C:\folder')
        answer = r'C:\folder'
        self.assertEqual(folder, answer)

    def test_config_dir3(self):
        folder = DirConfig().config_dir(['folder1', 'folder2'])
        answer = f'{os.getcwd()}\\folder1\\folder2'
        self.assertEqual(folder, answer)

    def test_config_dir4(self):
        folder = DirConfig().config_dir(('C:', 'folder'))
        answer = r'C:\folder'
        self.assertEqual(folder, answer)

    def test_config_dir5(self):
        folder = DirConfig().config_dir((r'C:\folder1', 'folder2'))
        answer = r'C:\folder1\folder2'
        self.assertEqual(folder, answer)

    def test_home_config_dir1(self):
        folder = DirConfig(home_dir='folder1').config_dir('folder2')
        answer = f'{os.getcwd()}\\folder1\\folder2'
        self.assertEqual(folder, answer)

    def test_home_config_dir2(self):
        folder = DirConfig(home_dir=r'folder1\folder2').config_dir(r'folder3\folder4')
        answer = f'{os.getcwd()}\\folder1\\folder2\\folder3\\folder4'
        self.assertEqual(folder, answer)

    def test_home_config_dir3(self):
        folder = DirConfig(home_dir='C:').config_dir('folder')
        answer = r'C:\folder'
        self.assertEqual(folder, answer)

    def test_home_config_dir4(self):
        folder = DirConfig(home_dir=['C:', 'folder1']).config_dir('folder2')
        answer = r'C:\folder1\folder2'
        self.assertEqual(folder, answer)

    def test_home_config_dir5(self):
        folder = DirConfig(home_dir=DirConfig(r'C:')).config_dir('folder')
        answer = r'C:\folder'
        self.assertEqual(folder, answer)


# TODO: test for "time_series_year"-method

# TODO: test for "coral_only_function"-method


if __name__ == '__main__':
    unittest.main()
