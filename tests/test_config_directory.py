"""
Tests for the DirConfig-class in utils/config_directory.py

@author: Gijs G. Hendrickx
"""
import os
import unittest

from utils.config_directory import DirConfig


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
