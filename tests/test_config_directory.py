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
        self.assertEqual(folder.__repr__(), os.getcwd())
        self.assertEqual(folder.__str__(), os.getcwd())

    def test_home_dir1(self):
        folder = DirConfig(r'folder1\folder2', create_dir=False)
        answer = f'{os.getcwd()}\\folder1\\folder2'
        self.assertEqual(folder.__str__(), answer)

    def test_home_dir2(self):
        folder = DirConfig(r'C:\folder', create_dir=False)
        answer = r'C:\folder'
        self.assertEqual(folder.__str__(), answer)

    def test_home_dir3(self):
        folder = DirConfig(['folder1', 'folder2'], create_dir=False)
        answer = f'{os.getcwd()}\\folder1\\folder2'
        self.assertEqual(folder.__str__(), answer)

    def test_home_dir4(self):
        folder = DirConfig(('C:', 'folder'), create_dir=False)
        answer = r'C:\folder'
        self.assertEqual(folder.__str__(), answer)

    def test_home_dir5(self):
        folder = DirConfig([r'C:\folder1', 'folder2'], create_dir=False)
        answer = r'C:\folder1\folder2'
        self.assertEqual(folder.__str__(), answer)

    def test_home_dir6(self):
        folder = DirConfig(DirConfig(r'C:\folder', create_dir=False), create_dir=False)
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
        folder = DirConfig('folder1', create_dir=False).config_dir('folder2')
        answer = f'{os.getcwd()}\\folder1\\folder2'
        self.assertEqual(folder, answer)

    def test_home_config_dir2(self):
        folder = DirConfig(r'folder1\folder2', create_dir=False).config_dir(r'folder3\folder4')
        answer = f'{os.getcwd()}\\folder1\\folder2\\folder3\\folder4'
        self.assertEqual(folder, answer)

    def test_home_config_dir3(self):
        folder = DirConfig('C:', create_dir=False).config_dir('folder')
        answer = r'C:\folder'
        self.assertEqual(folder, answer)

    def test_home_config_dir4(self):
        folder = DirConfig(['C:', 'folder1'], create_dir=False).config_dir('folder2')
        answer = r'C:\folder1\folder2'
        self.assertEqual(folder, answer)

    def test_home_config_dir5(self):
        folder = DirConfig(DirConfig(r'C:', create_dir=False), create_dir=False).config_dir('folder')
        answer = r'C:\folder'
        self.assertEqual(folder, answer)
