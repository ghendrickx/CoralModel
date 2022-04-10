"""
Configuration of directories to ensure absolute directories

@author: Gijs G. Hendrickx
"""
import logging
import os

LOG = logging.getLogger(__name__)


class DirConfig:
    __base_dirs = ('', 'C:', 'D:', 'P:', 'U:')

    def __init__(self, *home_dir, create_dir=True):
        """
        :param home_dir: home directory, defaults to None
        :param create_dir: automatically create home directory if non-existent, defaults to True

        :type home_dir: str, tuple, list, DirConfig
        :type create_dir: bool, optional
        """
        self._home = self._unpack(home_dir)
        self.create_dir() if create_dir else None

    def __repr__(self):
        """Representation of DirConfig."""
        return self._list2str(self._home_dir)

    @property
    def _sep(self):
        """Folder separator."""
        return os.sep

    @property
    def _current_dir(self):
        """Current directory.

        :rtype: list
        """
        return self._as_list(os.getcwd())

    @property
    def _home_dir(self):
        """Absolute home directory, set to current directory if no absolute directory is provided.

        :rtype: list
        """
        if not self._home:
            return self._current_dir

        list_dir = self._as_list(
            os.path.dirname(self._home) if os.path.splitext(self._home)[1] else self._home
        )
        return self._dir2abs(list_dir)

    def _unpack(self, directory):
        """Unpack defined directory, which may be a mix of str, tuple, list, and/or DirConfig.

        :param directory: defined directory
        :type directory: tuple, list

        :return: directory
        :rtype: str
        """
        out = []
        for item in directory:
            if isinstance(item, type(self)):
                out.append(str(item))
            elif isinstance(item, str):
                out.append(str(item))
            elif isinstance(item, (tuple, list)):
                out.append(self._unpack(item))
        return self._list2str(out)

    @staticmethod
    def _str2list(str_dir):
        """Translate string- to list-directory.

        :param str_dir: string-based directory
        :type str_dir: str

        :return: list-based directory
        :rtype: list
        """
        return str_dir.replace('/', '\\').split('\\')

    def _as_list(self, folder):
        """Ensure directory to be a list.

        :param folder: directory to be checked
        :type folder: str, list, tuple

        :return: list-based directory
        :rtype: list
        """
        if isinstance(folder, (str, DirConfig)):
            return self._str2list(str(folder))

        elif isinstance(folder, (list, tuple)):
            list_dir = []
            [list_dir.extend(self._str2list(i)) for i in folder]
            return list_dir

        else:
            msg = f'Directory must be str, list, or tuple; {type(folder)} is given.'
            raise TypeError(msg)

    def _list2str(self, list_dir):
        """Translate list- to string-directory.

        :param list_dir: list-based directory
        :type list_dir: list

        :return: string-based directory
        :rtype: str
        """
        return self._sep.join(list_dir)

    def _dir2abs(self, folder):
        """Translate directory to absolute directory.

        :param folder: directory to be converted
        :type folder: list

        :return: absolute directory
        :rtype: list
        """
        if folder[0] in self.__base_dirs:
            return folder
        return [*self._current_dir, *folder]

    def _is_abs_dir(self, folder):
        """Verify if directory is an absolute directory.

        :param folder: directory to be verified
        :type folder: list

        :return: directory is an absolute directory, or not
        :rtype: bool
        """
        if folder[0] in self.__base_dirs:
            return True
        return False

    def config_dir(self, *folder, relative_dir=False):
        """Configure directory.

        :param folder: directory to be converted
        :param relative_dir: directory as relative directory, defaults to True

        :type folder: list, tuple, str
        :type relative_dir: bool, optional

        :return: (absolute) configured directory
        :rtype: str
        """
        list_dir = self._as_list(self._unpack(folder))
        if self._is_abs_dir(list_dir) or relative_dir:
            return self._list2str(list_dir)
        return self._list2str([*self._home_dir, *list_dir])

    def create_dir(self, folder=None):
        """Configure and create directory, if non-existing.

        :param folder: directory to be created
        :type folder: list, tuple, str
        """
        folder = self.__str__() if folder is None else self.config_dir(folder)
        if not os.path.exists(folder):
            os.makedirs(folder)
            LOG.info(f'Directory created: {folder}')
        return folder

    def delete_file(self, file_name):
        """Delete file, if existing.

        :param file_name: file name
        :type file_name: list, tuple, str
        """
        if self.existence_file(file_name):
            os.remove(self.config_dir(file_name))
            LOG.info(f'File deleted: {self.config_dir(file_name)}')

    def existence_file(self, file_name):
        """Verify if file exists.

        :param file_name: file name
        :type file_name: list, tuple, str

        :rtype: bool
        """
        file = self.config_dir(file_name)
        if os.path.exists(file):
            LOG.info(f'File exists: {file}')
            return True
        LOG.warning(f'File does not exist: {file}')
        return False
