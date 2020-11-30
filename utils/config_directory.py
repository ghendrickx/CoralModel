"""
Configuration of directories to ensure absolute directories

@author: Gijs G. Hendrickx
"""
import os


class DirConfig:

    __base_dirs = ('C:', )

    def __init__(self, home_dir=None):
        """
        :param home_dir: home directory
        :type home_dir: list, tuple, str
        """
        self.__home = home_dir

    def __repr__(self):
        """Representation of DirConfig."""
        return f'DirConfig(home_dir={self.__home})'

    def __str__(self):
        """String-representation."""
        return self._list2str(self.__home_dir)

    @property
    def __sep(self):
        """Folder separator."""
        return os.sep

    @property
    def __current_dir(self):
        """Current directory.

        :rtype: list
        """
        return self._as_list(os.getcwd())

    @property
    def __home_dir(self):
        """Absolute home directory, set to current directory if no absolute directory is provided.

        :rtype: list
        """
        # TODO: Ensure this to be a folder, and not a file
        if self.__home is None:
            return self.__current_dir

        list_dir = self._as_list(self.__home)
        return self._dir2abs(list_dir)

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
        if isinstance(folder, str):
            return self._str2list(folder)

        elif isinstance(folder, (list, tuple)):
            list_dir = []
            for i in folder:
                list_dir.extend(self._str2list(i))
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
        return self.__sep.join(list_dir)

    def _dir2abs(self, folder):
        """Translate directory to absolute directory.

        :param folder: directory to be converted
        :type folder: list

        :return: absolute directory
        :rtype: list
        """
        if folder[0] in self.__base_dirs:
            return folder
        return [*self.__current_dir, *folder]

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

    def config_dir(self, folder):
        """Configure directory.

        :param folder: directory to be converted
        :type folder: list, tuple, str

        :return: absolute, configured directory
        :rtype: str
        """
        list_dir = self._as_list(folder)
        if self._is_abs_dir(list_dir):
            return self._list2str(list_dir)
        return self._list2str([*self.__home_dir, *list_dir])

    def create_dir(self, folder):
        """Create directory, if non-existing.

        :param folder: directory to be created
        :type folder: list, tuple, str
        """
        folder = self.config_dir(folder)
        if not os.path.exists(folder):
            os.makedirs(folder)
