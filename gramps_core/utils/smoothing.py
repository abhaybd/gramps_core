"""
Modified from:
    https://github.com/personalrobotics/tycho_env/blob/main/tycho_env/smoother.py
    https://github.com/personalrobotics/tycho_env/blob/main/tycho_env/IIR.py
"""

from typing import Union, Optional
import numpy as np

class WindowSmoother(object):
    def __init__(self, dim, window_len, window='hanning'):
        self.window = \
                (np.ones(window_len, 'd') if window == 'flat'
                 else eval('np.'+window+'(window_len)'))
        self.w = np.array(self.window / self.window.sum()).reshape(1, -1)
        self.window_len = window_len
        self.data = np.zeros((window_len, dim))
        self.is_init = False

    def append(self, datapoint):
        if self.is_init:
            self.data[:-1] = self.data[1:]
            self.data[-1,:] = datapoint
        else:
            self.data[:] = datapoint
            self.is_init = True

    def get(self) -> np.ndarray:
        wl = self.window_len // 2
        wr = self.window_len - wl
        return (np.matmul(self.w[:,:wr], self.data[wl:])
              + np.matmul(self.w[:,wr:], self.data[-2:-wl-2:-1])).squeeze()

    def reset(self, new_window_len=None):
        self.is_init = False
        self.window_len = new_window_len or self.window_len

class IIRFilter(object):
    """
    Implements a multidimensional IIR filter.
    """
    def __init__(self, alpha: Union[np.ndarray, float]) -> None:
        """
        alpha - (n,) array of alpha gains or single float, in [0,1]. 1 is no smoothing, 0 is completely damped.
        """
        self._alpha = alpha
        self._x = None

    def append(self, x: Union[np.ndarray, float]) -> None:
        if self._x is None:
            self._x = x
        else:
            self._x = self._alpha * x + (1. - self._alpha) * self._x

    def get(self) -> Optional[Union[np.ndarray, float]]:
        """
        Returns None if no data is in the filter.
        """
        return self._x

    def reset(self) -> None:
        self._x = None
