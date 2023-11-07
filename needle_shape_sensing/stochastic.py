import copy
from typing import (
    Tuple,
    Union,
)

import torch
import numpy as np
import scipy as sp
import cupy as cp
from numba import (
    jit,
    cuda as nmb_cuda,
)

from needle_shape_sensing.sensorized_needles import (
    Needle,
    FBGNeedle,
)

class CurvatureDistribution:
    """
        data: (N, M, K, L) Array
            N: index of arclength
            M, K, L: indices of the w_1, w_2, w_3 
    
    """
    def __init__(self, data: Union[np.ndarray, cp.ndarray, torch.Tensor], ds: float = 0.5):
        
        self.data = data
        self.ds   = ds
        self.s    = np.arange(
            0, 
            (self.data.shape[0] + 1)/self.ds,
            ds
        )

    # __init__

    # ================================ MAGIC METHODS =====================================================

    def __div__(self, other):
        new = self.copy()
        if isinstance(other, CurvatureDistribution):
            new.data = self.data / other.data
            return new
        
        # if
        new.data = self.data / other
        return new
    
    # __div__

    def __eq__(self, other):
        new = self.copy()
        if isinstance(other, CurvatureDistribution):
            new.data = self.data == other.data
            return new
        
        # if
        new.data = self.data == other
        return new
    
    # __eq__

    def __ge__(self, other):
        new = self.copy()
        if isinstance(other, CurvatureDistribution):
            new.data = self.data >= other.data
            return new
        
        # if
        new.data = self.data >= other
        return new
    
    # __ge__

    def __gt__(self, other):
        new = self.copy()
        if isinstance(other, CurvatureDistribution):
            new.data = self.data > other.data
            return new
        
        # if
        new.data = self.data > other
        return new
    
    # __gt__

    def __le__(self, other):
        new = self.copy()
        if isinstance(other, CurvatureDistribution):
            new.data = self.data <= other.data
            return new
        
        # if
        new.data = self.data <= other
        return new
    
    # __le__

    def __len__(self):
        return self.data.shape[0]
    
    # __len__

    def __lt__(self, other):
        new = self.copy()
        if isinstance(other, CurvatureDistribution):
            new.data = self.data < other.data
            return new
        
        # if
        new.data = self.data < other
        return new
    
    # __lt__

    def __mul__(self, other):
        new = self.copy()
        if isinstance(other, CurvatureDistribution):
            new.data = self.data * other.data
            return new
        
        # if
        new.data = self.data * other
        return new
    
    # __mul__

    def __ne__(self, other):
        new = self.copy()
        if isinstance(other, CurvatureDistribution):
            new.data = self.data != other.data
            return new
        
        # if
        new.data = self.data != other
        return new
    
    # __ne__

    def __pow__(self, power: Union[int, float]):
        new = self.copy()
        new.data = self.data ** power
        return new
    
    # __pow__

    def __truediv__(self, other):
        return self.__div__(other)
    
    # __truediv__

    # ================================ FUNCTIONS =====================================================

    def _normalize(self, inplace=True):
        data               = self.data.copy()
        mask_nonzero       = data.sum(axis=0) > 0
        data[mask_nonzero] = data[mask_nonzero] / self.data[mask_nonzero].sum(axis=0, keepdims=True)
        if inplace:
            self.data = data
            return self
        
        # if
        
        return CurvatureDistribution(
            data=data,
            ds=self.ds,
        )

    # _normalize

    def copy(self):
        return copy.deepcopy(self)
    
    # copy

    @classmethod
    def zeros(cls, N_s: int, N_w: Tuple[int, int, int], ds: float = 0.5, cuda: bool = False):
        data = np.zeros((N_s, *N_w), dtype=np.float64)
        if cuda:
            data = cp.asarray(data)

        return cls(
            data=data,
            ds=ds,
        )
    # zeros

    @classmethod
    def uniform_distribution(cls, N_s: int, N_w: Tuple[int, int, int], ds: float = 0.5, cuda: bool = False):
        data = np.ones((N_s, *N_w), dtype=np.float64)
        if cuda:
            data = cp.asarray(data)

        obj = cls(
            data=data,
            ds=ds,
        )._normalize(inplace=True)

        return obj
    
    # uniform_distribution

# class: CurvatureDistribution

class StochasticShapeModel:
    def __init__(
        self,
        needle: Needle,
        ds: float = 0.5,
        dw: float = 0.002,
        w_bounds: Tuple[float, float] = (-0.05, 0.05),
        use_cuda: bool = False,
    ):
        self.needle   = needle
        self.ds       = ds
        self.dw       = dw
        assert w_bounds[0] < w_bounds[1], f"Lower bound must be first and < upper bound. {w_bounds[0]} is not < {w_bounds[1]}"
        self.w_bounds = w_bounds
        self._cuda    = use_cuda

        # algorithm parameters
        self.s                     : np.ndarray            = None
        self.curvature_grid        : np.ndarray            = None
        self.curvature_distribution: CurvatureDistribution = None


    # __init__

    def init_probability(self):
        self.s              = np.arange(0, self.needle.length + self.ds, self.ds, dtype=np.float64)
        self.curvature_grid = np.stack(
            np.meshgrid(
                np.arange(self.w_bounds[0], self.w_bounds[1] + self.dw, self.dw, dtype=np.float64), # x-curvature
                np.arange(self.w_bounds[0], self.w_bounds[1] + self.dw, self.dw, dtype=np.float64), # y-curvature
                np.arange(self.w_bounds[0], self.w_bounds[1] + self.dw, self.dw, dtype=np.float64), # z-curvature
            ),
            axis=-1
        ) # (N_wx, N_wy, N_wz, 3)

        self.curvature_distribution = CurvatureDistribution.uniform_distribution(
            N_s=self.s.shape[0],
            N_w=self.curvature_grid.shape[0:3],
            ds=self.ds,
            cuda=self._cuda,
        )

    # init_probability