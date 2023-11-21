from abc import (
    ABC,
    abstractmethod,
    abstractclassmethod,
)
import copy
from typing import (
    Dict,
    Tuple,
    Union,
)

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
from needle_shape_sensing.intrinsics import (
    ShapeModelParameters,
)

class CurvatureDistribution:
    """
        data: (N, M, K, L) Array
            N: index of arclength
            M, K, L: indices of the w_1, w_2, w_3

    """
    def __init__(
            self,
            data: Union[np.ndarray, cp.ndarray],
            w_bounds: Tuple[float, float] = (-0.05, 0.05),
            dw: float = 0.001,
            ds: float = 0.5,
        ):
        self.data = data
        self.ds   = ds
        self.s    = np.arange(
            0,
            (self.data.shape[0] + 1) * self.ds,
            ds,
            dtype=np.float64,
        )

        self.dw = dw # FIXME: calculate dw
        assert w_bounds[0] < w_bounds[1], f"Lower bound must be first and < upper bound. {w_bounds[0]} is not < {w_bounds[1]}"
        self.curvature_grid = np.stack( # (3, N_1, N_2, N_3)
            np.meshgrid(
                np.arange(w_bounds[0], w_bounds[1] + self.dw, self.dw, dtype=np.float64), # x-curvature
                np.arange(w_bounds[0], w_bounds[1] + self.dw, self.dw, dtype=np.float64), # y-curvature
                np.arange(w_bounds[0], w_bounds[1] + self.dw, self.dw, dtype=np.float64), # z-curvature
            ),
            axis=0,
        )

        assert self.curvature_grid.shape[1:] == self.data.shape[1:]

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

    def _normalize_slice(self, idx: int, inplace=False):
        """ Normalize a slice of the term and return the slice probability distribution """
        data_slice: np.ndarray = self.data[idx].copy()
        norm                   = data_slice.sum()
        if norm > 0:
            data_slice /= norm

        if inplace:
            self.data[idx] = data_slice

        return data_slice

    # _normalize_slice

    def bayesian_fusion(self, other, inplace=False):
        if inplace:
            fused = self

        fused: CurvatureDistribution = self * other
        fused._normalize(inplace = True)

        return fused

    # bayesian_fusion

    def bayesian_fusion_slice(self, prob_slice: Union[np.ndarray, cp.ndarray], s_index: int, inplace=False):
        fused = self
        if not inplace:
            fused = self.copy()

        fused.data[s_index] *= prob_slice

        fused._normalize(inplace=True)

        return fused

    # bayesian_fusion_slice

    def copy(self):
        return copy.deepcopy(self)

    # copy

    # ================================ CLASS FUNCTIONS ===============================================

    @classmethod
    def dirac_delta_init(
        cls,
        w_init  : np.ndarray,
        N_s     : int,
        w_bounds: Tuple[float, float],
        dw      : float,
        ds      : float = 0.5,
        cuda    : bool  = False,
    ):
        dist = cls.uniform_distribution(
            N_s,
            w_bounds=w_bounds,
            dw=dw,
            ds=ds,
            cuda=cuda,
        )

        # find the indices where w_init is
        diff = np.linalg.norm(dist.curvature_grid - w_init[:, np.newaxis, np.newaxis, np.newaxis], axis=0)
        idx  = np.argmin(
            diff,
            axis=None,
        )
        idx_1, idx_2, idx_3 = np.unravel_index(idx, diff.shape)

        # set the distribution to the dirac delta function
        dist.data[:]                      = 0
        dist.data[:, idx_1, idx_2, idx_3] = 1

        dist._normalize(inplace=True)

        return dist

    # dirac_delta_init

    @classmethod
    def uniform_distribution(
        cls,
        N_s     : int,
        w_bounds: Tuple[float, float] = (-0.05, 0.05),
        dw      : float               = 0.001,
        ds      : float               = 0.5,
        cuda    : bool                = False,
    ):
        N_w = abs(w_bounds[1] - w_bounds[0]) // dw + 1
        data = np.ones((N_s, N_w, N_w, N_w), dtype=np.float64)
        if cuda:
            data = cp.asarray(data)

        obj = cls(
            data=data,
            w_bounds=w_bounds,
            ds=ds,
            dw=dw,
        )._normalize(inplace=True)

        return obj

    # uniform_distribution

# class: CurvatureDistribution

class StochasticModel(ABC):
    """ Abstract base class of stochastic modeling

    Args:
        ds: arclength resolution (units: mm)
        dw: curvature resolution (units: 1/mm)
        w_bounds: curvature bounds (units: 1/mm)

    """
    def __init__(
        self,
        needle  : Needle,
        w_bounds: Tuple[float, float] = (-0.05, 0.05),
        dw      : float               = 0.002,
        ds      : float               = 0.5,
        use_cuda: bool                = False,
    ):
        self.needle   = needle
        self.ds       = ds
        self.dw       = dw
        assert w_bounds[0] < w_bounds[1], f"Lower bound must be first and < upper bound. {w_bounds[0]} is not < {w_bounds[1]}"
        self.w_bounds = w_bounds
        self._cuda    = use_cuda

        # algorithm parameters
        self.curvature_distribution: CurvatureDistribution = None

    # __init__

    @property
    def cuda(self):
        return self._cuda

    # property: cuda

    @cuda.setter
    def cuda(self, use_cuda: bool):
        if use_cuda == self.cuda:
            return

        if use_cuda:
            self.data = cp.asarray(self.data)

        else:
            self.data = self.data.get()

        self._cuda = use_cuda

    # property setter: cuda

    @property
    def is_initialized(self):
        return (
            self.s is not None
            and self.curvature_grid is not None
            and self.curvature_distribution is not None
        )

    # property: is_initalized

    def copy(self):
        new = copy.copy(self)

        if self.is_initialized:
            new.curvature_distribution = self.curvature_distribution.copy()

        # if

        return new

    # copy

    def init_probability(self, w_init: np.ndarray = None):
        """ Initializes the probability distribution. Needs to be called before usage."""
        N_s=self.needle.length // self.ds + 1,
        if w_init is None:
            self.curvature_distribution = CurvatureDistribution.uniform_distribution(
                N_s=N_s,
                w_bounds=self.w_bounds,
                dw=self.dw,
                ds=self.ds,
                cuda=self._cuda,
            )

        else:
            self.curvature_distribution = CurvatureDistribution.dirac_delta_init(
                w_init=w_init,
                N_s=N_s,
                w_bounds=self.w_bounds,
                dw=self.dw,
                ds=self.ds,
                cuda=self._cuda,
            )

    # init_probability

    @abstractmethod
    def solve(self):
        """ Solve the distribution """
        pass

    # solve


# abstract class: StochasticModel

class StochasticShapeModel(StochasticModel):
    """ The model to be solved using the Fokker-Planck Equation """
    def __init__(
        self,
        needle             : Needle,
        shape_mdlp         : ShapeModelParameters,
        ds                 : float               = 0.5,
        dw                 : float               = 0.002,
        w_bounds           : Tuple[float, float] = (-0.05, 0.05),
        sigma_curvature    : float               = 0.0005,
        use_cuda           : bool                = False,
    ):
        super().__init__(
            needle=needle,
            ds=ds,
            dw=dw,
            w_bounds=w_bounds,
            use_cuda=use_cuda
        )
        self.shape_model_params = shape_mdlp
        self.sigma_w            = sigma_curvature # uncertainty in curvature


    # __init__

    @property
    def _array_cls(self):
        if self.cuda:
            return cp

        return np

    # property: _array_cls

    def posterior_update(s: float, prob: Union[np.ndarray, cp.ndarray]):
        """ Update the probability with a measurement

            *** To be extended by child class ***

        """

        return prob

    # posterior_update

    def solve(self):
        """ Solve the stochastic shape model using the Fokker-Planck Equation"""
        # kappa 0 function
        k0_fn = self.shape_model_params.get_k0(return_callable=True)

        # helper variables
        e1 = np.reshape([1, 0, 0], (-1, 1))
        e2 = np.reshape([0, 1, 0], (-1, 1))
        e3 = np.reshape([0, 0, 1], (-1, 1))

        A0 = self.needle.bend_stiffness
        G0 = self.needle.torsional_stiffness

        w_shape  = self.curvature_distribution.curvature_grid.shape[1:]
        N_sysmtx = self._array_cls.prod(w_shape)

        # iterate over arclengths
        for l in range(1, self.curvature_distribution.s.shape[0]):
            s_l = self.curvature_distribution.s[l]

            # get kappa0 and kappa0_prime
            k0_i, k0p_i = k0_fn(s_l)

            # generate system matrix
            # TODO: update to a sparse array
            system_matrix = self._array_cls.zeros(
                (N_sysmtx, N_sysmtx),
                dtype=self.curvature_distribution.data.dtype,
            )

            # - generate index arrays
            W_ijk       = self.curvature_distribution.curvature_grid[1:-1, 1:-1, 1:-1].reshape(3, -1)
            indices_ijk = np.stack(
                np.meshgrid(
                    np.arange(1, w_shape[0] - 1),
                    np.arange(1, w_shape[1] - 1),
                    np.arange(1, w_shape[2] - 1),
                ),
                axis=0,
            ).reshape(*W_ijk.shape)

            # - raveled index arrays
            rindx_ijk = np.ravel_multi_index(indices_ijk, w_shape)

            rindx_im1 = np.ravel_multi_index(indices_ijk - e1, w_shape)
            rindx_ip1 = np.ravel_multi_index(indices_ijk + e1, w_shape)

            rindx_jm1 = np.ravel_multi_index(indices_ijk - e2, w_shape)
            rindx_jp1 = np.ravel_multi_index(indices_ijk + e2, w_shape)

            rindx_km1 = np.ravel_multi_index(indices_ijk - e3, w_shape)
            rindx_kp1 = np.ravel_multi_index(indices_ijk + e3, w_shape)

            # - diagonal values
            system_matrix[rindx_ijk, rindx_ijk] =  (
                1 / self.ds
                + 3 * (self.sigma_w / self.dw ) ** 2
            )

            # - (i-1) & (i+1)
            system_matrix[rindx_ijk, rindx_im1] = 1/2 * (
                -(k0p_i + (A0-G0)/A0 * W_ijk[:, 1] * W_ijk[:, 2]) / self.dw
                -(self.sigma_w / self.dw)**2
            )
            system_matrix[rindx_ijk, rindx_ip1] = 1/2 * (
                (k0p_i + (A0-G0)/A0 * W_ijk[:, 1] * W_ijk[:, 2]) / self.dw
                -(self.sigma_w / self.dw)**2
            )

            # - (j-1) & (j+1)
            system_matrix[rindx_ijk, rindx_jm1] = 1/2 * (
                -(k0_i*W_ijk[2] + (G0-A0)/A0 * W_ijk[:, 0] * W_ijk[:, 2]) / self.dw
                -(self.sigma_w / self.dw)**2
            )
            system_matrix[rindx_ijk, rindx_jp1] = 1/2 * (
                (k0_i*W_ijk[2] + (G0-A0)/A0 * W_ijk[:, 0] * W_ijk[:, 2]) / self.dw
                -(self.sigma_w / self.dw)**2
            )

            # - (k-1) & (k+1)
            system_matrix[rindx_ijk, rindx_km1] = 1/2 * (
                (A0/G0 * k0_i * W_ijk[1]) / self.dw
                -(self.sigma_w / self.dw)**2
            )
            system_matrix[rindx_ijk, rindx_kp1] = 1/2 * (
                -(A0/G0 * k0_i * W_ijk[1]) / self.dw
                -(self.sigma_w / self.dw)**2
            )

            # - multiply by ds
            system_matrix *= self.ds

            # - remove all zero rows (?) TODO

            # generate update vector
            prob_lm1 = self.curvature_distribution.data[l-1].reshape(-1, )

            # - remove boundary (?) TODO

            # least squares solve
            prob_l   = self._array_cls.linalg.lstsq(
                system_matrix,
                prob_lm1,
                rcond=None,
            )

            # update the current slice
            prob_prior = prob_l.reshape(w_shape)
            self.curvature_distribution.data[l] = self.posterior_update(prob_prior)

            # normalize the slice
            self.curvature_distribution._normalize_slice(l, inplace=True)

        # for

        # normalize the distribution (safety measure)
        self.curvature_distribution._normalize(inplace=True)

        return self.curvature_distribution

    # solve

# class: StochasticShapeModel