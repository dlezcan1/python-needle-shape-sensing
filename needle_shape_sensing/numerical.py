"""

Library of numerical computations for needle shape sensing

Author: Dimitri Lezcano

"""
import warnings
from typing import Union, Callable

import numpy as np
import scipy.integrate.odepack
import scipy.optimize as spoptim
from scipy.integrate import odeint, solve_ivp
from scipy import interpolate
from spatialmath.base import exp2r
from numba import jit

from . import geometry, cost_functions
from .sensorized_needles import FBGNeedle


class NeedleParamOptimizations:
    def __init__(
            self, fbgneedle: FBGNeedle, ds: float = 0.5, optim_options: dict = None,
            continuous: bool = True ):
        assert (ds > 0)
        self.fbg_needle = fbgneedle

        # don't calculate each time
        self.__needle_B = self.fbg_needle.B
        self.__needle_Binv = np.linalg.inv( self.fbg_needle.B )

        self.ds = ds

        # optimizer options
        default_options = {
                'w_init_bounds': [ (-0.01, 0.01) ] * 3,
                'kc_bounds'    : [ (0, 0.01) ],
                'tol'          : 1e-8
                }
        self.options = default_options
        self.options.update( optim_options if optim_options is not None else { } )

        # integration options
        self.continuous = continuous

    # __init__

    def constant_curvature( self, data: np.ndarray, L: float ):
        """
            Compute the constant curvature

        """
        assert (L > 0)
        # check for inserted into tissue
        s_data_ins, inserted_sensors = self.fbg_needle.calculate_length_measured(
                L, tip=True, valid=True )
        data_ins = data[ inserted_sensors ]
        if len( self.fbg_needle.weights ) > 0:
            weights = np.array(
                    [ self.fbg_needle.weights[ key ] for (key, inserted) in
                      zip( self.fbg_needle.weights.keys(), inserted_sensors ) if inserted ] )
        # if
        else:
            weights = np.ones_like( s_data_ins )

        # else

        # normalize the weights
        weights = weights / np.sum( weights )

        # compute the weighted-mean curvature from the measurements
        data_ins_w = data_ins * weights.reshape( -1, 1 )
        curvature = data_ins_w.mean( axis=0 )

        return curvature

    # constant_curvature

    def singlebend_singlelayer_k0(
            self, k_c_0: float, w_init_0: np.ndarray, data: np.ndarray, L: float,
            R_init: np.ndarray = np.eye( 3 ), **kwargs ):
        """
            :param k_c_0: initial intrinsic curvature
            :param w_init_0: initial w_init
            :param data: N x 3 array of curvatures for each of the active areas
            :param L: the insertion depth
            :param R_init: (Default = numpy.eye(3)) the initial rotation offset

            :keyword: any scipy optimize (least_squares) kwargs

            :return: kc, w_init optimized, optimization results
        """
        # argument checking
        assert (L > 0)
        # check for inserted into tissue
        s_data_ins, inserted_sensors = self.fbg_needle.calculate_length_measured(
                L, tip=True, valid=True )
        data_ins = data[ inserted_sensors ]
        if len( self.fbg_needle.weights ) > 0:
            weights = np.array(
                    [ self.fbg_needle.weights[ key ] for (key, inserted) in
                      zip( self.fbg_needle.weights.keys(), inserted_sensors ) if inserted ] )
        # if
        else:
            weights = None

        # else

        # perform the optimization
        eta_0 = np.append( [ k_c_0 ], w_init_0 )
        Binv = kwargs.get( 'Binv', self.__needle_Binv )
        cost_kwargs = {
                'L'      : L,
                'Binv'   : Binv,
                'R_init' : R_init,
                'weights': weights
                }
        c_0 = cost_functions.singlebend_singlelayer_cost(
                eta_0, data_ins, s_data_ins, self.ds, self.__needle_B,
                scalef=1, arg_check=True, **cost_kwargs )
        c_f = 1 / c_0 if c_0 > 0 else 1
        cost_fn = lambda eta: cost_functions.singlebend_singlelayer_cost(
                eta, data_ins, s_data_ins, self.ds,
                self.fbg_needle.B,
                scalef=c_f, arg_check=False,
                continuous=self.continuous,
                **cost_kwargs )
        res = self.__optimize( cost_fn, eta_0, **kwargs )
        kc, w_init = res.x[ 0 ], res.x[ 1:4 ]

        return kc, w_init, res

    # singlebend_singlelayer_k0

    def singlebend_doublelayer_k0(
            self, kc1_0: float, kc2_0: float, w_init_0: np.ndarray, data: np.ndarray, L: float,
            s_crit: float = None, z_crit: float = None, R_init: np.ndarray = np.eye( 3 ),
            **kwargs ):
        """
            :param kc1_0: initial intrinsic curvature for layer 1
            :param kc2_0: initial intrinsic curvature for layer 2
            :param w_init_0: initial w_init
            :param data: N x 3 array of curvatures for each of the active areas
            :param L: the insertion depth
            :param z_crit: (Default = None) the length of the first layer
            :param s_crit: (Default = None) the length of the needle in first layer
            :param R_init: (Default = numpy.eye(3)) the initial rotation offset

            :keyword: any scipy optimize (least_squares) kwargs

            :return: kc1, kc2, w_init optimized, optimization results
        """
        assert (L >= z_crit > 0)
        # check for inserted into tissue
        s_data_ins, inserted_sensors = self.fbg_needle.calculate_length_measured(
                L, tip=True, valid=True )
        data_ins = data[ inserted_sensors ]
        if len( self.fbg_needle.weights ) > 0:
            weights = np.array(
                    [ self.fbg_needle.weights[ key ] for (key, inserted) in
                      zip( self.fbg_needle.weights.keys(), inserted_sensors ) if inserted ] )
        # if
        else:
            weights = None

        # else

        # perform the optimization
        eta_0 = np.append( [ kc1_0, kc2_0 ], w_init_0 )
        Binv = kwargs.get( 'Binv', self.__needle_Binv )
        cost_kwargs = {
                'L'      : L,
                's_crit' : s_crit,
                'z_crit' : z_crit,
                'Binv'   : Binv,
                'R_init' : R_init,
                'weights': weights
                }
        c_0 = cost_functions.singlebend_doublelayer_cost(
                eta_0, data_ins, s_data_ins, self.ds, self.__needle_B,
                scalef=1, arg_check=True, **cost_kwargs )
        c_f = 1 / c_0 if c_0 > 0 else 1
        cost_fn = lambda eta: cost_functions.singlebend_doublelayer_cost(
                eta, data_ins, s_data_ins, self.ds,
                self.fbg_needle.B,
                scalef=c_f, arg_check=False,
                continuous=self.continuous,
                **cost_kwargs )
        res = self.__optimize( cost_fn, eta_0, **kwargs )
        kc1, kc2, w_init = res.x[ 0 ], res.x[ 1 ], res.x[ 2:5 ]

        return kc1, kc2, w_init, res

    # singlebend_doublelayer_k0

    def doublebend_singlelayer_k0(
            self, kc_0: float, w_init_0: np.ndarray, data: np.ndarray, L: float,
            s_crit: float, R_init: np.ndarray = np.eye( 3 ), **kwargs ):
        """
            :param kc_0: initial intrinsic curvature for layer 1
            :param w_init_0: initial w_init
            :param data: N x 3 array of curvatures for each of the active areas
            :param L: the insertion depth
            :param s_crit: (Default = None) the length of the needle in first layer
            :param R_init: (Default = numpy.eye(3)) the initial rotation offset

            :keyword: any scipy optimize (least_squares) kwargs

            :return: kc1, kc2, w_init optimized, optimization results
        """
        assert (L >= s_crit > 0)
        # check for inserted into tissue
        s_data_ins, inserted_sensors = self.fbg_needle.calculate_length_measured(
                L, tip=True, valid=True )
        data_ins = data[ inserted_sensors ]
        if len( self.fbg_needle.weights ) > 0:
            weights = np.array(
                    [ self.fbg_needle.weights[ key ] for (key, inserted) in
                      zip( self.fbg_needle.weights.keys(), inserted_sensors ) if inserted ] )
        # if
        else:
            weights = None

        # else

        # perform the optimization
        eta_0 = np.append( [ kc_0 ], w_init_0 )
        Binv = kwargs.get( 'Binv', self.__needle_Binv )
        cost_kwargs = {
                'L'      : L,
                'Binv'   : Binv,
                'R_init' : R_init,
                'weights': weights
                }
        c_0 = cost_functions.doublebend_singlelayer_cost(
                eta_0, data_ins, s_data_ins, self.ds, s_crit,
                self.fbg_needle.B,
                scalef=1, arg_check=True, **cost_kwargs )
        c_f = 1 / c_0 if c_0 > 0 else 1
        cost_fn = lambda eta: cost_functions.doublebend_singlelayer_cost(
                eta, data_ins, s_data_ins, self.ds,
                s_crit, self.__needle_B,
                scalef=c_f, arg_check=False,
                continuous=self.continuous,
                **cost_kwargs )
        res = self.__optimize( cost_fn, eta_0, **kwargs )

        kc, w_init = res.x[ 0 ], res.x[ 1:4 ]

        return kc, w_init, res

    # doublebend_singlelayer_k0

    def __optimize( self, cost_fn, eta_0, **kwargs ):
        """ Optimize wrapper for multiple functions"""
        optim_options = self.options.copy()
        optim_options.update( kwargs )

        # filter specific bounds and add bounds if not already specified
        exclude_keys = [ 'w_init_bounds', 'kc_bounds', 'needle_rotations' ]
        optim_options = dict(
                filter( lambda x: x[ 0 ] not in exclude_keys, optim_options.items() ) )
        if optim_options.get( 'bounds' ) is None:
            bounds = self.options[ 'kc_bounds' ] * (eta_0.size - 3) + self.options[
                'w_init_bounds' ]
            optim_options[ 'bounds' ] = bounds

        # bounds

        # perform optimization
        with warnings.catch_warnings():
            warnings.simplefilter( "ignore", scipy.integrate.odepack.ODEintWarning )
            result = spoptim.minimize( cost_fn, eta_0, **optim_options )

        # with

        return result

    # __optimize


# NeedleParamOptimizations

@jit( nopython=True, parallel=True )
def jit_linear_interp1d( t: float, x_all: np.ndarray, t_all: np.ndarray, is_sorted: bool = False ):
    """ numba 1d linear interpolation"""
    if not is_sorted:
        idxs = np.argsort( t_all )
        t_all = t_all[ idxs ]
        x_all = x_all[ idxs ]

    # if
    if t in t_all:
        return x_all[ t == t_all ]

    else:
        t1_idx = np.argmin( np.abs( t - t_all ) )
        t1 = t_all[ t1_idx ]
        x1 = x_all[ t1 ]

        t2_idx = t1_idx + 1 if t1 < t else t1_idx - 1
        t2 = t_all[ t2_idx ]
        x2 = x_all[ t2 ]

        return x1 + (x2 - x1) * (t - t1) / (t2 - t1)

    # else


# jit_linear_interp1d

# @jit( nopython=True, parallel=True )
def differential_EPeq( wv: np.ndarray, s: float, w0, w0prime, B: np.ndarray, Binv: np.ndarray ):
    """ The differential form of the Euler-Poincare equation equation

        :param wv: the current omega v
        :param s: the current arclength
        :param w0: the intrinsic angular deformation
        :param w0prime: d(w0)/ds
        :param B: the stiffness matrix
        :param Binv: the inverse of the stiffness matrix

        :returns: dwv/ds

    """
    dwvds = w0prime( s ).ravel() - Binv @ np.cross( wv, B @ (wv - w0( s ).ravel()) )

    return dwvds


# differential_EPeq

def integrateEP_w0(
        w_init: np.ndarray, w0: np.ndarray, w0prime: np.ndarray, B: np.ndarray,
        s: np.ndarray = None, s0: float = 0, ds: float = None, R_init: np.ndarray = np.eye( 3 ),
        Binv: np.ndarray = None, needle_rotations: list = None, arg_check: bool = True,
        wv_only: bool = False ) -> (
        np.ndarray, np.ndarray, np.ndarray):
    """ integrate Euler-Poincare equation for needle shape sensing for given intrinsic angular deformation

        Original Author: Jin Seob Kim
        Editing Author: Dimitri Lezcano

        Args:
            w_init: 3-D initial deformation vector
            w0: N x 3 intrinsic angular deformation
            w0prime: N x 3 d/ds w0
            B: 3 x 3 needle stiffness matrix
            s: (Default = None) the arclengths desired (Not implemented)
            s0: (Default = 0) the initial length to start off with
            ds: (Default = None) the arclength increments desired
            Binv: (Default = None) inv(B) Can be provided for numerical efficiency
            R_init: (Default = 3x3 identity) SO3 matrix for initial rotation angle
            needle_rotations: (Default = None) list of rotations in degrees of the needle rotations
            arg_check: (Default = False) whether to check if the arguments are valid
            wv_only: (Default = False) whether to only integrate wv or not.

        Return:
            (N x 3 needle shape, N x 3 x 3 SO3 matrices of orientations), N x 3 angular deformation)
            (None, None, N x 3 angular deformation) if 'wv_only' is True
    """
    # argument validation
    if arg_check:
        assert (w_init.size == 3)
        w_init = w_init.flatten()

        assert (w0.shape[ 1 ] == 3 and w0prime.shape[ 1 ] == 3)
        assert (w0.shape[ 0 ] == w0prime.shape[ 0 ])

        assert (B.shape == (3, 3))
        assert (geometry.is_SO3( R_init ))

        assert (s0 >= 0)

    # if

    # argument parsing
    N = w0.shape[ 0 ]
    if (s is None) and (ds is not None):
        s = s0 + ds * np.arange( N )

    elif s is not None:
        if np.any( np.diff( s )[ 0 ] != np.diff( s ) ):  # ensure constant ds
            raise NotImplementedError( 'You must use ds here only for uniform distribution' )

        else:
            ds = np.diff( s )[ 0 ]

    # elif

    else:
        raise ValueError( "Either `s` or `ds` needs to be provided. One or the other." )

    s = s[ s >= s0 ]

    if Binv is None:
        Binv = np.linalg.inv( B )

    elif arg_check:
        assert (Binv.shape == (3, 3))

    # needle rotation axis
    if needle_rotations is None:
        needle_rotations = [ 0 ] * len( s )

    else:
        needle_rotations = needle_rotations[ s >= s0 ]

    # update w0 and w0prime
    for idx in range( N ):
        Rz = geometry.rotz( needle_rotations[ idx ] )
        w0[ idx ] = Rz @ w0[ idx ]
        w0prime[ idx ] = Rz @ w0prime[ idx ]

    # for

    # initialize the return matrices
    wv = np.zeros( (N, 3) )

    # initial conditions
    wv[ 0 ] = w_init

    # perform integration to calculate angular deformation vector
    for idx in range( 1, N ):
        ds = s[ idx ] - s[ idx - 1 ]  # calculate ds
        Rz = geometry.rotz( needle_rotations[ idx ] )
        if idx == 1:
            wv[ idx ] = w_init + ds * (
                    w0prime[ 0 ] - Binv @ np.cross( wv[ 0 ], B @ (w_init - w0[ 0 ]) ))

        else:
            wv[ idx ] = wv[ idx - 2 ] + 2 * ds * (
                    w0prime[ idx - 1 ] - Binv @ np.cross(
                    wv[ idx - 1 ], B @ (wv[ idx - 1 ] - w0[ idx - 1 ]) ))

    # for

    # integrate angular deviation vector in order to get the pose
    if wv_only:
        pmat, Rmat = None, None
    else:
        pmat, Rmat = integratePose_wv( wv, s=s, s0=s0, ds=ds, R_init=R_init )

    return pmat, Rmat, wv


# integrateEP_w0

def integrateEP_w0_ode(
        w_init: np.ndarray, w0: Union[ Callable, np.ndarray ],
        w0prime: Union[ Callable, np.ndarray ],
        B: np.ndarray, s: np.ndarray, s0: float = 0, ds: float = None,
        R_init: np.ndarray = np.eye( 3 ), Binv: np.ndarray = None, arg_check: bool = True,
        needle_rotations: list = None, wv_only: bool = False, integration_method = 'RK45' ) -> (
        np.ndarray, np.ndarray, np.ndarray):
    """ integrate Euler-Poincare equation for needle shape sensing for given intrinsic angular deformation
        using scipy.integrate

        Author: Dimitri Lezcano

        Args:
            w_init: 3-D initial deformation vector
            w0: Callable function or N x 3 intrinsic angular deformation
            w0prime: Callable function or N x 3 d/ds w0
            B: 3 x 3 needle stiffness matrix
            s: the arclengths desired (Not implemented)
            s0: (Default = 0) the initial length to start off with
            ds: (Default = None) the arclength increments desired
            Binv: (Default = None) inv(B) Can be provided for numerical efficiency
            R_init: (Default = 3x3 identity) SO3 matrix for initial rotation angle
            arg_check: (Default = False) whether to check if the arguments are valid
            needle_rotations: (Default = None) a list of needle axis rotations
            wv_only: (Default = False) whether to only integrate wv or not.
            integration_method: (Default = 'RK45') type of integration ('odeint', 'RK23', 'RK45')
                'odeint': speed but unstable
                'RK23':   slower than 'odeint', faster than 'RK45', more stable than 'odeint'
                'RK45':   slowest, but more stable than the rest.

        Return:
            (N x 3 needle shape, N x 3 x 3 SO3 matrices of orientations), N x 3 angular deformation)
            (None, None, N x 3 angular deformation) if 'wv_only' is True
    """
    if arg_check:
        assert (w_init.size == 3)
        w_init = w_init.flatten()

        assert (B.shape == (3, 3))
        assert (geometry.is_SO3( R_init ))

        assert (s0 >= 0)

    # if

    # argument parsing
    s = s[ s >= s0 ]

    if Binv is None:
        Binv = np.linalg.inv( B )

    elif arg_check:
        assert (Binv.shape == (3, 3))

    # setup intrinsic curvature functions
    if callable( w0 ):
        w0_fn = w0
    else:
        w0_fn = interpolate.interp1d( s, w0.T, fill_value='extrapolate' )
        # w0_fn = lambda t: jit_linear_interp1d( t, w0, s )

    if callable( w0prime ):
        w0prime_fn = w0prime
    else:
        w0prime_fn = interpolate.interp1d( s, w0prime.T, fill_value='extrapolate' )
        # w0prime_fn = lambda t: jit_linear_interp1d( t, w0prime, s )

    # use needle rotation interpolation
    if needle_rotations is None:
        needle_rotation_fn = lambda s: 0  # default nullify implementation

    elif callable( needle_rotations ):
        needle_rotation_fn = needle_rotations

    else:
        needle_rotation_fn = interpolate.interp1d( s, needle_rotations, fill_value=0 )

    w0_rot_fn = lambda s: geometry.rotz( needle_rotation_fn( s ) ) @ w0_fn( s )
    w0prime_rot_fn = lambda s: geometry.rotz( needle_rotation_fn( s ) ) @ w0prime_fn( s )

    # perform integration
    ode_EP = lambda s, wv: differential_EPeq( wv, s, w0_rot_fn, w0prime_rot_fn, B, Binv )
    if integration_method.lower() == 'odeint':
        wv = odeint( ode_EP, w_init, s, full_output=False, hmin=ds / 2, h0=ds / 2, tfirst=True )

    else:
        wv_res = solve_ivp( ode_EP, (s0, s.max()), w_init, method=integration_method, t_eval=s,
                    first_step=ds )  # 'RK23' for speed (all slower than odeint) 'RK45' for accuracy
        wv = wv_res.y.T

    # else


    # integrate angular deviation vector in order to get the pose
    if wv_only:
        pmat, Rmat = None, None
    else:
        pmat, Rmat = integratePose_wv( wv, s=s, s0=s0, ds=ds, R_init=R_init )

    return pmat, Rmat, wv


# integrateEP_w0_ode

def integratePose_wv(
        wv, s: np.ndarray = None, s0: float = 0, ds: float = None,
        R_init: np.ndarray = np.eye( 3 ) ):
    """ Integrate angular deformation to get the pose of the needle along it's arclengths

        :param wv: N x 3 angular deformation vector
        :param s: numpy array of arclengths to integrate
        :param ds: (Default = None) the arclength increments desired
        :param s0: (Default = 0) the initial length to start off with
        :param R_init: (Default = numpy.eye(3)) Rotation matrix of the inital pose

        :returns: pmat, Rmat
            - pmat: N x 3 position for the needle shape points in-tissue
            -Rmat: N x 3 x 3 SO(3) rotation matrices for
    """
    # set-up the containers
    N = wv.shape[ 0 ]
    pmat = np.zeros( (N, 3) )
    Rmat = np.expand_dims( np.eye( 3 ), axis=0 ).repeat( N, axis=0 )
    Rmat[ 0 ] = R_init

    # get the arclengths
    if (s is None) and (ds is not None):
        s = s0 + np.arange( N ) * ds
    elif s is not None:
        pass
    else:
        raise ValueError( "Either 's' or 'ds' must be used, not both." )

    # else

    # integrate angular deviation vector in order to get the pose
    for i in range( 1, N ):
        ds = s[ i ] - s[ i - 1 ]
        Rmat[ i ] = exp2r( ds * np.mean( wv[ i - 1:i ], axis=0 ) )

        e3vec = Rmat[ :i + 1, :, 2 ].T  # grab z-direction coordinates

        if i == 1:
            pmat[ i ] = pmat[ i - 1 ] + Rmat[ i, :, 2 ] * ds
        else:
            pmat[ i ] = simpson_vec_int( e3vec, ds )

    # for

    return pmat, Rmat


# integratePose_wv

def interpolate_curve_s( points: np.ndarray, s_interp: np.ndarray, axis=0 ):
    """ Interpolate a curve based on its arclength via linear interpolation

        :param points: the points to interpolate by (N x M array)
        :param s_interp: 1D array of arclengths to interpolate
        :param axis: (Default=0), the axis to perform the interpolation on.

        :return: Interpolated curve points w.r.t s_interp, s_interp


    """
    # get the arclengths of the curve
    L, s = geometry.arclength( points, axis=axis )

    s_interp = s_interp[
        (s_interp <= L) & (s_interp >= s.min()) ]  # cut-off extrapolated curve values

    # begin interpolation
    points_interp = interpolate.interp1d( s, points, axis=axis )( s_interp )

    return points_interp, s_interp


# interpolate_curve_s


def simpson_vec_int( f: np.ndarray, dx: float ) -> np.ndarray:
    """ Implementation of Simpson vector integration

        Original Author (MATLAB): Jin Seob Kim
        Translated Author: Dimitri Lezcano

        Args:
             f:  m x n numpy array where m is the dimension of the vector and n is the dimension of the parameter ( n > 2 )
                    Integration intervals
             dx: float of the step size

        Return:
            numpy vector of shape (m,)

    """
    num_intervals = f.shape[ 1 ] - 1
    assert (num_intervals > 1)  # need as least a single interval

    # TODO: non-uniform dx integration

    # perform the integration
    if num_intervals == 2:  # base case 1
        int_res = dx / 3 * np.sum( f[ :, 0:3 ] * [ [ 1, 4, 1 ] ], axis=1 )

    # if
    elif num_intervals == 3:  # base case 2
        int_res = 3 / 8 * dx * np.sum( f[ :, 0:4 ] * [ [ 1, 3, 3, 1 ] ], axis=1 )

    # elif

    else:
        int_res = np.zeros( (f.shape[ 0 ]) )

        if num_intervals % 2 != 0:
            int_res += 3 / 8 * dx * np.sum( f[ :, -4: ] * [ [ 1, 3, 3, 1 ] ], axis=1 )
            m = num_intervals - 3

        # if
        else:
            m = num_intervals

        # else

        int_res += dx / 3 * (f[ :, 0 ] + 4 * np.sum( f[ :, 1:m:2 ], axis=1 ) + f[ :, m ])

        if m > 2:
            int_res += dx / 3 * 2 * np.sum( f[ :, 2:m:2 ], axis=1 )

        # if

    # else

    return int_res

# simpson_vec_int
