"""

Library of numerical computations for needle shape sensing

Author: Dimitri Lezcano

"""
import numpy as np
import scipy.optimize as spoptim
from spatialmath.base import exp2r

from . import geometry, cost_functions
from .sensorized_needles import FBGNeedle


class NeedleParamOptimizations:
    def __init__( self, fbgneedle: FBGNeedle, ds: float = 0.5 ):
        assert (ds > 0)
        self.fbg_needle = fbgneedle
        self.ds = ds

    # __init__

    def singlebend_singlelayer_k0( self, k_c_0: float, w_init_0: np.ndarray, data: np.ndarray, L: float,
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
        slocs_tip = np.array( self.fbg_needle.sensor_location_tip )
        inserted_sensors = slocs_tip <= L
        data_ins = data[ inserted_sensors ]
        s_data_ins = L - slocs_tip[ inserted_sensors ]

        # perform the optimization
        eta_0 = np.append( [ k_c_0 ], w_init_0 )
        Binv = np.linalg.inv( self.fbg_needle.B )
        cost_kwargs = { 'L'      : L,
                        'Binv'   : Binv,
                        'R_init' : R_init,
                        'weights': self.fbg_needle.weights if len( self.fbg_needle.weights ) > 0 else None
                        }
        c_0 = cost_functions.singlebend_singlelayer_cost( eta_0, data_ins, s_data_ins, self.ds, self.fbg_needle.B,
                                                          scalef=1, arg_check=True, **cost_kwargs )
        cost_fn = lambda eta: cost_functions.singlebend_singlelayer_cost( eta, data_ins, s_data_ins, self.ds,
                                                                          self.fbg_needle.B,
                                                                          scalef=1 / c_0, arg_check=False,
                                                                          **cost_kwargs )
        res = spoptim.minimize( cost_fn, eta_0, **kwargs )
        kc, w_init = res.x[ 0 ], res.x[ 1:4 ]

        return kc, w_init, res

    # singlebend_singlelayer_k0


# NeedleParamOptimiztions

def intEP_w0( w_init: np.ndarray, w0: np.ndarray, w0prime: np.ndarray, B: np.ndarray,
              s: np.ndarray = None, s0: float = 0, ds: float = None, R_init: np.ndarray = np.eye( 3 ),
              Binv: np.ndarray = None, arg_check: bool = True ) -> (np.ndarray, np.ndarray, np.ndarray):
    """ integrate Euler-Poincare equation for needle shape sensing for given intrinsic angular deformation

        Original Author: Jin Seob Kim
        Editing Author: Dimitri Lezcano

        Args:
            w_init: 3-D initial deformation vector
            w0: N x 3 intrinsic angular deformation
            w0prime: N x 3 d/ds w0
            B: 3 x 3 needle stiffness matrix
            s: (Default = None) the arclengths desired (Not implemented)
            s0: (Deafult = 0) the initial length to start off with
            ds: (Default = None) the arclength increments desired
            Binv: (Default = None) inv(B) Can be provided for numerical efficiency
            R_init: (Default = 3x3 identity) SO3 matrix for initial rotation angle
            arg_check: (Default = False) whether to check if the arguments are valid

        Return:
            (N x 3 needle shape, N x 3 x 3 SO3 matrices of orientations), N x 3 angular deformation)
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
        raise NotImplementedError( 'You must use ds here only for uniform distribution' )
        # s = s.flatten()
        # assert (s.size == N)

    # elif

    else:
        raise ValueError( "Either `s` or `ds` needs to be provided. One or the other." )

    s = s[ s >= s0 ]

    if Binv is None:
        Binv = np.linalg.inv( B )

    elif arg_check:
        assert (Binv.shape == (3, 3))

    # initialize the return matrices
    wv = np.zeros( (N, 3) )
    pmat = np.zeros( (N, 3) )
    Rmat = np.expand_dims( np.eye( 3 ), axis=0 ).repeat( N, axis=0 )

    # initial conditions
    wv[ 0 ] = w_init
    Rmat[ 0 ] = R_init

    # perform integration to calculate angular deformation vector
    for i in range( 1, N ):
        ds = s[ i ] - s[ i - 1 ]  # calculate ds
        if i == 1:
            wv[ i ] = w_init + ds * (w0prime[ 0 ] - Binv @ np.cross( wv[ 0 ], B @ (w_init - w0[ 0 ]) ))

        else:
            wv[ i ] = wv[ i - 2 ] + 2 * ds * (
                    w0prime[ i - 1 ] - Binv @ np.cross( wv[ i - 1 ], B @ (wv[ i - 1 ] - w0[ i - 1 ]) ))

    # for

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

    return pmat, Rmat, wv


# intEP_w0

def simpson_vec_int( f: np.ndarray, dx: float ) -> np.ndarray:
    """ Implementation of Simpon vector integration

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
            m = num_intervals - 1

        # else

        int_res += dx / 3 * (f[ :, 1 ] + 4 * np.sum( f[ :, 1:2:m ], axis=1 ) + f[ :, m + 1 ])

        if m > 2:
            int_res += dx / 3 * 2 * np.sum( f[ :, 2:2:m ], axis=1 )

        # if

    # else

    return int_res

# simpson_vec_int
