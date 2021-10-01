"""

Library of needle shape sensing cost functions.

Author: Dimitri Lezcano

"""

from typing import Union

import numpy as np

from . import numerical
from .intrinsics import SingleBend


def singlebend_singlelayer_cost( eta: np.ndarray, data: np.ndarray, s_m: Union[ list, np.ndarray ], ds: float,
                                 B: np.ndarray, L: float = None, N: int = None, Binv: np.ndarray = None,
                                 R_init: np.ndarray = np.eye( 3 ),
                                 weights: np.ndarray = None, scalef: float = 1, arg_check: bool = False ) -> float:
    """ Single bend and single layer needle cost function

        Args:
            :param eta: 4-d numpy vector of the format [kc; w_init]
            :param data: N x 3 numpy array of the curvatures at the measurement locations
            :param s_m: N-list of indexed measurement locations (must be integers >= 0)
            :param ds: float of the integration arclength increments
            :param B: the needle stiffness matrix
            :param L: (Default = None) float of the length of needle
            :param N: (Default = None) int of the number of points we are integrating
            :param Binv: (Default = None) inv(B)
            :param R_init: (Default = 3x3 identity) initial orientation of the needle
            :param weights: (Default = 1) the reliability weights for each of the active areas
            :param scalef: (Default = 1) the scaling for the cost function
            :param arg_check: (Default = False) whether to check if the arguments are valid

        Return:
            :return: cost for single bend and single-layer shape sensing

    """

    # argument checking
    if arg_check:
        if isinstance( weights, np.ndarray ):
            weights = weights[ :data.shape[ 0 ] ] / np.sum( weights[ :data.shape[ 0 ] ] )

    if weights is None:
        weights = np.ones(data.shape[0])/data.shape[0]

    # unpack the arguments and set-up
    kc = eta[ 0 ]
    w_init = eta[ 1:4 ]

    # determine the needle arclength components
    if N is not None:
        s = ds * np.arange( N ).reshape( -1, 1 )
        L = max( s )
    # N

    elif L is not None:
        N = int( L / ds ) + 1
        s = ds * np.arange( N ).reshape( -1, 1 )

    # elif
    else:
        raise AttributeError( "Either 'L' or 'N' needs to be set." )

    # else

    # determine w0 and w0prime
    k0, k0prime = SingleBend.k0_1layer( s, kc, L )

    w0 = np.hstack( (k0, np.zeros( (N, 2) )) )
    w0prime = np.hstack( (k0prime, np.zeros( (N, 2) )) )

    # perform integration to get wv
    _, _, wv = numerical.intEP_w0( w_init, w0, w0prime, B, s0=0, ds=ds, R_init=R_init, Binv=Binv,
                                   arg_check=arg_check )

    s_m_idx = np.argwhere(s_m.reshape(-1,1) == s.ravel())[:,1]
    delta = wv[ s_m_idx, 0:2 ] - data[ :, 0:2 ]
    delta_w = delta * np.reshape( weights, (-1, 1) )
    cost = scalef * np.trace( delta_w @ delta_w.T )

    return cost

# singlebend_cost
