"""

Library of needle shape sensing cost functions.

Author: Dimitri Lezcano

"""

import numpy as np
from typing import Union

import numerical
from needle_intrinsics import SingleBend


def singlebend_cost( eta: np.ndarray, data: np.ndarray, s_m: Union[ list, np.ndarray ], ds: float, N: int,
                     B: np.ndarray, Binv: np.ndarray = None, R_init: np.ndarray = np.eye( 3 ),
                     weights: np.ndarray = 1, scalef: float = 1, arg_check: bool = False ) -> float:
    """ Single bend needle cost function

        Args:
            :param eta: 4-d numpy vector of the format [kc; w_init]
            :param data: N x 3 numpy array of the curvatures at the measurement locations
            :param s_m: N-list of indexed measurement locations (must be integers >= 0)
            :param ds: float of the integration arclength increments
            :param N: int of the number of points we are integrating
            :param B: the needle stiffness matrix
            :param Binv: (Default = None) inv(B)
            :param R_init: (Default = 3x3 identity) initial orientation of the needle
            :param weights: (Default = 1) the reliability weights for each of the active areas
            :param scalef: (Default = 1) the scaling for the cost function
            :param arg_check: (Default = False) whether to check if the arguments are valid

        Return:
            :return: cost for single bend shape sensing

    """

    # argument checking
    if arg_check:
        assert (len( weights ) == data.shape[ 0 ])

    # unpack the arguments and set-up
    kc = eta[ 0 ]
    w_init = eta[ 1:3 ]

    # determine the needle arclength components
    s = ds * np.arange( N ).reshape( -1, 1 )
    L = max( s )

    # determine w0 and w0prime
    k0, k0prime = SingleBend.k0_1layer( s, kc, L )

    w0 = np.hstack( (k0, np.zeros( (N, 2) )) )
    w0prime = np.hstack( (k0prime, np.zeros( (N, 2) )) )

    # perform integration to get wv
    _, _, wv = numerical.intEP_w0( w_init, w0, w0prime, B, s0=0, ds=ds, R_init=R_init, Binv=Binv,
                                   arg_check=arg_check )

    delta = wv[ s_m, 0:1 ] - data[ :, 0:1 ]
    delta_w = delta * np.reshape( weights, (-1, 1) )
    cost = scalef * np.trace( delta_w @ delta_w.T )

    return cost

# singlebend_cost
