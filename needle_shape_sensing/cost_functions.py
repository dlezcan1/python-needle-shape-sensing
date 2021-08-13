"""

Library of needle shape sensing cost functions.

Author: Dimitri Lezcano

"""

import numpy as np
from typing import Union

import numerical


def singlebend_cost( eta: np.ndarray, data: np.ndarray, s_m: Union[ list, np.ndarray ], ds: float, N: int,
                     B: np.ndarray, Binv: np.ndarray = None, R_init: np.ndarray = np.eye( 3 ),
                     scalef: float = 1, arg_check: bool = False ) -> float:
    """ Single bend needle cost function

        Args:
            eta: 4-d numpy vector of the format [kc; w_init]
            data: N x 3 numpy array of the curvatures at the measurement locations
            s_m: N-list of indexed measurement locations (must be integers >= 0)
            ds: float of the integration arclength increments
            N: int of the number of points we are integrating
            B: the needle stiffness matrix
            Binv: (Default = None) inv(B)
            R_init: (Default = 3x3 identity) initial orientation of the needle
            scalef: (Default = 1) the scaling for the cost function
            arg_check: (Default = False) whether to check if the arguments are valid

        Return:
            cost for single bend shape sensing

    """
    # unpack the arguments and set-up
    kc = eta[ 0 ]
    w_init = eta[ 1:3 ]

    # determine the needle arclength components
    s = ds * np.arange( N )
    L = max( s )

    # determine w0 and w0prime
    k0 = kc * (1 - s / L) ** 2
    k0prime = -2 * kc / L * (1 - s / L)

    w0 = np.vstack( (k0, np.zeros( (2, N) )) ).T
    w0prime = np.vstack( (k0prime, np.zeros( 2, N )) ).T

    # perform integration to get wv
    _, _, wv = numerical.intEP_w0( w_init, w0, w0prime, B, s0=0, ds=ds, R_init=R_init, Binv=Binv,
                                   arg_check=arg_check )

    deltav = wv[ s_m, 0:1 ] - data[ :, 0:1 ]
    cost = scalef * np.trace( deltav @ deltav.T )

    return cost

# singlebend_cost
