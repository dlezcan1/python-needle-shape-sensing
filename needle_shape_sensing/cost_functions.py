"""

Library of needle shape sensing cost functions.

Author: Dimitri Lezcano

"""

from typing import Union

import numpy as np

from . import numerical, intrinsics


def constant_curvature_cost( eta: np.ndarray, data, s_m: Union[ list, np.ndarray ], ds: float,
                             L: float = None, N: int = None, weights: np.ndarray = None, scalef: float = 1 ) -> float:
    """ Perform constant curvature cost functions"""
    # determine the needle arclength components
    if N is not None:
        L = N * ds
    # N

    elif L is not None:
        pass  # no need to do anything

    # elif
    else:
        # raise AttributeError( "Either 'L' or 'N' needs to be set." )
        L = np.inf  # assume all sensing locations are valid

    # else

    curvatures = np.repmat( eta[ :2 ].reshape( 1, -1 ), data.shape[ 0 ], axis=0 )

    cost = scalef * curvature_cost( data, curvatures, np.arange( data.shape[ 0 ] ), weights=weights )

    return cost


# constant_curvature_cost

def singlebend_singlelayer_cost( eta: np.ndarray, data: np.ndarray, s_m: Union[ list, np.ndarray ], ds: float,
                                 B: np.ndarray, L: float = None, N: int = None, Binv: np.ndarray = None,
                                 R_init: np.ndarray = np.eye( 3 ), weights: np.ndarray = None, scalef: float = 1,
                                 arg_check: bool = False, continuous: bool = False ) -> float:
    """ Single bend and single layer needle cost function

        Args:
            :param eta: 4-d numpy vector of the format [kc; w_init]
            :param data: N x 3 numpy array of the curvatures at the measurement locations
            :param s_m: N-list of indexed measurement locations from tip of needle (must be integers >= 0)
            :param ds: float of the integration arclength increments
            :param B: the needle stiffness matrix
            :param L: (Default = None) float of the length of needle
            :param N: (Default = None) int of the number of points we are integrating
            :param Binv: (Default = None) inv(B)
            :param R_init: (Default = 3x3 identity) initial orientation of the needle
            :param weights: (Default = 1) the reliability weights for each of the active areas
            :param scalef: (Default = 1) the scaling for the cost function
            :param arg_check: (Default = False) whether to check if the arguments are valid
            :param continuous: (Default = False) whether to perform continuous integration or not

        Return:
            :return: cost for single bend and single-layer shape sensing

    """

    # argument checking
    if arg_check:
        pass

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
    k0, k0prime = intrinsics.SingleBend.k0_1layer( s, kc, L, return_callable=continuous )

    # perform integration to get wv
    if continuous:
        w0 = lambda s: np.append( k0( s ), [ 0, 0 ] )
        w0prime = lambda s: np.append( k0prime( s ), [ 0, 0 ] )

        _, _, wv = numerical.integrateEP_w0_ode( w_init, w0, w0prime, B, s, s0=0, ds=ds, R_init=R_init, Binv=Binv,
                                                 arg_check=arg_check, wv_only=True )

    # if
    else:
        w0 = np.hstack( (k0.reshape( -1, 1 ), np.zeros( (N, 2) )) )
        w0prime = np.hstack( (k0prime.reshape( -1, 1 ), np.zeros( (N, 2) )) )

        _, _, wv = numerical.integrateEP_w0( w_init, w0, w0prime, B, s0=0, ds=ds, R_init=R_init, Binv=Binv,
                                             arg_check=arg_check, wv_only=True )

    # else

    # determine AA locations
    # s_m_idx = np.argwhere( s_m.reshape( -1, 1 ) == s.ravel() )[ :, 1 ]
    s_m_idx = np.argmin(np.abs(s_m.reshape(-1,1) - s.ravel()), axis=1)

    # compute cost
    cost = scalef * curvature_cost( data, wv, s_m_idx, weights=weights )

    return cost


# singlebend_singlelayer_cost


def singlebend_doublelayer_cost( eta: np.ndarray, data: np.ndarray, s_m: Union[ list, np.ndarray ], ds: float,
                                 B: np.ndarray, L: float = None, s_crit: float = None, z_crit: float = None,
                                 N: int = None, Binv: np.ndarray = None, R_init: np.ndarray = np.eye( 3 ),
                                 weights: np.ndarray = None, scalef: float = 1, arg_check: bool = False,
                                 continuous: bool = False ) -> float:
    """ Single bend and single layer needle cost function

        Args:
            :param eta: 5-d numpy vector of the format [kc1; kc2; w_init]
            :param data: N x 3 numpy array of the curvatures at the measurement locations
            :param s_m: N-list of indexed measurement locations (must be integers >= 0)

            :param ds: float of the integration arclength increments
            :param B: the needle stiffness matrix
            :param L: (Default = None) float of the length of needle
            :param s_crit: (Default = None) the length of the needle at boundary of insertion
            :param z_crit: (Default = None) the z-critical layer boundary (will calculate s_crit)
            :param N: (Default = None) int of the number of points we are integrating
            :param Binv: (Default = None) inv(B)
            :param R_init: (Default = 3x3 identity) initial orientation of the needle
            :param weights: (Default = 1) the reliability weights for each of the active areas
            :param scalef: (Default = 1) the scaling for the cost function
            :param arg_check: (Default = False) whether to check if the arguments are valid
            :param continuous: (Default = False) whether to perform continuous integration or not

        Return:
            :return: cost for single bend and single-layer shape sensing

    """
    # argument checking
    if arg_check:
        pass

    # unpack the arguments and set-up
    kc1 = eta[ 0 ]
    kc2 = eta[ 1 ]
    w_init = eta[ 2:5 ]

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

    # determine s_crit: the length of the needle in the first layer
    if s_crit is not None:
        pass

    # if
    elif z_crit is not None:
        s_crit = intrinsics.SingleBend.determine_2layer_boundary( kc1, L, z_crit, B, w_init=w_init, s0=0, ds=ds,
                                                                  R_init=R_init, Binv=Binv, continuous=continuous )
    # elif
    else:
        raise ValueError( "Either s_crit or z_crit must be defined." )

    # else

    # determine k0 and k0prime
    if s_crit < 0:  # single-layer
        k0, k0prime = intrinsics.SingleBend.k0_1layer( s, kc1, L, return_callable=continuous )

    else:  # double-layer
        k0, k0prime = intrinsics.SingleBend.k0_2layer( s, kc1, kc2, L, s_crit, return_callable=continuous )

    # perform integration to get wv
    if continuous:
        w0 = lambda s: np.append( k0( s ), [ 0, 0 ] )
        w0prime = lambda s: np.append( k0prime( s ), [ 0, 0 ] )

        _, _, wv = numerical.integrateEP_w0_ode( w_init, w0, w0prime, B, s, s0=0, ds=ds, R_init=R_init, Binv=Binv,
                                                 arg_check=arg_check, wv_only=True )

    # if
    else:
        w0 = np.hstack( (k0.reshape( -1, 1 ), np.zeros( (N, 2) )) )
        w0prime = np.hstack( (k0prime.reshape( -1, 1 ), np.zeros( (N, 2) )) )

        _, _, wv = numerical.integrateEP_w0( w_init, w0, w0prime, B, s0=0, ds=ds, R_init=R_init, Binv=Binv,
                                             arg_check=arg_check, wv_only=True )

    # else

    # calculate cost
    s_m_idx = np.argwhere( s_m.reshape( -1, 1 ) == s.ravel() )[ :, 1 ]
    cost = scalef * curvature_cost( data, wv, s_m_idx, weights=weights )

    return cost


# singlebend_doublelayer_cost


def doublebend_singlelayer_cost( eta: np.ndarray, data: np.ndarray, s_m: Union[ list, np.ndarray ], ds: float,
                                 s_crit: float, B: np.ndarray, L: float = None, N: int = None, Binv: np.ndarray = None,
                                 R_init: np.ndarray = np.eye( 3 ), weights: np.ndarray = None, scalef: float = 1,
                                 arg_check: bool = False, continuous: bool = False ) -> float:
    """ Single bend and single layer needle cost function

        Args:
            :param eta: 4-d numpy vector of the format [kc; w_init]
            :param data: N x 3 numpy array of the curvatures at the measurement locations
            :param s_m: N-list of indexed measurement locations (must be integers >= 0)
            :param ds: float of the integration arclength increments
            :param B: the needle stiffness matrix
            :param L: (Default = None) float of the length of needle
            :param s_crit: (Default = None) the length of the needle at boundary of insertion
            :param N: (Default = None) int of the number of points we are integrating
            :param Binv: (Default = None) inv(B)
            :param R_init: (Default = 3x3 identity) initial orientation of the needle
            :param weights: (Default = 1) the reliability weights for each of the active areas
            :param scalef: (Default = 1) the scaling for the cost function
            :param arg_check: (Default = False) whether to check if the arguments are valid
            :param continuous: (Default = False) whether to perform continuous integration or not

        Return:
            :return: cost for single bend and single-layer shape sensing

    """
    # argument checking
    if arg_check:
        pass
    # if

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

    # determine k0 and k0prime
    if s_crit < 0:  # single-layer
        k0, k0prime = intrinsics.SingleBend.k0_1layer( s, kc, L, return_callable=continuous )

    else:  # double-bend
        k0, k0prime = intrinsics.DoubleBend.k0_1layer( s, kc, L, s_crit, return_callable=continuous )

    # perform integration to get wv
    if continuous:
        w0 = lambda s: np.append( k0( s ), [ 0, 0 ] )
        w0prime = lambda s: np.append( k0prime( s ), [ 0, 0 ] )

        _, _, wv = numerical.integrateEP_w0_ode( w_init, w0, w0prime, B, s, s0=0, ds=ds, R_init=R_init, Binv=Binv,
                                                 arg_check=arg_check, wv_only=True )

    # if
    else:
        w0 = np.hstack( (k0.reshape( -1, 1 ), np.zeros( (N, 2) )) )
        w0prime = np.hstack( (k0prime.reshape( -1, 1 ), np.zeros( (N, 2) )) )

        _, _, wv = numerical.integrateEP_w0( w_init, w0, w0prime, B, s0=0, ds=ds, R_init=R_init, Binv=Binv,
                                             arg_check=arg_check, wv_only=True )

    # else

    # determine AA locations
    s_m_idx = np.argwhere( s_m.reshape( -1, 1 ) == s.ravel() )[ :, 1 ]

    # compute cost
    cost = scalef * curvature_cost( data, wv, s_m_idx, weights=weights )

    return cost


# doublebend_singlelayer_cost


def curvature_cost( data: np.ndarray, wv: np.ndarray, s_m_idx: np.ndarray, weights: np.ndarray = None ):
    """ Calculate the curvature cost"""
    if weights is None:
        weights = np.ones( data.shape[ 0 ] )
    # if

    else:
        weights = weights[ -data.shape[ 0 ]: ]

    # else

    weights = weights / np.sum( weights )

    # calculate the error
    delta = wv[ s_m_idx, 0:2 ] - data[ :, 0:2 ]
    delta_w = delta * np.reshape( weights, (-1, 1) )
    cost = np.trace( delta_w @ delta_w.T )

    return float( cost )

# curvature_cost
