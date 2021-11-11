"""

Library of needle shape sensing intrinic measurement functions

Author: Dimitri Lezcano

"""

from enum import Enum, Flag, auto
from typing import Union

import numpy as np

from . import numerical, geometry


class SHAPETYPE( Flag ):
    # first byte is for the number of bends, the second byte is for the number of layers ( - 1)
    CONSTANT_CURVATURE = 0x00
    SINGLEBEND_SINGLELAYER = 0x01
    SINGLEBEND_DOUBLELAYER = 0x02
    DOUBLEBEND_SINGLELAYER = 0x10 | SINGLEBEND_SINGLELAYER


# enum class: SHAPETYPES

class ConstantCurvature:
    @staticmethod
    def k0( s: np.ndarray, kc: float ):
        """
            Intrinsic curvatures of the constant curvature needle

            :param s: numpy array of the arclengths
            :param kc: intrinsic curvature constant

            :returns: (k0(s), k0'(s))
        """
        k0 = kc * np.ones_like( s )
        k0prime = np.zeros_like( k0 )

        return k0, k0prime

    # k0

    @staticmethod
    def w0( s: np.ndarray, kc: float, thetaz: float = 0 ):
        """
            Intrinsic curvatures of the constant curvature needle

            :param s: numpy array of the arclengths
            :param kc: intrinsic curvature constant
            :param thetaz: (Default = 0) the angle of rotation in the xy plane

            :returns: (k0(s), k0'(s))
        """
        k0, k0prime = ConstantCurvature.k0( s, kc )

        w0 = np.hstack( (k0.reshape( -1, 1 ), np.zeros( (k0.size, 2) )) )  # N x 3
        w0prime = np.hstack( (k0prime.reshape( -1, 1 ), np.zeros( (k0prime.size, 2) )) )  # N x 3

        Rz = geometry.rotz( thetaz )
        w0 = w0 @ Rz.T  # === Rz @ w0.T
        w0prime = w0prime @ Rz.T  # === Rz @ w0prime.T

        return w0, w0prime

    # w0

    @staticmethod
    def shape( s: np.ndarray, curvature: Union[ float, np.ndarray ], thetaz: float = 0 ):
        """ Determine the (3D) constant curvature shape"""
        if isinstance( curvature, (int, float) ):
            curvature, _ = ConstantCurvature.w0( s, curvature, thetaz=thetaz )

        # if
        else:
            curvature = curvature.reshape( -1, 3 ).repeat( s.size, axis=0 ) @ geometry.rotz( thetaz ).T

        # else

        pmat, Rmat = numerical.integratePose_wv( curvature, s=s )

        return pmat, Rmat

    # shape


# class: ConstantCurvature

class SingleBend:
    @staticmethod
    def k0_1layer( s: np.ndarray, kc: float, length: float ) -> (np.ndarray, np.ndarray):
        """
        Intrinsics curvatures of the double layer insertion scenario
        
        Original Author (MATLAB): Jin Seob Kim
        Translated Author: Dimitri Lezcano
        
        :param s: numpy array of the arclengths
        :param kc: intrinsic curvature constant
        :param length: length of needle insertion
        :returns: (k0(s), k0'(s)) numpy arrays of shape s.shape
        """
        k0 = kc * (1 - s / length) ** 2
        k0prime = -2 * kc / length * (1 - s / length)

        return k0, k0prime

    # k0_1layer

    @staticmethod
    def k0_2layer( s: np.ndarray, kc_1: float, kc_2: float, length: float, s_crit: float ):
        """
        Intrinsics curvatures of the double layer insertion scenario
        
        Original Author (MATLAB): Jin Seob Kim
        Translated Author: Dimitri Lezcano
        
        :param s: numpy array of the arclengths
        :param kc_1: intrinsic curvature constant for layer 1
        :param kc_2: intrinsic curvature constant for layer 2
        :param length: length of needle insertion
        :param s_crit: arclength where needle shape boundary is
        
        :returns: (k0(s), k0'(s)) numpy arrays of shape s.shape
        """
        # set-up
        k0 = np.zeros_like( s )
        k0prime = np.zeros_like( s )

        # first layer processing
        s_1 = s[ s <= s_crit ]
        k0_1 = kc_1 * ((s_crit - s_1) / length) ** 2 + kc_2 * (1 - s_crit / length) * (
                1 + s_crit / length - 2 * s_1 / length)
        k0prime_1 = -2 * kc_1 / length ** 2 * (s_crit - s_1) - 2 * kc_2 / length * (1 - s_crit / length)

        # second layer processing
        s_2 = s[ s > s_crit ]
        k0_2 = kc_2 * (1 - s_2 / length) ** 2
        k0prime_2 = -2 * kc_2 / length * (1 - s_2 / length)

        # set the return values
        k0[ s <= s_crit ] = k0_1
        k0[ s > s_crit ] = k0_2

        k0prime[ s <= s_crit ] = k0prime_1
        k0prime[ s > s_crit ] = k0prime_2

        return k0, k0prime

    # k0_2layer

    @staticmethod
    def k0_3layer( s: np.ndarray, kc_1: float, kc_2: float, kc_3: float, length: float, s_crit_1: float,
                   s_crit_2: float ):
        """
        Intrinsics curvatures of the double layer insertion scenario

        Original Author (MATLAB): Jin Seob Kim
        Translated Author: Dimitri Lezcano

        :param s: numpy array of the arclengths
        :param kc_1: intrinsic curvature constant for layer 1
        :param kc_2: intrinsic curvature constant for layer 2
        :param kc_3: intrinsic curvature constant for layer 3
        :param length: length of needle insertion
        :param s_crit_1: arclength where first needle shape boundary is
        :param s_crit_2: arclength where second needle shape boundary is

        :returns: (k0(s), k0'(s)) numpy arrays of shape s.shape
        """
        # set-up
        k0 = np.zeros_like( s )
        k0prime = np.zeros_like( s )

        # first layer processing
        mask_1 = s <= s_crit_1
        s_1 = s[ mask_1 ]
        # k0_1 = kc_1 * ((s_crit_1 - s_1) / length) ** 2 + kc_2 * (s_crit_2 - s_crit_1) / length * (
        #         s_crit_2 + s_crit_1)
        k0_1 = kc_1 * (s_crit_1 - s_1) ** 2 / length ** 2 + kc_2 * (s_crit_2 - s_crit_1) * (
                s_crit_2 + s_crit_1 - 2 * s_1) / length ** 2 + kc_3 * (1 - s_crit_2 / length) * (
                       1 + (s_crit_2 - 2 * s_1) / length)

        k0prime_1 = -2 * kc_1 / length ** 2 * (s_crit_1 - s_1) - 2 * kc_2 / length ** 2 * (
                s_crit_2 - s_crit_1) - 2 * kc_3 / length * (1 - s_crit_2 / length)

        # second layer processing
        mask_2 = (s_crit_1 < s) & (s <= s_crit_2)
        s_2 = s[ mask_2 ]
        k0_2 = kc_2 * (s_crit_2 - s_2) ** 2 / length ** 2 + kc_3 * (1 - s_crit_2 / length) * (
                1 + s_crit_2 / length - 2 * s_2 / length)
        k0prime_2 = -2 * kc_2 / length ** 2 * (s_crit_2 - s_2) - 2 * kc_3 / length * (1 - s_crit_2 / length)

        # third layer processing
        mask_3 = s > s_crit_2
        s_3 = s[ mask_3 ]
        k0_3 = kc_3 * (1 - s_3 / length) ** 2
        k0prime_3 = -2 * kc_3 / length * (1 - s_3 / length)

        # set the return values
        k0[ mask_1 ] = k0_1
        k0[ mask_2 ] = k0_2
        k0[ mask_3 ] = k0_3

        k0prime[ mask_1 ] = k0prime_1
        k0prime[ mask_2 ] = k0prime_2
        k0prime[ mask_3 ] = k0prime_3

        return k0, k0prime

    # k0_3layer

    @staticmethod
    def determine_2layer_boundary( kc1: float, length: float, z_crit, B: np.ndarray, w_init: np.ndarray = None,
                                   s0: float = 0, ds: float = 0.5, R_init: np.ndarray = np.eye( 3 ),
                                   Binv: np.ndarray = None ):
        """
            Determine the length of the needle that is inside the first layer only (s_crit)

            Original Author (MATLAB): Jin Seob Kim
            Translated Author: Dimitri Lezcano

            :param kc1: intrinsic curvature constant for layer 1
            :param s0: start position of s0
            :param length: length of needle insertion
            :param B: the needle stiffness matrix
            :param z_crit: length of the first layer
            :param w_init: (Default = [kc1; 0; 0]) the initial angular deformation
            :param s0: (Default = 0) a float of the inital length to integrate
            :param ds: (Default = 0.5) a float tof the ds to integrate
            :param R_init: (Default = 3x3 identity) SO3 matrix for initial rotation angle
            :param Binv: (Default = None) inv(B) Can be provided for numerical efficiency

            :returns: s_crit: the critical arclength (rounded to the resolution of the arclength's ds)
                                (-1 if not in second-layer yet)
        """
        # w_init check
        if w_init is None:
            w_init = np.array( [ kc1, 0, 0 ] )

        # if

        # compute w0 and w0prime
        s = np.arange( s0, length + ds, ds )
        k0, k0prime = SingleBend.k0_1layer( s, kc1, length )

        w0 = np.hstack( (k0.reshape(-1,1), np.zeros( (k0.size, 2) )) )
        w0prime = np.hstack( (k0prime.reshape(-1,1), np.zeros( (k0prime.size, 2) )) )

        # compute position of single-layer approximation
        pmat_single, *_ = numerical.integrateEP_w0( w_init, w0, w0prime, B, s=s, R_init=R_init, Binv=Binv,
                                                    arg_check=False )

        # determine the point closest (but after) the boundary
        dz = pmat_single[ :, 2 ] - z_crit
        if np.all( dz <= 0 ):
            s_crit = -1  # not in double-layer yet

        # if
        else:
            s_crit_idx = np.argmin( np.abs( dz ) )
            s_crit = s[ s_crit_idx ]

        # else

        return s_crit

    # determine_2layer_boundary


# class: SingleBend

class DoubleBend:
    @staticmethod
    def k0_1layer( s: np.ndarray, kc: float, length: float, s_crit: float, p: float = 2 / 3 ):
        """
            Intrinsics curvatures of the double layer insertion scenario

            Original Author (MATLAB): Jin Seob Kim
            Translated Author: Dimitri Lezcano

            :param s: numpy array of the arclengths
            :param kc: intrinsic curvature constant
            :param length: length of needle insertion
            :param s_crit: the 180 degree turn insertion depth
            :param p: (Default = 2/3) float of the kappa_c scaling parameter
            :returns: (k0(s), k0'(s)) numpy arrays of shape s.shape
        """
        # arclength setups (before & after double-bend)
        s1 = s[ s <= s_crit ]
        s2 = s[ s >= s_crit ]

        # kappa_c values
        kc1 = kc * ((s1.max() - s1.min()) / length) ** p
        kc2 = kc * ((s2.max() - s2.min()) / length) ** p

        # kappa_0 calculations
        k0_1 = kc1 * (1 - s1 / length) ** 2
        k0_2 = -kc2 * (1 - s2 / length) ** 2
        k0_12 = 1 / 2 * (k0_1[ -1 ] + k0_2[ 0 ])

        k0 = np.hstack( (k0_1[ :-1 ], k0_12, k0_2[ 1: ]) )

        # kappa_0' calculations
        k0prime_1 = -2 * kc1 / length * (1 - s1 / length)
        k0prime_2 = -2 * kc1 / length * (1 - s2 / length)
        k0prime_12 = 1 / 2 * (k0prime_1[ -1 ] + k0prime_2[ 0 ])

        k0prime = np.hstack( (k0prime_1[ :-1 ], k0prime_12, k0prime_2[ 1: ]) )

        return k0, k0prime

    # k0_1layer

# class: DoubleBend
