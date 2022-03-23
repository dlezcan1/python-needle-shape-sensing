"""

Library of needle shape sensing intrinic measurement functions

Author: Dimitri Lezcano

"""

from enum import Flag
from typing import Union, Callable

import numpy as np

from . import numerical, geometry


class SHAPETYPE( Flag ):
    # first byte is for the number of bends, the second byte is for the number of layers ( - 1)
    CONSTANT_CURVATURE = 0x00
    SINGLEBEND_SINGLELAYER = 0x01
    SINGLEBEND_DOUBLELAYER = 0x02
    DOUBLEBEND_SINGLELAYER = 0x10 | SINGLEBEND_SINGLELAYER


# enum class: SHAPETYPES

class AirDeflection:
    @staticmethod
    def shape( s: np.ndarray, insertion_point: np.ndarray, cubic_fit: bool = False ):
        """
            Get the shape from in-air deflection.

            :param s: Arclengths of the needle to consider (cut-off so within insertion point region)
            :param insertion_point: the point (relative to needle base coordinate frame (0,0,0) where insertion is
            :param cubic_fit: (bool, Default=False), whether to use cubic fitting or rollback to quadratic fit.

            :return: numpy array of 3D points of the needle deflected in air.

        """
        if cubic_fit:  # mechanical modelling fit
            points = AirDeflection.shape_cubic( s, insertion_point )

        else:  # quadratic fit
            points = AirDeflection.shape_quadratic( s, insertion_point )

        return points

    # shape

    @staticmethod
    def shape_cubic( s: np.ndarray, insertion_point: np.ndarray ):
        """
            Get the shape from in-air deflection using cubic fit from mechanical modelling.

            :param s: Arclengths of the needle to consider (cut-off so within insertion point region)
            :param insertion_point: the point (relative to needle base coordinate frame (0,0,0) where insertion is

            :return: numpy array of 3D points of the needle deflected in air.

        """
        # length of entry point approximation # TODO: update for more accurate modelling
        L_entry = np.linalg.norm( insertion_point )

        # fit beam mechanics-based cubic polynomial based on insertion point
        z_ins = insertion_point[ 2 ]
        a_xy = insertion_point[ 0:2 ] / (z_ins ** 3 - 3 * L_entry * z_ins ** 2)

        # generate z-poitns
        s = np.unique( s )
        dz = np.min( np.abs( np.diff( s ) ) )
        z = np.linspace( 0, z_ins, int( z_ins // dz ) + 1 )

        # compute points along z-axis
        points_z = np.hstack( (a_xy.reshape( 1, -1 ) * (z ** 3 - 3 * L_entry * z_ins ** 2).reshape( -1, 1 ),
                               z.reshape( -1, 1 )) )

        L, _ = geometry.arclength( points_z, axis=0 )

        # interpolate to get s
        s = s[ s <= L ]  # cut-off extrapolated features
        points, _ = numerical.interpolate_curve_s( points_z, s, axis=0 )

        return points

    # shape_cubic

    @staticmethod
    def shape_quadratic( s: np.ndarray, insertion_point: np.ndarray ):
        """
            Get the shape from in-air deflection using quadratic fit.

            :param s: Arclengths of the needle to consider (cut-off so within insertion point region)
            :param insertion_point: the point (relative to needle base coordinate frame (0,0,0) where insertion is

            :return: numpy array of 3D points of the needle deflected in air.

        """
        # fit parabola based on insertion point
        z_ins = insertion_point[ 2 ]
        a_xy = insertion_point[ 0:2 ] / z_ins ** 2
        a_xy_norm = np.linalg.norm( a_xy )

        # determine z-coordinates to use based on arclengths
        if a_xy_norm > 0:
            s_tot = (z_ins * np.sqrt( 1 + 4 * (a_xy_norm ** 2) * (z_ins ** 2) ) + np.arcsinh(
                    2 * a_xy_norm * z_ins ) / (
                             2 * a_xy_norm)) / 2

        # if
        else:  # straight needle
            s_tot = z_ins

        # else

        s = np.unique( s[ s <= s_tot ] )  # remove longer and duplicate and sort

        dz = np.min( np.abs( np.diff( s ) ) )
        z = np.linspace( 0, z_ins, int( z_ins // dz + 1 ) )

        # compute 3D points
        points_z = np.hstack( (a_xy.reshape( 1, -1 ) * z.reshape( -1, 1 ) ** 2, z.reshape( -1, 1 )) )

        # interpolate to get s
        points, _ = numerical.interpolate_curve_s( points_z, s, axis=0 )

        return points

    # shape_quadratic


# class: AirDeflection

class ConstantCurvature:
    @staticmethod
    def k0( s: np.ndarray, kc: float, return_callable: bool = False ):
        """
            Intrinsic curvatures of the constant curvature needle

            :param s: numpy array of the arclengths
            :param kc: intrinsic curvature constant
            :param return_callable: (Default = False) returns the callable function

            :returns: (k0(s), k0'(s))
        """
        if return_callable:
            k0 = lambda s: kc
            k0prime = lambda s: 0

        # if
        else:
            k0 = kc * np.ones_like( s )
            k0prime = np.zeros_like( k0 )

        # else

        return k0, k0prime

    # k0

    @staticmethod
    def w0( s: np.ndarray, kc: float, thetaz: float = 0, return_callable: bool = False ):
        """
            Intrinsic curvatures of the constant curvature needle

            :param s: numpy array of the arclengths
            :param kc: intrinsic curvature constant
            :param thetaz: (Default = 0) the angle of rotation in the xy plane
            :param return_callable: (Default = False) returns the callable function

            :returns: (k0(s), k0'(s))
        """
        k0, k0prime = ConstantCurvature.k0( s, kc, return_callable=return_callable )

        # rotate the curvature
        Rz = geometry.rotz( thetaz )
        v = np.array( [ 1, 0, 0 ] ) @ Rz.T

        if return_callable:
            w0 = lambda s: k0( s ) * v
            w0prime = lambda s: k0prime( s ) * v

        # if
        else:
            w0 = k0 * v
            w0prime = k0prime * v

        # else

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
    def k0_1layer( s: np.ndarray, kc: float, length: float, return_callable: bool = False ) -> (
            Union[ np.ndarray, Callable ], Union[ np.ndarray, Callable ]):
        """
        Intrinsics curvatures of the double layer insertion scenario
        
        Original Author (MATLAB): Jin Seob Kim
        Translated Author: Dimitri Lezcano
        
        :param s: numpy array of the arclengths
        :param kc: intrinsic curvature constant
        :param length: length of needle insertion
        :param return_callable: (Default = False) returns the callable function

        :returns: (k0(s), k0'(s)) numpy arrays of shape s.shape
        """
        if return_callable:
            k0 = lambda s: kc * (1 - s / length) ** 2
            k0prime = lambda s: -2 * kc / length * (1 - s / length)

        # if
        else:
            k0 = kc * (1 - s / length) ** 2
            k0prime = -2 * kc / length * (1 - s / length)

        # else
        return k0, k0prime

    # k0_1layer

    @staticmethod
    def k0_2layer( s: np.ndarray, kc_1: float, kc_2: float, length: float, s_crit: float,
                   return_callable: bool = False ):
        """
        Intrinsics curvatures of the double layer insertion scenario
        
        Original Author (MATLAB): Jin Seob Kim
        Translated Author: Dimitri Lezcano
        
        :param s: numpy array of the arclengths
        :param kc_1: intrinsic curvature constant for layer 1
        :param kc_2: intrinsic curvature constant for layer 2
        :param length: length of needle insertion
        :param s_crit: arclength where needle shape boundary is
        :param return_callable: (Default = False) returns the callable function
        
        :returns: (k0(s), k0'(s)) numpy arrays of shape s.shape
        """
        if return_callable:
            def k0( s ):
                if s <= s_crit:
                    k0_s = kc_1 * ((s_crit - s) / length) ** 2 + kc_2 * (1 - s_crit / length) * (
                            1 + s_crit / length - 2 * s / length)

                # if
                else:
                    k0_s = kc_2 * (1 - s / length) ** 2

                # else

                return k0_s

            # k0

            def k0prime( s ):
                if s <= s_crit:
                    k0prime_s = -2 * kc_1 / length ** 2 * (s_crit - s) - 2 * kc_2 / length * (1 - s_crit / length)

                # if
                else:
                    k0prime_s = -2 * kc_2 / length * (1 - s / length)

                # else

                return k0prime_s

            # k0prime

        # if
        else:
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

        # else

        return k0, k0prime

    # k0_2layer

    @staticmethod
    def k0_3layer( s: np.ndarray, kc_1: float, kc_2: float, kc_3: float, length: float, s_crit_1: float,
                   s_crit_2: float, return_callable: bool = False ):
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
        :param return_callable: (Default = False) returns the callable function

        :returns: (k0(s), k0'(s)) numpy arrays of shape s.shape
        """
        if return_callable:
            def k0( s ):
                if s <= s_crit_1:
                    k0_s = kc_1 * (s_crit_1 - s) ** 2 / length ** 2 + kc_2 * (s_crit_2 - s_crit_1) * (
                            s_crit_2 + s_crit_1 - 2 * s) / length ** 2 + kc_3 * (1 - s_crit_2 / length) * (
                                   1 + (s_crit_2 - 2 * s) / length)

                # if
                elif s <= s_crit_2:
                    k0_s = kc_2 * (s_crit_2 - s) ** 2 / length ** 2 + kc_3 * (1 - s_crit_2 / length) * (
                            1 + s_crit_2 / length - 2 * s / length)

                # elif
                else:
                    k0_s = kc_3 * (1 - s / length) ** 2

                # else

                return k0_s

            # k0

            def k0prime( s ):
                if s <= s_crit_1:
                    k0prime_s = -2 * kc_1 / length ** 2 * (s_crit_1 - s) - 2 * kc_2 / length ** 2 * (
                            s_crit_2 - s_crit_1) - 2 * kc_3 / length * (1 - s_crit_2 / length)

                # if
                elif s <= s_crit_2:
                    k0prime_s = -2 * kc_2 / length ** 2 * (s_crit_2 - s) - 2 * kc_3 / length * (1 - s_crit_2 / length)

                # elif
                else:
                    k0prime_s = -2 * kc_3 / length * (1 - s / length)

                # else

                return k0prime_s

            # k0prime

        # if
        else:
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

        # else

        return k0, k0prime

    # k0_3layer

    @staticmethod
    def determine_2layer_boundary( kc1: float, length: float, z_crit, B: np.ndarray, w_init: np.ndarray = None,
                                   s0: float = 0, ds: float = 0.5, R_init: np.ndarray = np.eye( 3 ),
                                   Binv: np.ndarray = None, needle_rotations: list = None, continuous: bool = False ):
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
            :param needle_rotations: (Default = None) list of needle axis rotations in radians
            :param continuous: (Default = False) whether to perform continuous integration

            :returns: s_crit: the critical arclength (rounded to the resolution of the arclength's ds)
                                (-1 if not in second-layer yet)
        """
        # w_init check
        if w_init is None:
            w_init = np.array( [ kc1, 0, 0 ] )

        # if

        s = np.arange( s0, length + ds, ds )

        # compute position of single-layer approximation
        if continuous:
            k0 = lambda s: SingleBend.k0_1layer( s, kc1, length )[ 0 ]
            k0prime = lambda s: SingleBend.k0_1layer( s, kc1, length )[ 1 ]

            w0 = lambda s: np.append( k0( s ), [ 0, 0 ] )
            w0prime = lambda s: np.append( k0prime( s ), [ 0, 0 ] )

            pmat_single, *_ = numerical.integrateEP_w0_ode( w_init, w0, w0prime, B, s=s, R_init=R_init, Binv=Binv,
                                                            arg_check=False, needle_rotations=needle_rotations )
        # if
        else:
            k0, k0prime = SingleBend.k0_1layer( s, kc1, length )

            w0 = np.hstack( (k0.reshape( -1, 1 ), np.zeros( (k0.size, 2) )) )
            w0prime = np.hstack( (k0prime.reshape( -1, 1 ), np.zeros( (k0prime.size, 2) )) )

            pmat_single, *_ = numerical.integrateEP_w0( w_init, w0, w0prime, B, s=s, R_init=R_init, Binv=Binv,
                                                        arg_check=False, needle_rotations=needle_rotations )

        # else

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
    def k0_1layer( s: np.ndarray, kc: float, length: float, s_crit: float, p: float = 2 / 3,
                   return_callable: bool = False ):
        """
            Intrinsics curvatures of the double layer insertion scenario

            Original Author (MATLAB): Jin Seob Kim
            Translated Author: Dimitri Lezcano

            :param s: numpy array of the arclengths
            :param kc: intrinsic curvature constant
            :param length: length of needle insertion
            :param s_crit: the 180 degree turn insertion depth
            :param p: (Default = 2/3) float of the kappa_c scaling parameter
            :param return_callable: (Default = False) returns the callable function

            :returns: (k0(s), k0'(s)) numpy arrays of shape s.shape
        """
        if return_callable:
            # scale kappa_c values
            kc1 = kc * (s_crit / length) ** p
            kc2 = kc * (1 - s_crit / length) ** p

            def k0( s ):
                if s < s_crit:
                    k0_s = kc1 * (1 - s / length) ** 2

                elif s == s_crit:
                    k0_s = 1 / 2 * (kc1 - kc2) * (1 - s / length) ** 2

                else:
                    k0_s = -kc2 * (1 - s / length) ** 2

                return k0_s

            # k0

            def k0prime( s ):
                if s < s_crit:
                    k0prime_s = -2 * kc1 / length * (1 - s1 / length)

                elif s == s_crit:
                    k0prime_s = -(kc1 - kc2) * (1 - s / length)

                else:
                    k0prime_s = -2 * kc2 / length * (1 - s2 / length)

                return k0prime_s

            # k0prime
        # if
        else:
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
            k0prime_2 = -2 * kc2 / length * (1 - s2 / length)
            k0prime_12 = 1 / 2 * (k0prime_1[ -1 ] + k0prime_2[ 0 ])

            k0prime = np.hstack( (k0prime_1[ :-1 ], k0prime_12, k0prime_2[ 1: ]) )

        # else

        return k0, k0prime

    # k0_1layer

# class: DoubleBend
