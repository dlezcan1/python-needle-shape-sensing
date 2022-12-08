"""

Library of needle shape sensing intrinic measurement functions

Author: Dimitri Lezcano

"""

from enum import Flag

import torch

from needle_shape_sensing.pytorch import numerical, geometry


class SHAPETYPE( Flag ):
    # first byte is for the number of bends, the second byte is for the number of layers ( - 1)
    CONSTANT_CURVATURE = 0x00
    SINGLEBEND_SINGLELAYER = 0x01
    SINGLEBEND_DOUBLELAYER = 0x02
    DOUBLEBEND_SINGLELAYER = 0x10 | SINGLEBEND_SINGLELAYER

    def get_k0( self ):
        """ Get the intrinsics kappa 0 function

        """
        k0_fns = {
                # SHAPETYPE.CONSTANT_CURVATURE    : ConstantCurvature.k0,
                SHAPETYPE.SINGLEBEND_SINGLELAYER: SingleBend.k0_1layer,
                SHAPETYPE.SINGLEBEND_DOUBLELAYER: SingleBend.k0_2layer,
                SHAPETYPE.DOUBLEBEND_SINGLELAYER: DoubleBend.k0_1layer,
        }

        return k0_fns[ self ]

    # get_k0

    def get_shape_class( self ):
        """ Get the intrinsics shape class

        """
        classes = {
                # SHAPETYPE.CONSTANT_CURVATURE    : ConstantCurvature,
                SHAPETYPE.SINGLEBEND_SINGLELAYER: SingleBend,
                SHAPETYPE.SINGLEBEND_DOUBLELAYER: SingleBend,
                SHAPETYPE.DOUBLEBEND_SINGLELAYER: DoubleBend,
        }

        return classes[ self ]

    # get_shape_class

    def k0( self, *args, **kwargs ):
        """ Get kappa_0 """
        return self.get_k0()( *args, **kwargs )

    # k0

    def w0( self, *args, **kwargs ):
        """ Get omega_o intrinsic 3D curvature function """
        k0 = self.k0( *args, **kwargs )
        if callable( k0 ):
            w0 = lambda s: torch.tensor( [ k0( s ), 0, 0 ], dtype=k0( s ).dtype )

        # if
        else:
            w0 = torch.cat(
                    (k0, torch.zeros( (2, k0.shape[ 0 ]), dtype=k0.dtype )),
                    dim=0
            )

        # else

        return w0

    # w0


# enum class: SHAPETYPES

class SingleBend:
    @staticmethod
    def k0_1layer( s: torch.Tensor, kc: float, length: float, return_callable: bool = False ):
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
            seq_mask = s <= length
            k0       = (kc * (1 - s / length) ** 2) * seq_mask
            k0prime  = (-2 * kc / length * (1 - s / length)) * seq_mask

        # else
        return k0, k0prime

    # k0_1layer

    @staticmethod
    def k0_2layer(
            s: torch.Tensor, kc_1: float, kc_2: float, length: float, s_crit: float,
            return_callable: bool = False
    ):
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
            # first layer processing
            s_1 = s  # torch.masked_select( s, s <= s_crit )
            k0_1 = kc_1 * ((s_crit - s_1) / length) ** 2 + kc_2 * (1 - s_crit / length) * (
                    1 + s_crit / length - 2 * s_1 / length)
            k0prime_1 = -2 * kc_1 / length ** 2 * (s_crit - s_1) - 2 * kc_2 / length * (1 - s_crit / length)

            # second layer processing
            s_2 = s  # torch.masked_select( s, s > s_crit )
            k0_2 = kc_2 * (1 - s_2 / length) ** 2
            k0prime_2 = -2 * kc_2 / length * (1 - s_2 / length)

            # set the return values
            # k0 = torch.cat( (k0_1, k0_2), dim=0 )
            # k0prime = torch.cat( (k0prime_1, k0prime_2), dim=0 )
            seq_mask = s < length
            k0       = (k0_1 * (s <= s_crit) + k0_2 * (s > s_crit)) * seq_mask
            k0prime  = (k0prime_1 * (s <= s_crit) + k0prime_2 * (s > s_crit)) * seq_mask

        # else

        return k0, k0prime

    # k0_2layer

    @staticmethod
    def k0_3layer(
            s: torch.Tensor, kc_1: float, kc_2: float, kc_3: float, length: float, s_crit_1: float,
            s_crit_2: float, return_callable: bool = False
    ):
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

            # first layer processing
            s_1 = torch.masked_select( s, s <= s_crit_1 )
            # k0_1 = kc_1 * ((s_crit_1 - s_1) / length) ** 2 + kc_2 * (s_crit_2 - s_crit_1) / length * (
            #         s_crit_2 + s_crit_1)
            k0_1 = kc_1 * (s_crit_1 - s_1) ** 2 / length ** 2 + kc_2 * (s_crit_2 - s_crit_1) * (
                    s_crit_2 + s_crit_1 - 2 * s_1) / length ** 2 + kc_3 * (1 - s_crit_2 / length) * (
                           1 + (s_crit_2 - 2 * s_1) / length)

            k0prime_1 = -2 * kc_1 / length ** 2 * (s_crit_1 - s_1) - 2 * kc_2 / length ** 2 * (
                    s_crit_2 - s_crit_1) - 2 * kc_3 / length * (1 - s_crit_2 / length)

            # second layer processing
            s_2 = torch.masked_select( s, (s_crit_1 < s) & (s <= s_crit_2) )
            k0_2 = kc_2 * (s_crit_2 - s_2) ** 2 / length ** 2 + kc_3 * (1 - s_crit_2 / length) * (
                    1 + s_crit_2 / length - 2 * s_2 / length)
            k0prime_2 = -2 * kc_2 / length ** 2 * (s_crit_2 - s_2) - 2 * kc_3 / length * (1 - s_crit_2 / length)

            # third layer processing
            s_3 = torch.masked_select( s, s > s_crit_2 )
            k0_3 = kc_3 * (1 - s_3 / length) ** 2
            k0prime_3 = -2 * kc_3 / length * (1 - s_3 / length)

            # set the return values
            k0 = torch.cat( (k0_1, k0_2, k0_3), dim=0 )
            k0prime = torch.cat( (k0prime_1, k0prime_2, k0prime_3), dim=0 )

        # else

        return k0, k0prime

    # k0_3layer


# class: SingleBend

class DoubleBend:
    @staticmethod
    def k0_1layer(
            s: torch.Tensor, kc: float, length: float, s_crit: float, p: float = 2 / 3,
            return_callable: bool = False
    ):
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
                    k0prime_s = -2 * kc1 / length * (1 - s / length)

                elif s == s_crit:
                    k0prime_s = -(kc1 - kc2) * (1 - s / length)

                else:
                    k0prime_s = -2 * kc2 / length * (1 - s / length)

                return k0prime_s

            # k0prime
        # if
        else:
            # arclength setups (before & after double-bend)
            # s1 = s # torch.masked_select( s, s <= s_crit )
            # s2 = s # torch.masked_select( s, s >= s_crit )

            # kappa_c values
            kc1 = kc * ((torch.max( s[ s <= s_crit ] ) - torch.min( s[ s <= s_crit ] )) / length) ** p
            kc2 = kc * ((torch.max( s[ s >= s_crit ] ) - torch.min( s[ s >= s_crit ] )) / length) ** p

            # kappa_0 calculations
            k0_1 = kc1 * (1 - s / length) ** 2
            k0_2 = -kc2 * (1 - s / length) ** 2
            k0_12 = 1 / 2 * (k0_1 + k0_2)

            # kappa_0' calculations
            k0prime_1 = -2 * kc1 / length * (1 - s / length)
            k0prime_2 = -2 * kc2 / length * (1 - s / length)
            k0prime_12 = 1 / 2 * (k0prime_1[ -1 ] + k0prime_2[ 0 ])

            # concatenation
            seq_mask = s < length
            k0       = (k0_1 * (s < s_crit) + k0_2 * (s > s_crit) + k0_12 * (s == s_crit)) * seq_mask
            k0prime  = (k0prime_1 * (s < s_crit) + k0prime_2 * (s > s_crit) + k0prime_12 * (s == s_crit)) * seq_mask

            # k0 = torch.cat(
            #         (
            #                 k0_1[ :-1 ],
            #                 torch.tensor( [ k0_12 ], dtype=k0_1.dtype, device=k0_1.device ),
            #                 k0_2[ 1: ]
            #         ),
            #         dim=0
            # )
            # k0prime = torch.cat(
            #         (
            #                 k0prime_1[ :-1 ],
            #                 torch.tensor( [ k0prime_12 ], dtype=k0_1.dtype, device=k0prime_1.device ),
            #                 k0prime_2[ 1: ]
            #         ),
            #         dim=0
            # )

        # else

        return k0, k0prime

    # k0_1layer

# class: DoubleBend
