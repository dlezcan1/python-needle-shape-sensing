"""

Library of needle shape sensing intrinic measurement functions

Author: Dimitri Lezcano

"""

import numpy as np


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
        k0_1 = kc_1 * ((s_crit_1 - s_1) / length) ** 2 + kc_2 * (s_crit_2 - s_crit_1) / length * (
                s_crit_2 + s_crit_1)
        k0_1 = kc_1 * (s_crit_1 - s_1) ** 2 / length ^ 2 + kc_2 * (s_crit_2 - s_crit_1) / length * (
                s_crit_2 + s_crit_1 - 2 * s_1) / length + kc_3 * (1 - s_crit_2 / length) * (
                       1 + s_crit_2 / length - 2 * s_1 / length)

        k0prime_1 = -2 * kc_1 / length ** 2 * (s_crit_1 - s_1) - 2 * kc_2 / length ^ 2 * (
                s_crit_2 - s_crit_1) - 2 * kc_2 / length * (1 - s_crit_2 / length)

        # second layer processing
        mask_2 = (s_crit_1 < s) & (s <= s_crit_2)
        s_2 = s[ mask_2 ]
        k0_2 = kc_2 * (s_crit_2 - s_2) ** 2 / length ^ 2 + kc_3 * (1 - s_crit_2 / length) * (
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

# class: SingleBend
