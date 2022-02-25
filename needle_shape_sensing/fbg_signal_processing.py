"""
Created on Jul 28, 2021

@author: Dimitri Lezcano

@summary: This is a library for processing FBG signals

"""
import numpy as np


def process_signals( signals:np.ndarray, ref_wavelengths:np.ndarray ) -> np.ndarray:
    """ Remove reference wavelengths from the signals

    """

    return signals - ref_wavelengths.reshape(1,-1)


# process_signals

def temperature_compensation( signal_shifts, num_channels: int, num_active_areas: int ):
    """ Perform temperature compensation over the signal shits for FBG signals

        signal_shifts: numpy array of size (N, num_channels*num_active_areas)
        num_channels: int of the number of channels for the FBG needle
        num_active_areas: int of the number of active areas for the FBG needle
    """

    active_area_assignments = list( range( 1, num_active_areas + 1 ) ) * num_channels
    active_area_assignments = np.array( active_area_assignments )

    # iterate over the active areas removing temperature compensation
    signal_shifts_tempcomp = signal_shifts.copy()
    for aa_i in range( 1, num_active_areas + 1 ):
        aa_i_mask = active_area_assignments == aa_i

        signal_shifts_aa_i_mean = np.mean( signal_shifts[ :, aa_i_mask ], axis=1, keepdims=True )
        signal_shifts_tempcomp[ :, aa_i_mask ] -= signal_shifts_aa_i_mean  # remove mean value

    # aa_i

    return signal_shifts_tempcomp

# temperature_compensation
