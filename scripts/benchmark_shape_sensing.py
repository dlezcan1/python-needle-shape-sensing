import os
import random

import numpy as np

from needle_shape_sensing import (
    shape_sensing,
)

NUM_TRIALS = 20

def test_single_layer(fbg_needle: shape_sensing.ShapeSensingFBGNeedle):
    global NUM_TRIALS

    fbg_needle.update_wavelengths(
        np.random.randn(200, fbg_needle.num_signals).astype(np.float64),
        reference=False,
        temp_comp=True,
        processed=True,
    )

    dL               = 30
    insertion_depths = np.arange(0, fbg_needle.length + dL, dL, dtype=np.float64)

    for _ in range(NUM_TRIALS):
        kc_i       = random.random()*0.005
        winit_i    = np.asarray([kc_i, 0, 0], dtype=np.float64)

        for depth in insertion_depths:
            fbg_needle.current_depth = depth
            
            pmat, Rmat = fbg_needle.get_needle_shape(kc_i, winit_i)

        # for

    # for


# test_single_layer

def main():
    fbg_needle = shape_sensing.ShapeSensingFBGNeedle.load_json(os.path.join(
        "data",
        "needle_params_7CH-4AA-0001-MCF-even_2023-03-29_Jig-Calibration_clinically-relevant-2_weighted.json"
    ))

    fbg_needle.update_wavelengths(
        np.zeros((fbg_needle.num_signals), dtype=np.float64),
        reference=True,
    )

    test_single_layer(fbg_needle)


# main

if __name__ == "__main__":
    main()

# if __main__