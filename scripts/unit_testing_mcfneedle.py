import os

import numpy as np

from needle_shape_sensing.sensorized_needles import FBGNeedle, MCFNeedle
from needle_shape_sensing.shape_sensing import ShapeSensingFBGNeedle, ShapeSensingMCFNeedle


def test_ssmcfneedle():
    mcfneedle = MCFNeedle.load_json(
            os.path.join(
                    "data",
                    "needle_params_MCF-4CH-4AA-0001.json"
            )
    )

    ss_mcfneedle = ShapeSensingMCFNeedle.from_MCFNeedle( mcfneedle )

    print( repr( ss_mcfneedle ) )


def test_needle_load():
    fbgneedle_json = os.path.join( "data", "needle_params_2022-10-10_Jig-Calibration_all_weights.json" )
    mcfneedle_json = os.path.join( "data", "needle_params_MCF-4CH-4AA-0001.json" )

    fbgneedle = FBGNeedle.load_json( fbgneedle_json )
    mcfneedle = FBGNeedle.load_json( mcfneedle_json )

    ss_fbgneedle = ShapeSensingFBGNeedle.from_FBGNeedle( fbgneedle )
    ss_mcfneedle = ShapeSensingFBGNeedle.from_FBGNeedle( mcfneedle )

    print( "type of fbgneedle:", type( fbgneedle ) )
    print( "type of mcfneedle:", type( mcfneedle ) )

    print( "type of ss_fbgneedle", type( ss_fbgneedle ) )
    print( "type of ss_mcfneedle", type( ss_mcfneedle ) )

    ss_fbgneedle_fromjson = ShapeSensingFBGNeedle.load_json( fbgneedle_json )
    ss_mcfneedle_fbg_fromjson = ShapeSensingFBGNeedle.load_json( mcfneedle_json )
    ss_mcfneedle_mcf_fromjson = ShapeSensingMCFNeedle.load_json( mcfneedle_json )

    print( "type of ss_fbgneedle_fromjson", type( ss_fbgneedle_fromjson ) )
    print( "type of ss_mcfneedle_fbg_fromjson", type( ss_mcfneedle_fbg_fromjson ) )
    print( "type of ss_mcfneedle_mcf_fromjson", type( ss_mcfneedle_mcf_fromjson ) )


def test_mcfneedle_curvatures():
    mcfneedle = MCFNeedle.load_json(
            os.path.join(
                    "data",
                    "needle_params_MCF-4CH-4AA-0001.json"
            )
    )

    cal_mats = dict( (s_l, np.ones( (2, mcfneedle.num_channels - 1) )) for s_l in mcfneedle.sensor_location )
    mcfneedle.set_calibration_matrices( cal_mats )
    signals = np.stack(
            (np.zeros( mcfneedle.num_signals ), np.ones( mcfneedle.num_signals )),
            axis=0
    )
    curvatures = mcfneedle.curvatures_processed( signals )

    print( mcfneedle )
    print( "Signals:\n", signals )
    print( "Curvatures:\n", curvatures )


def main():
    fns = [
            test_ssmcfneedle,
            test_needle_load,
            test_mcfneedle_curvatures,
    ]

    for fn in fns:
        print()
        print( "Testing:", fn.__qualname__ )
        fn()
        print()
        print(80*'=')


    # for


# main

if __name__ == "__main__":
    main()

# if __main__