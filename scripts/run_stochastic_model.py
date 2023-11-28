import argparse as ap
import logging
import os

import numpy as np
import pandas as pd

from needle_shape_sensing.intrinsics import (
    SHAPETYPE,
    ShapeModelParameters,
)
from needle_shape_sensing.sensorized_needles import (
    Needle,
    FBGNeedle,
)
from needle_shape_sensing.stochastic import (
    StochasticShapeModel,
)

def __parse_args(args=None):
    parser = ap.ArgumentParser(
        "Script to run the stochastic modelling"
    )

    parser.add_argument(
        "--needle-json",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration",
    )

    parser.add_argument(
        "--odir",
        type=str,
        default=None,
        required=False,
        help="The output directory to store the results",
    )

    arggrp_shp_params = parser.add_argument_group("Shape Parameters")
    arggrp_shp_params.add_argument(
        "--kc",
        type=float,
        default=0.002,
        required=False,
    )

    arggrp_shp_params.add_argument(
        "--winit",
        type=float,
        nargs=3,
        default=None,
        required=False,
    )

    arggrp_mdl_params = parser.add_argument_group("Model Parameters")
    arggrp_mdl_params.add_argument(
        "--insertion-depth",
        type=float,
        required=True,
        help="The simulated insertion depth (mm)",
    )
    arggrp_mdl_params.add_argument(
        "--ds",
        type=float,
        default=0.5,
        required=False,
        help="Arclength resolution (mm)",
    )
    
    arggrp_mdl_params.add_argument(
        "--dw",
        type=float,
        default=0.002,
        required=False,
        help="Local deformation resolution (1/mm)",
    )

    arggrp_mdl_params.add_argument(
        "--w-bounds",
        type=float,
        nargs=2,
        default=(-0.05, 0.05),
        required=False,
        help="Simluated bounds of local curvature for wx, wy, wz"
    )

    arggrp_mdl_params.add_argument(
        "--std-curvature",
        type=float,
        default=0.0005,
        required=False,
        help="The standard deviation of curvature random error (1/mm)",
    )

    arggrp_mdl_params.add_argument(
        "--std-fbg-measurement",
        type=float,
        default=None,
        required=False,
        help="The standard deviation of FBG curvature estimation",
    )

    # parse the arguments
    ARGS = parser.parse_args(args)

    # post-process arguments
    if ARGS.winit is None:
        ARGS.winit = [ARGS.kc, 0, 0]

    if ARGS.std_fbg_measurement is None:
        ARGS.std_fbg_measurement = 2 * ARGS.std_curvature

    if ARGS.odir is not None:
        device = "gpu" if ARGS.gpu else "cpu"
        ARGS.odir = f"{ARGS.odir}-{device}"

    # if

    logging.getLogger().setLevel(logging.INFO)

    return ARGS

# __parse_args

def main(args=None):
    ARGS = __parse_args(args)

    fbgneedle = FBGNeedle.load_json(ARGS.needle_json)

    stochastic_model = StochasticShapeModel(
        needle=fbgneedle,
        shape_mdlp=ShapeModelParameters(
            kc_1=ARGS.kc,
            w_init=np.asarray(ARGS.winit),
            shape_type=SHAPETYPE.SINGLEBEND_SINGLELAYER,
        ),
        insertion_depth=ARGS.insertion_depth,
        ds=ARGS.ds,
        dw=ARGS.dw,
        w_bounds=ARGS.w_bounds,
        sigma_curvature=ARGS.std_curvature,
        use_cuda=ARGS.gpu,
    )
    
    stochastic_model.init_probability(
        w_init=np.asarray(ARGS.winit),
    )

    logging.log(logging.INFO, "Beginning to solve stochastic needle shape")
    solved_distribution = stochastic_model.solve(progress_bar=True)
    logging.log(logging.INFO, "Completed solving stochastic needle shape")

    if ARGS.odir is not None:
        os.makedirs(ARGS.odir, exist_ok=True)

        # save the curvature distribution
        ofile = os.path.join(ARGS.odir, "curvature_distribution.npz"),
        np.savez(ofile, solved_distribution)
        logging.log(logging.INFO, f"Saved curvature distribution to: {ofile}")

        # save the statistics
        ofile = os.path.join(ARGS.odir, "benchmark_results.csv")
        pd.DataFrame.from_dict(
            {
                "total elapsed time (secs)"   : [stochastic_model._timer.total_elapsed_time.total_seconds()],
                "average time per loop (secs)": [stochastic_model._timer.averaged_dt.total_seconds()],
            }
        ).to_csv(ofile)

    # if

# main

if __name__ == "__main__":
    main()

# if __main__