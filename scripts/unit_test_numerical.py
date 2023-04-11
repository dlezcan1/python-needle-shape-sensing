import json
import numpy as np

import matplotlib.pyplot as plt

from needle_shape_sensing import (
    numerical,
    intrinsics,
)

from needle_shape_sensing.shape_sensing import ShapeSensingFBGNeedle

def load_shape_matfile(matlab_file):
    with open(matlab_file, 'r') as data:
        shape_info = json.load(data)

    # with

    for k, v in shape_info.items():
        if isinstance(v, dict):
            for k2, v2 in shape_info[k].items():
                if isinstance(v, list):
                    shape_info[k][k2] = np.asarray(v2)

                # if
            # for
        # if
        elif isinstance(v, list):
            shape_info[k] = np.asarray(v)

        # elif

    # for

    return shape_info

# load_shape_matlab

def check_numerical_integrations(results_json_file):
    """ Check whether integrations are accurate"""
    print("Testing Numerical integration")
    matlab_shape_info = load_shape_matfile( results_json_file )

    ss_needle = ShapeSensingFBGNeedle.load_json("data/needle_params_2022-10-10_Jig-Calibration_all_weights.json")

    # unpack matlab shape-sensing
    ml_pmat = np.asarray(matlab_shape_info['shape']['pos']).T
    ml_wv   = np.asarray(matlab_shape_info['shape']['wv']).T
    ml_Rmat = np.rollaxis(np.asarray(matlab_shape_info['shape']['Rmat']), 2)

    # unpack and prepare data
    B = np.asarray(matlab_shape_info['needle_mech_params']['B'])
    kc = matlab_shape_info['kc']
    w_init = np.asarray(matlab_shape_info['winit'])
    L = matlab_shape_info['L']
    ds = 0.5
    s = np.arange( 0, L + ds, ds )

    # Test shape integration equations
    k0, k0prime = intrinsics.SingleBend.k0_1layer(
        s, kc, L, return_callable=False
    )
    w0 = np.hstack( (k0.reshape( -1, 1 ), np.zeros( (k0.size, 2) )) )
    w0prime = np.hstack( (k0prime.reshape( -1, 1 ), np.zeros( (k0prime.size, 2) )) )

    py_pmat, py_Rmat, py_wv = numerical.integrateEP_w0(
            w_init, w0, w0prime, B, s0=0, ds=ds
    )

    # compare shape stuff
    dev_wv = lambda a, b: np.linalg.norm(a - b, ord=2, axis=1)
    dev_pmat = dev_wv
    dev_Rmat = lambda a, b: np.linalg.norm( a - b, ord=2, axis=(1, 2))
    dev_Rmat_inv = lambda a, b: np.linalg.det(
        a @ np.swapaxes(b, 1, 2)
    )

    d_wv = dev_wv(py_wv, ml_wv)
    d_Rmat = dev_Rmat( py_Rmat, ml_Rmat )
    d_Rmat_inv = dev_Rmat_inv(py_Rmat, ml_Rmat)
    d_pmat = dev_pmat(py_pmat, ml_pmat)

    print("wv deviations:\n", d_wv)
    print()

    print("d_Rmat inverse:\n", d_Rmat_inv)
    print()

    print("d_pmat: (Max: {}, Mean: {}):\n{}".format(
            d_pmat.max(),
            d_pmat.mean(),
            d_pmat
    ))
    print()

    print("d_pmat: Max z: {} | Max x: {} | Max y: {}".format(
            *np.max(np.abs(py_pmat - ml_pmat), axis=0)
            ))

    print((py_pmat - ml_pmat).round(3))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(py_pmat[:, 2], py_pmat[:, 1], 'r*-', )
    ax1.plot( ml_pmat[ :, 2 ], ml_pmat[ :, 1 ], 'g*-', )
    ax1.legend(["Python", "Matlab"])
    ax1.set_xlabel("Z [mm]")
    ax1.set_ylabel("Y [mm]")

    ax2.plot( py_pmat[ :, 2 ], py_pmat[ :, 0 ], 'r*-', )
    ax2.plot( ml_pmat[ :, 2 ], ml_pmat[ :, 0 ], 'g*-', )
    ax2.legend( [ "Python", "Matlab" ] )
    ax2.set_xlabel( "Z [mm]" )
    ax2.set_ylabel( "X [mm]" )

    plt.show()


# check_numerical_integration

def test_simpson_vec_int(data_file):
    print("Testing simpson_vec_int")
    with open(data_file, 'r') as data:
        json_data = json.load(data)
        X = np.asarray(json_data['X'])
        dx = json_data['dx']
        ml_res = np.asarray(json_data['result']).ravel()

    # with

    py_res = numerical.simpson_vec_int(X, dx)
    print(py_res - ml_res)

def test_ideal_insertion():
    print("Testing ideal insertion shapes")

    kc = 0.000282
    winit = np.array([kc, 0, 0])
    winit = np.array([
        0.000242,
        -6.32e-4,
        -1.0e-5,
    ])
    L = 200
    ds = 0.5
    s = np.arange(0, L+ds, ds)
    B = np.diag((1, 1, 2)) * 1e5

    k0, k0prime = intrinsics.SingleBend.k0_1layer(s, kc, L, return_callable=False)
    w0 = np.hstack((k0[:, np.newaxis], np.zeros((k0.shape[0], 2))))
    w0prime = np.hstack( (k0prime[ :, np.newaxis ], np.zeros( (k0.shape[ 0 ], 2) )) )

    print(k0.shape, k0prime.shape)
    print(w0.shape, w0prime.shape)

    pmat, Rmat, wv = numerical.integrateEP_w0(
            winit,
            w0,
            w0prime,
            B,
            s0=0,
            ds=ds,
    )
    print(pmat[-1])
    print()

# test_ideal_insertion

def test_jig_insertion():
    print("Testing jig insertion")

    ss_needle = ShapeSensingFBGNeedle.load_json("data/needle_params_2022-10-10_Jig-Calibration_all_weights.json")
    ss_needle.update_shapetype(intrinsics.SHAPETYPE.CONSTANT_CURVATURE)

    curvature = [1/500, 0]
    L = 151

    ss_needle.ref_wavelengths = np.ones_like(ss_needle.num_signals)
    ss_needle.current_curvatures = np.tile(curvature, (ss_needle.num_aa, 1)).T
    ss_needle.current_depth = L

    pmat, Rmat = ss_needle.get_needle_shape()
    print("winit = ", ss_needle.current_winit)
    print("kc = ", ss_needle.current_kc)
    print("pmat tip = ", pmat[-1])
    print("Rmat base = \n", Rmat[1])
    print()

# test_jig_insertion


if __name__ == "__main__":
    # check_numerical_integrations("data/shape_stuff.json")
    # test_simpson_vec_int("data/simpson_vec_int.json")
    test_ideal_insertion()
    test_jig_insertion()

# if __main__