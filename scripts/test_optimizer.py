import numpy as np
import scipy.optimize

from needle_shape_sensing.intrinsics import SingleBend
from needle_shape_sensing.sensorized_needles import FBGNeedle
from needle_shape_sensing.numerical import NeedleParamOptimizations, integrateEP_w0


def main():
    fbg_file = "D:/git/amiro/data/3CH-4AA-0004/needle_params.json"
    ds = 0.5
    fbg_needle = FBGNeedle.load_json( fbg_file )
    optimizer = NeedleParamOptimizations( fbg_needle, ds=ds )

    # actual parameters
    kc_act = 0.002
    w_init_act = np.array( [ 0.005, 0.003, -0.001 ] )
    R_init = np.eye( 3 )
    L = 110
    N = int( L / ds ) + 1
    s = ds * np.arange( N )
    s_meas = L - np.array(fbg_needle.sensor_location_tip)
    s_idx_meas = np.argwhere( s_meas.reshape( -1, 1 ) == s )[ :, 1 ]
    print(fbg_needle.weights)

    # get the measurements
    k0, k0_prime = SingleBend.k0_1layer( s, kc_act, L )
    w0 = np.hstack( (k0.reshape( -1, 1 ), np.zeros( (len( k0 ),2) )) )
    w0_prime = np.hstack( (k0_prime.reshape( -1, 1 ), np.zeros( (len( k0_prime ),2) )) )

    # - integrate and get the shape
    _, _, wv = integrateEP_w0( w_init_act, w0, w0_prime, ds=ds, B=fbg_needle.B, R_init=np.eye( 3 ) )

    curv_meas = wv[ s_idx_meas, 0:2 ]  # actual measurements

    # perform the optimization
    kc_0 = kc_act / 2
    w_init_0 = 0*np.array( [ -0.0005, 0.0005, 0 ] )

    # - bounds
    lb = -0.01 * np.ones( 4 )
    ub = 0.01 * np.ones( 4 )
    bnds = scipy.optimize.Bounds(lb, ub, keep_feasible=True)

    kc, w_init, res = optimizer.singlebend_singlelayer_k0( kc_0, w_init_0, curv_meas, L, R_init=R_init,
                                                           bounds=bnds, tol=1e-8 )

    print( f"Actual: kc = {kc_act} | w_init = {w_init_act}" )
    print( f"Optimized: kc = {kc} | w_init = {w_init}" )
    print( f"Error: kc = {kc - kc_act} | w_init = {w_init - w_init_act}" )
    # print(res)


# main

if __name__ == "__main__":
    main()

# if __main__
