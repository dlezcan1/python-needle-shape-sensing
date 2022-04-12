from timeit import timeit

import numpy as np

from needle_shape_sensing.shape_sensing import ShapeSensingFBGNeedle
from needle_shape_sensing.intrinsics import SHAPETYPE, SingleBend
from needle_shape_sensing.cost_functions import singlebend_singlelayer_cost
from needle_shape_sensing.numerical import integrateEP_w0, integrateEP_w0_ode, integratePose_wv

avgtimeit = lambda f, number: timeit( f, number=int( number ) ) / int( number )


def integration_stability_test( ss_fbgneedle, methods: list, kc_l: list, w_init_l: list ):
    valid_methods = [ "odeint", "RK23", "RK45", "LSODA" ]

    # parameter updates
    B = ss_fbgneedle.B
    Binv = np.linalg.inv( B )
    ds = 0.5
    s = np.arange( 0, 130 + ds, ds )

    # integration functions
    integration_wv_fn = lambda int_method, w0, w0prime, w_init: integrateEP_w0_ode(
            w_init, w0, w0prime, B=B, s=s, ds=ds, Binv=Binv, wv_only=True,
            integration_method=int_method )[ 2 ]
    integration_wv_fn_gt = lambda w0, w0prime, w_init: integrateEP_w0(
            w_init, w0, w0prime, B=B, s=s, ds=ds, Binv=Binv, wv_only=True )[ 2 ]
    integration_pos_fn = lambda wv: integratePose_wv( wv, s, s.min(), ds )[ 0 ]  # pmat

    # iterate through the methods
    for method in methods:
        if method not in valid_methods:
            continue

        for kc, w_init in zip( kc_l, w_init_l ):
            k0v, k0primev = SingleBend.k0_1layer( s, kc, s.max(), return_callable=False )
            k0, k0prime = SingleBend.k0_1layer( s, kc, s.max(), return_callable=True )
            w0 = lambda s: np.append( k0( s ), [ 0, 0 ] )
            w0prime = lambda s: np.append( k0prime( s ), [ 0, 0 ] )
            w0v = np.hstack( (k0v.reshape( -1, 1 ), np.zeros( (len( k0v ), 2) )) )
            w0primev = np.hstack( (k0primev.reshape( -1, 1 ), np.zeros( (len( k0primev ), 2) )) )

            # get ground truth
            wv_gt = integration_wv_fn_gt( w0v, w0primev, w_init )
            pmat_gt = integration_pos_fn( wv_gt )

            # perform continuous integration
            wv_test = integration_wv_fn( method, w0, w0prime, w_init )
            pmat_test = integration_pos_fn( wv_test )

            # perform error computations
            # - test maximum curvature for wv
            wv_gt_max = np.linalg.norm( wv_gt, ord=2, axis=1 ).max()
            wv_test_max = np.linalg.norm( wv_test, ord=2, axis=1 ).max()

            # - perform maximum curvature error
            wv_err_max = np.linalg.norm( wv_gt - wv_test, ord=2, axis=1 ).max()

            # - test tip location deflection
            tip_defl_gt = np.linalg.norm( pmat_gt[ 0:2, -1 ], ord=2 )
            tip_defl_test = np.linalg.norm( pmat_test[ 0:2, -1 ], ord=2 )

            # - compute shape deviation between GT and test
            pmat_err_max = np.linalg.norm( pmat_gt - pmat_test, ord=2, axis=1 ).max()

            # summarize results
            print( f"Method: {method} | kc: {kc} 1/mm | w_init: {w_init} 1/mm" )
            print(
                    f" Curvature: Max GT = {wv_gt_max:.4f} 1/mm | Max Test = {wv_test_max:.4f} 1/mm |"
                    f" Max Error = {wv_err_max:.4f} 1/mm" )
            print( f" Tip Deflection: GT = {tip_defl_gt:.4f} mm | Test = {tip_defl_test:.4f} mm" )
            print( f" Maximum Shape error = {pmat_err_max:.4f} mm " )

        # for: kc, w_init
        print()
    # for: method


# integration_stability_tst

def integration_speed_test(
        ss_fbgneedle, methods: list, kc_l: list, w_init_l: list, N_trials: int = 1e3 ):
    # parameter updates
    B = ss_fbgneedle.B
    Binv = np.linalg.inv( B )
    ds = 0.5
    s = np.arange( 0, 130 + ds, ds )

    # integration function
    integration_fn = lambda int_method, w0, w0prime, w_init: integrateEP_w0_ode(
            w_init, w0, w0prime, B=B, s=s, ds=ds, Binv=Binv, wv_only=True,
            integration_method=int_method )
    integration_fn_gt = lambda w0, w0prime, w_init: integrateEP_w0(
            w_init, w0, w0prime, B=B, s=s, ds=ds, Binv=Binv, wv_only=True )

    for method in methods:
        for kc, w_init in zip( kc_l, w_init_l ):
            if method == "discrete":
                k0, k0prime = SingleBend.k0_1layer( s, kc, s.max(), return_callable=False )
                w0 = lambda s: np.append( k0( s ), [ 0, 0 ] )
                w0 = np.hstack( (k0.reshape( -1, 1 ), np.zeros( (len( k0 ), 2) )) )
                w0prime = np.hstack(
                        (k0prime.reshape( -1, 1 ), np.zeros( (len( k0prime ), 2) )) )
                test_fn = lambda: integration_fn_gt( w0, w0prime, w_init )

            # if
            else:
                k0, k0prime = SingleBend.k0_1layer( s, kc, s.max(), return_callable=True )
                w0 = lambda s: np.append( k0( s ), [ 0, 0 ] )
                w0prime = lambda s: np.append( k0prime( s ), [ 0, 0 ] )
                test_fn = lambda: integration_fn( method, w0, w0prime, w_init )

            # else
            tavg_cost = avgtimeit( test_fn, N_trials )
            print(
                    f"Results for '{method:8s}' integration method for kc = {kc:4f},"
                    f" w_init = [ {w_init[ 0 ]:.4f}, {w_init[ 1 ]:.4f}, {w_init[ 2 ]:.4f} | "
                    f"{tavg_cost * 1000:4f} ms/loop"
                    )

        # for: kc, w_init
        print()
    # for: methods


# integration_speed_test

def temperataure_compensation_test( ss_fbgneedle: ShapeSensingFBGNeedle ):
    print( "Testing Temperature compensation function" )

    wls_zeros1 = np.zeros( ss_fbgneedle.num_signals )
    print( f"Testing Zero array of size: {wls_zeros1.shape}" )
    print( "Result = " )
    print( ss_fbgneedle.temperature_compensate( wls_zeros1 ) )
    print()

    wls_zeros2 = np.zeros( (1, ss_fbgneedle.num_signals) )
    print( f"Testing Zero array of size: {wls_zeros2.shape}" )
    print( "Result = " )
    print( ss_fbgneedle.temperature_compensate( wls_zeros2 ) )
    print()

    wls_zeros3 = np.zeros( (6, ss_fbgneedle.num_signals) )
    print( f"Testing Zero array of size: {wls_zeros3.shape}" )
    print( "Result = " )
    print( ss_fbgneedle.temperature_compensate( wls_zeros3 ) )
    print()

    wls = np.arange( ss_fbgneedle.num_activeAreas )[ np.newaxis ]. \
        repeat( ss_fbgneedle.num_channels, axis=0 ).ravel().astype( float )
    print( f"Testing array = \n{wls}" )
    print( "Result = " )
    print( ss_fbgneedle.temperature_compensate( wls ) )
    print()

    wls_mat = wls[ np.newaxis ].repeat( 5, axis=0 )
    print( f"Testing array = \n{wls_mat}" )
    print( "Result = " )
    print( ss_fbgneedle.temperature_compensate( wls_mat ) )
    print()


# temperature_compensation_test


def main():
    ss_fbgneedle = ShapeSensingFBGNeedle.load_json( 'needle_params.json' )
    ss_fbgneedle.continuous_integration = True
    print( ss_fbgneedle )
    print()

    ss_fbgneedle.current_depth = 122.2
    ref = np.random.randint(
            1500, 1510, size=(200, ss_fbgneedle.num_channels * ss_fbgneedle.num_activeAreas) )
    wls = np.random.randint(
            1500, 1510, size=(200, ss_fbgneedle.num_channels * ss_fbgneedle.num_activeAreas) )
    wls = ref + np.random.randn( *ref.shape ) / 100

    ss_fbgneedle.update_wavelengths( ref, reference=True )
    ss_fbgneedle.update_wavelengths( wls )
    temperataure_compensation_test( ss_fbgneedle )

    # test straight needle insertion (Constant Curvature)
    print( 'Constant Curvature' )
    print( "Updated: ", ss_fbgneedle.update_shapetype( SHAPETYPE.CONSTANT_CURVATURE ) )
    ps, Rs = ss_fbgneedle.get_needle_shape()
    print( ps.shape, Rs.shape, ss_fbgneedle.current_kc, ss_fbgneedle.current_winit )
    print()

    # test straight needle insertion (Single bend Single layer)
    print( 'Single-Bend Single Layer' )
    print( "Updated: ", ss_fbgneedle.update_shapetype( SHAPETYPE.SINGLEBEND_SINGLELAYER ) )
    ps, Rs = ss_fbgneedle.get_needle_shape( 0.002 )
    print( ps.shape, Rs.shape, ss_fbgneedle.current_kc, ss_fbgneedle.current_winit )
    print()

    # time straight needle insertion
    print( "Single-Bend Single-Layer Time (Straight needle)" )

    B = ss_fbgneedle.B
    Binv = np.linalg.inv( B )

    # cost_fn = lambda: singlebend_singlelayer_cost(
    #         1e-3 * np.ones( 4 ), ss_fbgneedle.current_curvatures.T,
    #         np.array( ss_fbgneedle.sensor_location_tip ),
    #         ss_fbgneedle.ds, B, Binv=Binv,
    #         L=ss_fbgneedle.current_depth, continuous=True )
    # timeavg_cost = avgtimeit( cost_fn, int( 1e3 ) )  # ~ 0.0009-0.0034 seconds/loop
    # print( f"Average time to evaluate cost function: {timeavg_cost} seconds/loop" )
    #
    # # Prof. Kim integration
    # ds = 0.5
    # s = np.arange( 0, 130 + ds, ds )
    # k0, k0prime = SingleBend.k0_1layer( s, 1e-9, np.max( s ) )
    # w0 = np.hstack( (k0.reshape( -1, 1 ), np.zeros( (k0.size, 2) )) )
    # w0prime = np.hstack( (k0prime.reshape( -1, 1 ), np.zeros( (k0prime.size, 2) )) )

    # integration_fn = lambda: integrateEP_w0(np.zeros(3), w0, w0prime, B=B, ds=ds, Binv=Binv,wv_only=True)
    # N_loops = 1e3
    # timeavg_integration = timeit(integration_fn, number=int(N_loops)) # slow! ~ 0.024 seconds/loop
    #
    # print( f"Average time to evaluate integration function: {timeavg_integration/N_loops} seconds/loop" )
    # print()

    # scipy integration - tests
    kc = [ 1e-4, 1e-3, 3e-3, 5e-3 ]
    w_init = [ np.zeros( 3 ), 0.01 * np.ones( 3 ), kc[ 2 ] * np.array( [ 1, 0, 0 ] ),
               kc[ 3 ] * np.ones( 3 ) ]
    methods = [ 'odeint', 'RK23', 'RK45', 'LSODA', "discrete" ]
    integration_speed_test( ss_fbgneedle, methods, kc, w_init )
    integration_stability_test( ss_fbgneedle, methods, kc, w_init )

    return

    k0, k0prime = SingleBend.k0_1layer( s, 1e-4, np.max( s ), return_callable=True )
    w0 = lambda s: np.append( k0( s ), [ 0, 0 ] )
    w0prime = lambda s: np.append( k0prime( s ), [ 0, 0 ] )

    integration_fn2 = lambda: integrateEP_w0_ode(
            np.zeros( 3 ), w0, w0prime, B=B, s=s, ds=ds, Binv=Binv, wv_only=True,
            integration_method='odeint' )
    timeavg_integration2 = avgtimeit(
            integration_fn2, int( 1e3 ) )  # ~0.0012 seconds with cts integration
    print(
            f"Average time to evaluate scipy version of odeint integration function: {timeavg_integration2} seconds/loop" )
    print()

    integration_fn3 = lambda: integrateEP_w0_ode(
            np.zeros( 3 ), w0, w0prime, B=B, s=s, ds=ds, Binv=Binv, wv_only=True,
            integration_method='RK23' )
    timeavg_integration3 = avgtimeit(
            integration_fn3, int( 1e3 ) )  # ~0.0012 seconds with cts integration
    print(
            f"Average time to evaluate scipy version of RK23 integration function: {timeavg_integration3} seconds/loop" )
    print()

    integration_fn4 = lambda: integrateEP_w0_ode(
            np.zeros( 3 ), w0, w0prime, B=B, s=s, ds=ds, Binv=Binv, wv_only=True,
            integration_method='RK45' )
    timeavg_integration4 = avgtimeit(
            integration_fn4, int( 1e3 ) )  # ~0.0012 seconds with cts integration
    print(
            f"Average time to evaluate scipy version of RK45 integration function: {timeavg_integration4} seconds/loop" )
    print()

    # integratePose_wv
    integratePose_fn = lambda: integratePose_wv( np.zeros( (s.shape[ 0 ], 3) ), s=s )
    timeavg_integratePose = avgtimeit( integratePose_fn, int( 1e2 ) )  # ~0.03 seconds

    print( f"Average time to integrate needle pose: {timeavg_integratePose} seconds/loop" )
    print()

    # shape-sensing
    print( ss_fbgneedle.current_curvatures )
    ss_fbgneedle.optimizer.options.update(
            {
                    'tol'    : 1e-4,
                    'options': { 'maxiter': 20 }  # about 0.14 s
                    } )
    ss_fn = lambda: ss_fbgneedle.get_needle_shape( 1e-4, np.zeros( 3 ) )
    timeavg_ss = avgtimeit( ss_fn, int( 1e2 ) )
    print( f"Average time to determine shape of needle function: {timeavg_ss} seconds/loop" )
    print()

    return

    # test straight needle insertion (Single bend Double layer)
    print( 'Single-Bend Double Layer' )
    print(
            "Updated: ",
            ss_fbgneedle.update_shapetype(
                    SHAPETYPE.SINGLEBEND_DOUBLELAYER, ss_fbgneedle.current_depth // 2 ) )
    print( ss_fbgneedle.insertion_parameters )
    ps, Rs = ss_fbgneedle.get_needle_shape( 0.000, 0.000 )
    print( ps.shape, Rs.shape, ss_fbgneedle.current_kc, ss_fbgneedle.current_winit )
    print()

    # test straight needle insertion (Double bend Single layer)
    print( 'Single-Bend Double Layer' )
    print(
            "Updated: ",
            ss_fbgneedle.update_shapetype(
                    SHAPETYPE.DOUBLEBEND_SINGLELAYER, ss_fbgneedle.current_depth // 2 ) )
    print( ss_fbgneedle.insertion_parameters )
    ps, Rs = ss_fbgneedle.get_needle_shape( 0.002 )
    print( ps.shape, Rs.shape, ss_fbgneedle.current_kc, ss_fbgneedle.current_winit )
    print()

    print( "Curved needle:" )
    ss_fbgneedle.update_wavelengths( wls, temp_comp=True )
    ss_fbgneedle.current_depth = 125
    # test straight needle insertion (Single bend Single layer)
    print( 'Single-Bend Single Layer' )
    print( "Updated: ", ss_fbgneedle.update_shapetype( SHAPETYPE.SINGLEBEND_SINGLELAYER ) )
    ps, Rs = ss_fbgneedle.get_needle_shape( 0.002 )
    print( ps.shape, Rs.shape, ss_fbgneedle.current_kc, ss_fbgneedle.current_winit )
    print()

    # test straight needle insertion (Single bend Double layer)
    print( 'Single-Bend Double Layer' )
    print(
            "Updated: ",
            ss_fbgneedle.update_shapetype(
                    SHAPETYPE.SINGLEBEND_DOUBLELAYER, ss_fbgneedle.current_depth // 2 ) )
    print( ss_fbgneedle.insertion_parameters )
    ps, Rs = ss_fbgneedle.get_needle_shape( 0.002, 0.001 )
    print( ps.shape, Rs.shape, ss_fbgneedle.current_kc, ss_fbgneedle.current_winit )
    print()

    # test straight needle insertion (Double bend Single layer)
    print( 'Single-Bend Double Layer' )
    print(
            "Updated: ",
            ss_fbgneedle.update_shapetype(
                    SHAPETYPE.DOUBLEBEND_SINGLELAYER, ss_fbgneedle.current_depth // 2 ) )
    print( ss_fbgneedle.insertion_parameters )
    ps, Rs = ss_fbgneedle.get_needle_shape( 0.002 )
    print( ps.shape, Rs.shape, ss_fbgneedle.current_kc, ss_fbgneedle.current_winit )
    print()


# main

if __name__ == "__main__":
    main()

# if __main__
