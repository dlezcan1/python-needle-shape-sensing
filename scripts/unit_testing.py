from timeit import timeit

import numpy as np

from needle_shape_sensing.shape_sensing import ShapeSensingFBGNeedle
from needle_shape_sensing.intrinsics import SHAPETYPE, SingleBend
from needle_shape_sensing.cost_functions import singlebend_singlelayer_cost
from needle_shape_sensing.numerical import integrateEP_w0_ode, integratePose_wv


def main():
    ss_fbgneedle = ShapeSensingFBGNeedle.load_json( 'needle_params.json' )
    ss_fbgneedle.continuous_integration = True
    print( ss_fbgneedle )
    print()

    ss_fbgneedle.current_depth = 122.2
    ref = np.random.randint( 1500, 1510, size=(200, ss_fbgneedle.num_channels * ss_fbgneedle.num_activeAreas) )
    wls = np.random.randint( 1500, 1510, size=(200, ss_fbgneedle.num_channels * ss_fbgneedle.num_activeAreas) )
    wls = ref + np.random.randn( *ref.shape ) / 100

    ss_fbgneedle.update_wavelengths( ref, reference=True )

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
    avgtimeit = lambda f, number: timeit( f, number=int( number ) ) / int( number )
    B = ss_fbgneedle.B
    Binv = np.linalg.inv( B )

    cost_fn = lambda: singlebend_singlelayer_cost( 1e-3 * np.ones( 4 ), ss_fbgneedle.current_curvatures.T,
                                                   np.array( ss_fbgneedle.sensor_location_tip ),
                                                   ss_fbgneedle.ds, B, Binv=Binv,
                                                   L=ss_fbgneedle.current_depth, continuous=True )
    timeavg_cost = avgtimeit( cost_fn, int( 1e3 ) )  # ~ 0.0009-0.0034 seconds/loop
    print( f"Average time to evaluate cost function: {timeavg_cost} seconds/loop" )

    # Prof. Kim integration
    ds = 0.5
    s = np.arange( 0, 130 + ds, ds )
    k0, k0prime = SingleBend.k0_1layer( s, 1e-9, np.max( s ) )
    w0 = np.hstack( (k0.reshape( -1, 1 ), np.zeros( (k0.size, 2) )) )
    w0prime = np.hstack( (k0prime.reshape( -1, 1 ), np.zeros( (k0prime.size, 2) )) )

    # integration_fn = lambda: integrateEP_w0(np.zeros(3), w0, w0prime, B=B, ds=ds, Binv=Binv,wv_only=True)
    # N_loops = 1e3
    # timeavg_integration = timeit(integration_fn, number=int(N_loops)) # slow! ~ 0.024 seconds/loop
    #
    # print( f"Average time to evaluate integration function: {timeavg_integration/N_loops} seconds/loop" )
    # print()

    # scipy integration
    k0, k0prime = SingleBend.k0_1layer( s, 1e-4, np.max( s ), return_callable=True )
    w0 = lambda s: np.append( k0( s ), [ 0, 0 ] )
    w0prime = lambda s: np.append( k0prime( s ), [ 0, 0 ] )

    integration_fn2 = lambda: integrateEP_w0_ode( np.zeros( 3 ), w0, w0prime, B=B, s=s, ds=ds, Binv=Binv, wv_only=True )
    timeavg_integration2 = avgtimeit( integration_fn2, int( 1e3 ) )  # ~0.0012 seconds with cts integration

    print(
            f"Average time to evaluate scipy version of integration function: {timeavg_integration2} seconds/loop" )
    print()

    # integratePose_wv
    integratePose_fn = lambda: integratePose_wv(np.zeros((s.shape[0],3)), s=s)
    timeavg_integratePose = avgtimeit(integratePose_fn, int(1e2)) # ~0.03 seconds

    print(f"Average time to integrate needle pose: {timeavg_integratePose} seconds/loop")
    print()

    # shape-sensing
    print( ss_fbgneedle.current_curvatures )
    ss_fbgneedle.optimizer.options.update( { 'tol'    : 1e-4,
                                             'options': { 'maxiter': 20 }  # about 0.14 s
                                             } )
    ss_fn = lambda: ss_fbgneedle.get_needle_shape( 1e-4, np.zeros( 3 ) )
    timeavg_ss = avgtimeit( ss_fn, int( 1e2 ) )
    print( f"Average time to determine shape of needle function: {timeavg_ss} seconds/loop" )
    print()

    return

    # test straight needle insertion (Single bend Double layer)
    print( 'Single-Bend Double Layer' )
    print( "Updated: ",
           ss_fbgneedle.update_shapetype( SHAPETYPE.SINGLEBEND_DOUBLELAYER, ss_fbgneedle.current_depth // 2 ) )
    print( ss_fbgneedle.insertion_parameters )
    ps, Rs = ss_fbgneedle.get_needle_shape( 0.000, 0.000 )
    print( ps.shape, Rs.shape, ss_fbgneedle.current_kc, ss_fbgneedle.current_winit )
    print()

    # test straight needle insertion (Double bend Single layer)
    print( 'Single-Bend Double Layer' )
    print( "Updated: ",
           ss_fbgneedle.update_shapetype( SHAPETYPE.DOUBLEBEND_SINGLELAYER, ss_fbgneedle.current_depth // 2 ) )
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
    print( "Updated: ",
           ss_fbgneedle.update_shapetype( SHAPETYPE.SINGLEBEND_DOUBLELAYER, ss_fbgneedle.current_depth // 2 ) )
    print( ss_fbgneedle.insertion_parameters )
    ps, Rs = ss_fbgneedle.get_needle_shape( 0.002, 0.001 )
    print( ps.shape, Rs.shape, ss_fbgneedle.current_kc, ss_fbgneedle.current_winit )
    print()

    # test straight needle insertion (Double bend Single layer)
    print( 'Single-Bend Double Layer' )
    print( "Updated: ",
           ss_fbgneedle.update_shapetype( SHAPETYPE.DOUBLEBEND_SINGLELAYER, ss_fbgneedle.current_depth // 2 ) )
    print( ss_fbgneedle.insertion_parameters )
    ps, Rs = ss_fbgneedle.get_needle_shape( 0.002 )
    print( ps.shape, Rs.shape, ss_fbgneedle.current_kc, ss_fbgneedle.current_winit )
    print()


# main

if __name__ == "__main__":
    main()

# if __main__
