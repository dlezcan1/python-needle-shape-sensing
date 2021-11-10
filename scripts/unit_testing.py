import numpy as np
from needle_shape_sensing.shape_sensing import ShapeSensingFBGNeedle
from needle_shape_sensing.intrinsics import SHAPETYPE

def main():
    ss_fbgneedle = ShapeSensingFBGNeedle.load_json('needle_params.json')
    print(ss_fbgneedle)
    print()

    ss_fbgneedle.current_depth = 20
    ref = np.random.randint(1500, 1510, size=(200, ss_fbgneedle.num_channels * ss_fbgneedle.num_activeAreas))
    wls = np.random.randint(1500, 1510, size=(200, ss_fbgneedle.num_channels * ss_fbgneedle.num_activeAreas))

    ss_fbgneedle.update_wavelengths(ref, reference=True)

    # test straight needle insertion (Constant Curvature)
    print('Constant Curvature')
    print("Updated: ", ss_fbgneedle.update_shapetype( SHAPETYPE.CONSTANT_CURVATURE ) )
    ps, Rs = ss_fbgneedle.get_needle_shape()
    print(ps.shape, Rs.shape)
    print()

    # test straight needle insertion (Single bend Single layer)
    print( 'Single-Bend Single Layer' )
    print("Updated: ", ss_fbgneedle.update_shapetype( SHAPETYPE.SINGLEBEND_SINGLELAYER ) )
    ps, Rs = ss_fbgneedle.get_needle_shape(0.002)
    print( ps.shape, Rs.shape )
    print()

    # test straight needle insertion (Single bend Double layer)
    print( 'Single-Bend Double Layer' )
    print("Updated: ", ss_fbgneedle.update_shapetype( SHAPETYPE.SINGLEBEND_DOUBLELAYER, ss_fbgneedle.current_depth//2 ) )
    print(ss_fbgneedle.insertion_parameters)
    ps, Rs = ss_fbgneedle.get_needle_shape( 0.002, 0.003 )
    print( ps.shape, Rs.shape )
    print()

    # test straight needle insertion (Double bend Single layer)
    print( 'Single-Bend Double Layer' )
    print( "Updated: ",
           ss_fbgneedle.update_shapetype( SHAPETYPE.DOUBLEBEND_SINGLELAYER, ss_fbgneedle.current_depth//2 ) )
    print( ss_fbgneedle.insertion_parameters )
    ps, Rs = ss_fbgneedle.get_needle_shape( 0.002 )
    print( ps.shape, Rs.shape )
    print()

    print("Curved needle:")
    ss_fbgneedle.update_wavelengths(wls, temp_comp=True)
    ss_fbgneedle.current_depth = 125
    # test straight needle insertion (Single bend Single layer)
    print( 'Single-Bend Single Layer' )
    print( "Updated: ", ss_fbgneedle.update_shapetype( SHAPETYPE.SINGLEBEND_SINGLELAYER ) )
    ps, Rs = ss_fbgneedle.get_needle_shape( 0.002 )
    print( ps.shape, Rs.shape )
    print()

    # test straight needle insertion (Single bend Double layer)
    print( 'Single-Bend Double Layer' )
    print( "Updated: ",
           ss_fbgneedle.update_shapetype( SHAPETYPE.SINGLEBEND_DOUBLELAYER, ss_fbgneedle.current_depth//2 ) )
    print( ss_fbgneedle.insertion_parameters )
    ps, Rs = ss_fbgneedle.get_needle_shape( 0.002, 0.003 )
    print( ps.shape, Rs.shape )
    print()

    # test straight needle insertion (Double bend Single layer)
    print( 'Single-Bend Double Layer' )
    print( "Updated: ",
           ss_fbgneedle.update_shapetype( SHAPETYPE.DOUBLEBEND_SINGLELAYER, ss_fbgneedle.current_depth//2 ) )
    print( ss_fbgneedle.insertion_parameters )
    ps, Rs = ss_fbgneedle.get_needle_shape( 0.002 )
    print( ps.shape, Rs.shape )
    print()



# main

if __name__ == "__main__":
    main()

# if __main__