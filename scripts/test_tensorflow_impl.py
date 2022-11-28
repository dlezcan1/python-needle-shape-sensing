import numpy as np
import tensorflow as tf

import needle_shape_sensing as nss
import needle_shape_sensing.tensorflow as nss_tf


def test_integrateEP_w0():
    print( "Test integrateEP_w0" )
    N, M = 10, 5
    ds = 0.5
    w_init = tf.random.normal( (N, 3) )
    w0 = tf.random.normal( (N, M, 3) )
    w0prime = tf.concat( (tf.zeros( (N, 1, 3) ), w0[ :, :-1 ] - w0[ :, 1: ]), axis=1 ) / ds
    B = tf.linalg.diag( tf.range( 1, 4, dtype=w0.dtype ) )
    Binv = tf.linalg.inv( B )
    seq_mask = tf.reduce_all( tf.ones_like( w0 ) > 0, axis=-1 )

    # original solution
    wv_nss = np.zeros_like( w0 )
    for i in range( wv_nss.shape[ 0 ] ):
        _, _, wv_nss[ i ] = nss.numerical.integrateEP_w0(
                w_init[ i ].numpy(), w0[ i ].numpy(), w0prime[ i ].numpy(), B.numpy(), ds=ds, Binv=Binv.numpy(),
        )

    # for

    # tensorflow implementation
    _, _, wv_nss_tf, _ = nss_tf.numerical.integrateEP_w0(
            w_init, w0, w0prime, B, seq_mask, ds, Binv=Binv,
    )

    # comparison
    print( "Original Solution:", wv_nss.shape, "\n", wv_nss[ 0 ] )
    print( "Tensorflow Implementation", wv_nss_tf.shape, "\n", wv_nss_tf.numpy()[ 0 ] )
    print( "Error:\n", np.linalg.norm( wv_nss - wv_nss_tf.numpy(), axis=-1 ) )


# test_integrateEP_w0


def test_integratePose_wv():
    print( "Test integratePose_wv" )
    wv = tf.random.normal( (10, 5, 3), dtype=tf.float64 )
    seq_mask = tf.reduce_all( tf.ones_like( wv ) > 0, axis=-1 )
    ds = 0.5

    # original solution
    pmat_nss = np.zeros_like( wv )
    Rmat_nss = np.zeros( (*wv.shape, 3), dtype=np.float64 )
    for i in range( pmat_nss.shape[ 0 ] ):
        pmat_nss[ i ], Rmat_nss[ i ] = nss.numerical.integratePose_wv( wv[ i ].numpy(), ds=ds )

    # for

    # tensorflow implementation
    pmat_nss_tf, Rmat_nss_tf, seq_mask_tf = nss_tf.numerical.integratePose_wv( wv, seq_mask, ds )

    # comparison
    print( "Position matrix" )
    print( "\tOriginal solution:", pmat_nss.shape, "\n", pmat_nss[ 0 ] )
    print( "\tTensorflow impl.:", pmat_nss_tf.shape, "\n", pmat_nss_tf[ 0 ].numpy() )
    print( "\tError:\n", np.linalg.norm( pmat_nss - pmat_nss_tf.numpy(), axis=-1 ) )
    print( "Rotation Matrix" )
    print( "\tOriginal solution:", Rmat_nss.shape, )
    print( "\tTensorflow impl.:", Rmat_nss_tf.shape, )
    print( "\tError:\n", np.linalg.norm( Rmat_nss - Rmat_nss_tf.numpy(), axis=(2, 3) ) )


# test_integratePose_wv


def test_simpson_vec_int():
    print( "Test simpson_vec_int" )
    f = tf.random.normal( (10, 6, 3), dtype=tf.float32 )
    seq_mask = tf.reduce_all( tf.ones_like( f ) > 0, axis=-1 )
    dx = 0.5

    # original solution
    nss_int = np.zeros( (f.shape[ 0 ], f.shape[ 2 ]), dtype=np.float64 )
    for i in range( nss_int.shape[ 0 ] ):
        nss_int[ i ] = nss.numerical.simpson_vec_int( f[ i ].numpy().T, dx )

    # tensorflow implementation
    nss_tf_int = nss_tf.numerical.simpson_vec_int( f, dx, seq_mask )

    # compare
    print( "Original Solution:", nss_int.shape, "\n", nss_int )
    print( "Tensorflow Impl.:", nss_tf_int.shape, "\n", nss_tf_int.numpy() )
    print( "Error:", nss_int - nss_tf_int.numpy() )


# test_simpson_vec_int

def test_shapes():
    print( "Testing shape constructions" )

    lengths = [ 15.0, 35.0, 95.0, 125.0 ]
    lengths = list( sorted( lengths, reverse=True ) )
    ds = 0.5
    kc = 0.002
    winit = tf.cast(
            tf.repeat(
                    [ [ 0.005, 0.002, 0.001 ] ],
                    len( lengths ),
                    axis=0
            ), dtype=tf.float64
    )
    B = tf.linalg.diag( tf.range( 1, 4, dtype=winit.dtype ) )
    Binv = tf.linalg.inv( B )

    # construct array tensors
    w0_ta = tf.TensorArray(
            winit.dtype,
            size=len( lengths ),
            dynamic_size=False,
            infer_shape=True,
            clear_after_read=False,
    )
    w0prime_ta = tf.TensorArray(
            winit.dtype,
            size=len( lengths ),
            dynamic_size=False,
            infer_shape=True,
            clear_after_read=False,
    )
    seq_mask_ta = tf.TensorArray(
            tf.bool,
            size=len( lengths ),
            dynamic_size=False,
            infer_shape=True,
            clear_after_read=False,
    )

    # create list of needle shapes for numpy version
    nss_shape = [ ]

    # iterate through lengths
    M = -1
    for i, L in enumerate( sorted( lengths, reverse=True ) ):
        s_L = np.arange( 0, L + ds, ds )
        M = max( len( s_L ), M )

        # intrinsics
        k0, k0prime = nss.intrinsics.SingleBend.k0_1layer( s_L, kc, L, return_callable=False, )
        w0_L = np.vstack( (k0, np.zeros( (2, k0.size) )) ).T
        w0prime_L = np.vstack( (k0prime, np.zeros( (2, k0prime.size) )) ).T

        # determine needle shape for numpy version
        pmat_L, Rmat_L, wv_L = nss.numerical.integrateEP_w0(
                winit[ i ].numpy(),
                w0_L,
                w0prime_L,
                B.numpy(),
                s0=0,
                ds=ds,
                Binv=Binv.numpy(),
                wv_only=False,
        )

        # cast to full size for tensor
        seq_mask_L = np.asarray( [ True ] * w0_L.shape[ 0 ] + [ False ] * (M - w0_L.shape[ 0 ]) )
        w0_L = np.vstack(
                (
                        w0_L,
                        np.zeros( (M - w0_L.shape[ 0 ], w0_L.shape[ 1 ]) )
                )
        )
        w0prime_L = np.vstack(
                (
                        w0prime_L,
                        np.zeros( (M - w0prime_L.shape[ 0 ], w0prime_L.shape[ 1 ]) )
                )
        )

        # append to tensor arrays
        w0_ta = w0_ta.write( i, tf.convert_to_tensor( w0_L, dtype=w0_ta.dtype ) )
        w0prime_ta = w0prime_ta.write( i, tf.convert_to_tensor( w0prime_L, dtype=w0prime_ta.dtype ) )
        seq_mask_ta = seq_mask_ta.write( i, tf.convert_to_tensor( seq_mask_L, dtype=seq_mask_ta.dtype ) )

        # append to numpy solution
        nss_shape.append( pmat_L )

    # for

    # concatenate arrays
    w0 = w0_ta.stack()
    w0prime = w0prime_ta.stack()
    seq_mask = seq_mask_ta.stack()

    # get the tensorflow needle shapes
    nss_tf_shape, nss_tf_Rmat, nss_tf_wv, nss_tf_seq_mask = nss_tf.numerical.integrateEP_w0(
            winit,
            w0,
            w0prime,
            B,
            seq_mask,
            ds=ds,
            Binv=Binv,
    )

    # compare the needle shapes
    pmat_errors = dict()
    for i, L in enumerate( sorted( lengths, reverse=True ) ):
        nss_pmat_L = nss_shape[ i ]
        nss_tf_pmat_L = tf.boolean_mask( nss_tf_shape[ i ], nss_tf_seq_mask[ i ], axis=0 )

        deviations = nss_pmat_L - nss_tf_pmat_L.numpy()
        errors = np.linalg.norm( deviations, axis=0, ord=2 )

        # compute errors
        pmat_errors[ L ] = {
                "max" : np.max( errors ),
                "rmse": np.sqrt( np.mean( errors ** 2 ) ),
        }

    # for

    # print solutions
    for L, errors in pmat_errors.items():
        print( f"Errors for {L}: Max = {errors[ 'max' ]} | RMSE = {errors[ 'rmse' ]}" )

    # for


# test_shapes


def main():
    test_fns = [
            test_simpson_vec_int,
            test_integratePose_wv,
            test_integrateEP_w0,
            test_shapes,
    ]

    for fn in test_fns:
        fn()
        print()
        print( 100 * "=" )
        print()

    # for


# main

if __name__ == "__main__":
    main()

# if __main__
