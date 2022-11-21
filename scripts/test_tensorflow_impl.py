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
    Binv = tf.linalg.inv(B)
    seq_mask = tf.reduce_all( tf.ones_like( w0 ) > 0, axis=-1 )

    # original solution
    wv_nss = np.zeros_like( w0 )
    for i in range( wv_nss.shape[ 0 ] ):
        _, _, wv_nss[ i ] = nss.numerical.integrateEP_w0(
                w_init[i].numpy(), w0[i].numpy(), w0prime[i].numpy(), B.numpy(), ds=ds, Binv=Binv.numpy(),
        )

    # for

    # tensorflow implementation
    _, _, wv_nss_tf, _ = nss_tf.numerical.integrateEP_w0(
            w_init, w0, w0prime, B, seq_mask, ds, Binv=Binv,
    )

    # comparison
    print( "Original Solution:", wv_nss.shape, "\n", wv_nss[0] )
    print( "Tensorflow Implementation", wv_nss_tf.shape, "\n", wv_nss_tf.numpy()[0] )
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


def main():
    test_fns = [
            test_simpson_vec_int,
            test_integratePose_wv,
            test_integrateEP_w0,
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
