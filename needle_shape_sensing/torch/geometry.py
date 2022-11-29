import tensorflow as tf


def exp2r( w ):
    """ Convert R^3 tensor to SO(3) tensor

    Args:
        w - (N, 3) angular rotation vector

    Return:
        R - (N, 3, 3) Rotation matrices


    """
    Rta = tf.TensorArray(
            w.dtype,
            size=w.shape[ 0 ],
            clear_after_read=False,
            dynamic_size=False,
    )

    W = skew( w )

    for i in range( w.shape[ 0 ] ):
        theta = tf.norm( w[ i ], ord='euclidean' )
        if theta == 0:
            Rta.write( i, tf.eye( 3, dtype=w.dtype ) )
            continue

        # iF

        R_i = (
                tf.eye( 3, dtype=Rta.dtype )
                + tf.sin( theta ) * W[ i ] / theta
                + (1 - tf.cos( theta )) * (W[ i ] @ W[ i ]) / theta ** 2
        )

        Rta.write( i, R_i )

    # for

    Rmat = Rta.stack()

    return Rmat


# exp2r

def skew( w ):
    """ Skewify R^3 tensor

    Args:
        w - (N, 3)

    Return:
        W - (N, 3, 3) skew-symmetric matrix

    """

    W_ta = tf.TensorArray(
            w.dtype,
            size=w.shape[ 0 ],
            clear_after_read=False,
            dynamic_size=False,
    )

    for i in range( w.shape[ 0 ] ):
        W_ta.write(
                i,
                [
                        [ 0, -w[ i, 2 ], w[ i, 1 ] ],
                        [ w[ i, 2 ], 0, -w[ i, 0 ] ],
                        [ -w[ i, 1 ], w[ i, 0 ], 0 ],
                ]
        )

    # for

    W = W_ta.stack()

    return W

# skew
