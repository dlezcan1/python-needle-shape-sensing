import tensorflow as tf

from needle_shape_sensing.tensorflow import geometry


def integrateEP_w0(
        w_init: tf.Tensor, w0: tf.Tensor, w0prime: tf.Tensor, B: tf.Tensor,
        seq_mask: tf.Tensor, ds: float, R_init: tf.Tensor = tf.eye( 3 ),
        Binv: tf.Tensor = None, wv_only: bool = False
):
    """
    integrate Euler-Poincare equation for needle shape sensing for given intrinsic angular deformation

    Args:
            w_init: (N, 3) initial deformation vector
            w0: (N, M, 3) intrinsic angular deformation
            w0prime: (N, M, 3) d/ds w0
            B: 3 x 3 needle stiffness matrix
            seq_mask: (N, M) mask of which are valid points to use
            ds: the arclength increments
            Binv: (Default = None) inv(B) Can be provided for numerical efficiency
            R_init: (Default = 3x3 identity) SO3 matrix for initial rotation angle
            wv_only: (Default = False) whether to only integrate wv or not.

        Return:
            (N, M, 3) needle shape, (N, M, 3, 3) SO3 matrices of orientations), (N, M, 3) angular deformation), (N, M) sequence mask
            (None, None, (N, M, 3) angular deformation) if 'wv_only' is True

    """

    if Binv is None:
        Binv = tf.linalg.inv( B )

    # if

    # unpack the shapes
    N, M = w0.shape[ 0:2 ]

    # tensor-ify B and Binv
    B_T = tf.cast(
            tf.tile(
                    tf.reshape( B, (1, B.shape[ 0 ], B.shape[ 1 ]) ),
                    (N, 1, 1)
            ), dtype=w0.dtype
    )
    Binv_T = tf.cast(
            tf.tile(
                    tf.reshape( Binv, (1, B.shape[ 0 ], B.shape[ 1 ]) ),
                    (N, 1, 1)
            ), dtype=w0.dtype
    )

    # prepare tensors
    wv = tf.TensorArray(
            w0.dtype,
            size=M,
            dynamic_size=False,
            infer_shape=True,
            clear_after_read=False,
    )
    wv = wv.write( 0, w_init )

    for idx in range( 1, M ):
        seq_mask_idx = tf.expand_dims( seq_mask[ :, idx:idx + 1 ], axis=-1 )

        # unpack veriables
        w0_im1      = tf.expand_dims( w0[ :, idx - 1 ], axis=-1 )
        w0prime_im1 = tf.expand_dims( w0prime[ :, idx - 1 ], axis=-1 )
        wv_im1      = tf.expand_dims( wv.read( idx - 1 ), axis=-1 )

        if idx == 1:
                wv_i = (
                        wv_im1 + 1 * ds * (
                                w0prime_im1 - Binv_T @ tf.linalg.cross(
                                        tf.squeeze(wv_im1, axis=-1),
                                        tf.squeeze(B_T @ (wv_im1 - w0_im1), axis=-1)
                                 )[:, :, tf.newaxis]
                        )
                ) * tf.cast( seq_mask_idx, dtype=wv.dtype )

        # if
        else:
            wv_im2 = tf.expand_dims( wv.read( idx - 2 ), axis=-1 )
            wv_i   = (
                        wv_im2 + 2 * ds * (
                                w0prime_im1 - Binv_T @ tf.linalg.cross(
                                        tf.squeeze(wv_im1, axis=-1),
                                        tf.squeeze(B_T @ (wv_im1 - w0_im1), axis=-1)
                                )[:, :, tf.newaxis]
                        )
                ) * tf.cast( seq_mask_idx, dtype=wv.dtype )

        wv = wv.write( idx, tf.squeeze( wv_i, axis=-1 ) )

    # for

    wv = tf.transpose( wv.stack(), [ 1, 0, 2 ] )

    if wv_only:
        pmat, Rmat = None, None

    else:
        pmat, Rmat, seq_mask = integratePose_wv( wv, seq_mask, ds=ds, R_init=R_init, )

    return pmat, Rmat, wv, seq_mask


# integrateEP_w0

def integratePose_wv(
        wv: tf.Tensor, seq_mask: tf.Tensor, ds: float, R_init=tf.eye( 3 )
):
    """ Integrate angular deformation to get the pose of the needle along it's arclengths

        :param wv: (N, M, 3) angular deformation vector
        :param seq_mask: (N, M) sequence mask for which are viable
        :param ds: (Default = None) the arclength increments desired
        :param R_init: (Default = numpy.eye(3)) Rotation matrix of the inital pose

        :returns: pmat, Rmat, sequence mask
            - pmat: (N, M, 3) position for the needle shape points in-tissue
            - Rmat: (N, M, 3, 3) SO(3) rotation matrices for
            - sequence mask: (N, M) of boolean for which sequences are viable
    """
    N, M = wv.shape[ 0:2 ]

    pmat_ta = tf.TensorArray(
            wv.dtype,
            size=M,
            dynamic_size=False,
            infer_shape=True,
            clear_after_read=False,
    )  # (M, N, 3)
    Rmat_ta = tf.TensorArray(
            wv.dtype,
            size=M,
            dynamic_size=False,
            infer_shape=True,
            clear_after_read=False,
    )  # (M, N, 3, 3)

    # initial conditions
    pmat_ta = pmat_ta.write( 0, tf.zeros( (N, 3), dtype=pmat_ta.dtype ) )
    Rmat_ta = Rmat_ta.write(
            0, tf.cast(
                    tf.tile(
                            tf.reshape( R_init, (1, R_init.shape[ 0 ], R_init.shape[ 1 ]) ),
                            (N, 1, 1)
                    ), dtype=Rmat_ta.dtype
            )
    )

    for idx in range( 1, M ):
        # unpack vars
        seq_mask_i = seq_mask[ :, idx ]
        Rmat_im1 = Rmat_ta.read( idx - 1 )  # (N, 3, 3)
        pmat_im1 = pmat_ta.read( idx - 1 )  # (N, 3)

        # integrate
        Rmat_i = Rmat_im1 @ (
                geometry.exp2r(
                        ds * tf.reduce_mean( wv[ :, idx - 1:idx ], axis=1 )
                ) * tf.cast( seq_mask_i[ :, tf.newaxis, tf.newaxis ], dtype=wv.dtype )
        )

        # add results for Rmat
        Rmat_ta = Rmat_ta.write( idx, Rmat_i )

        e3vec = tf.transpose( Rmat_ta.stack()[ :idx + 1, :, :, 2 ], [ 1, 0, 2 ] )  # grab z-directions
        if idx == 1:
            pmat_i = pmat_im1 + Rmat_i[ :, :, 2 ] * ds

        else:
            pmat_i = simpson_vec_int( e3vec, ds, seq_mask[ :, :idx + 1 ] )

        # add results for pmat
        pmat_ta = pmat_ta.write( idx, pmat_i )

    # for

    Rmat = tf.transpose( Rmat_ta.stack(), [ 1, 0, 2, 3 ] )
    pmat = tf.transpose( pmat_ta.stack(), [ 1, 0, 2 ] )

    return pmat, Rmat, seq_mask


# integratePose_wv


def simpson_vec_int( f: tf.Tensor, dx: float, seq_mask: tf.Tensor ):
    """ Implementation of Simpson vector integration for tensor integration

            Original Author (MATLAB): Jin Seob Kim
            Translated Author: Dimitri Lezcano

            Args:
                 f:  (N, M, D) tensor where D is the dimension of the vector is the dimension of the parameter ( N > 2 )
                        and M Integration samples
                 dx: float of the step size
                 seq_mask: (N, M) boolean mask for which parts of the integration is valid

            Return:
                integrated tensor of shape (N, D)

    """
    N, M, D = f.shape[ :3 ]
    num_intervals = M - 1
    assert (num_intervals > 1)

    f_masked = f * tf.cast( seq_mask[ :, :, tf.newaxis ], dtype=f.dtype )

    # perform the integration
    int_res = tf.zeros( (f.shape[ 0 ], f.shape[ 2 ]), dtype=f.dtype )
    if num_intervals == 2:
        int_res = dx / 3 * tf.reduce_sum(
                f_masked[ :, 0:3 ] * tf.cast( tf.reshape( [ 1., 4., 1. ], (1, -1, 1) ), dtype=f_masked.dtype ),
                axis=1
        )

        return int_res

    # if

    elif num_intervals == 3:
        int_res = 3 / 8 * dx * tf.reduce_sum(
                f_masked[ :, 0:4 ] * tf.cast( tf.reshape( [ 1., 3., 3., 1. ], (1, -1, 1) ), dtype=f_masked.dtype ),
                axis=1
        )

        return int_res

    # elif

    if num_intervals % 2 != 0:
        int_res = (
                int_res +
                3 / 8 * dx * tf.reduce_sum(
                f_masked[ :, -4: ] * tf.cast( tf.reshape( [ 1., 3., 3., 1. ], (1, -1, 1) ), dtype=f_masked.dtype ),
                axis=1
        )
        )
        m = num_intervals - 3

    # if
    else:
        m = num_intervals

    # else

    int_res = (
            int_res
            + dx / 3 * (
                    f_masked[ :, 0 ] + f_masked[ :, m ]
                    + 4 * tf.reduce_sum(
                    f_masked[ :, 1:m:2 ], axis=1
            )
            )
    )

    if m > 2:
        int_res = (
                int_res
                + 2 / 3 * dx * tf.reduce_sum(
                f_masked[ :, 2:m:2 ], axis=1
        )
        )
    # if

    return int_res

# simpson_vec_int
