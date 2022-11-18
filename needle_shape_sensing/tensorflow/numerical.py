import tensorflow as tf
import numpy as np

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
        Binv = tf.linalg.inv(B)

    # if

    # unpack the shapes
    N, M = w0.shape[0:2]

    # tensor-ify B and Binv
    B_T = tf.tile(
            tf.reshape(B, (1, B.shape[0], B.shape[1])),
            (N, 1, 1)
    )
    Binv_T = tf.tile(
            tf.reshape(Binv, (1, B.shape[0], B.shape[1])),
            (N, 1, 1)
    )

    # prepare tensors
    wv = tf.TensorArray(
            tf.float32,
            size=M,
            dynamic_size=False,
            infer_shape=True,
            clear_after_read=False,
    )
    wv = wv.write(0, w_init)

    for idx in range(1, M):
        seq_mask_idx = tf.expand_dims(seq_mask[:, idx:idx+1], axis=-1)

        # unpack veriables
        w0_im1 = tf.expand_dims( w0[ :, idx - 1 ], axis=-1 )
        w0prime_im1 = tf.expand_dims( w0prime[ :, idx - 1 ], axis=-1 )
        wv_im1 = tf.expand_dims( wv.read( idx - 1 ), axis=-1 )
        scale = 1 if idx == 1 else 2

        wv_i = (
            wv_im1 + scale * ds * (
                w0prime_im1 - Binv_T @ tf.linalg.cross(wv_im1, B @ (wv_im1 - w0_im1))
            )
        ) * tf.cast(seq_mask_idx, dtype=wv.dtype)

        wv = wv.write(idx, tf.squeeze(wv_i, axis=-1))

    # for

    wv = tf.transpose(wv.stack(), [1, 0, 2])

    if wv_only:
        pmat, Rmat = None, None

    else:
        pmat, Rmat = integratePose_wv( wv, seq_mask, ds=ds, R_init=R_init, )

    return pmat, Rmat, wv, seq_mask


# integrateEP_w0

def integratePose_wv(
       wv: tf.Tensor, seq_mask: tf.Tensor, ds: float, R_init = tf.eye(3)
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
    N, M = wv.shape[0:2]

    pmat = tf.TensorArray(
            tf.float32,
            size=M,
            dynamic_size=False,
            infer_shape=True,
            clear_after_read=False,
    )
    Rmat =tf.TensorArray(
            tf.float32,
            size=M,
            dynamic_size=False,
            infer_shape=True,
            clear_after_read=False,
    )

    # initial conditions
    pmat = pmat.write(0, tf.zeros(N, 3))
    Rmat = Rmat.write(0, tf.tile(
            tf.reshape(R_init, (1, R_init.shape[0], R_init.shape[1])),
            (N, 1, 1)
    ))

    for idx in range(1, M):
        # unpack vars
        seq_mask_i = seq_mask[:, idx]
        Rmat_im1 = Rmat.read( idx - 1 )
        pmat_im1 = pmat.read( idx - 1 )


        # integrate
        Rmat_i = 0 # TODO
        pmat_i = 0 # TODO

        # add results
        Rmat = Rmat.write(idx, Rmat_i)
        pmat = pmat.write(idx, pmat_i)


