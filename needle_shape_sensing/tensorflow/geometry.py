import tensorflow as tf


def exp2r(w):
    """ Convert R^3 tensor to SO(3) tensor

    Args:
        w - (N, 3) angular rotation vector

    Return:
        R - (N, 3, 3) Rotation matrices


    """
    pass # TODO

# exp2r

def skew(w):
    """ Skewify R^3 tensor

    Args:
        w - (N, 3)

    Return:
        W - (N, 3, 3) skew-symmetric matrix

    """

    W_half = tf.TensorArray(
            tf.float32,
            size=w.shape[0],
            clear_after_read=False,
            dynamic_size=False,
    )

    

