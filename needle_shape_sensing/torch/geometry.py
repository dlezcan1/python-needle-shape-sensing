import torch


def exp2r( w ):
    """ Convert R^3 tensor to SO(3) tensor

    Args:
        w - (N, 3) angular rotation vector

    Return:
        R - (N, 3, 3) Rotation matrices


    """
    Rmat = torch.eye( 3, dtype=w.dtype )[ None ].repeat( w.shape[ 0 ], dim=0 )
    W = skew( w )

    for i in range( w.shape[ 0 ] ):
        theta = torch.norm( w[ i ], p='fro' )
        if theta == 0:
            continue

        # iF

        R_i = (
                torch.eye( 3, dtype=Rmat.dtype )
                + torch.sin( theta ) * W[ i ] / theta
                + (1 - torch.cos( theta )) * (W[ i ] @ W[ i ]) / theta ** 2
        )

        Rmat[ i ] = R_i

    # for

    return Rmat


# exp2r

def skew( w ):
    """ Skewify R^3 tensor

    Args:
        w - (N, 3)

    Return:
        W - (N, 3, 3) skew-symmetric matrix

    """

    W = torch.zeros( (w.shape[ 0 ], 3, 3), dtype=w.dtype )

    for i in range( w.shape[ 0 ] ):
        W[ i ] = [
                [ 0, -w[ i, 2 ], w[ i, 1 ] ],
                [ w[ i, 2 ], 0, -w[ i, 0 ] ],
                [ -w[ i, 1 ], w[ i, 0 ], 0 ],
                ]

    # for

    return W

# skew
