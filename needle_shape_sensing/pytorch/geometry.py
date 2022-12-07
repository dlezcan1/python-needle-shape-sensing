import torch


def exp2r( w ):
    """ Convert R^3 tensor to SO(3) tensor

    Args:
        w - (N, 3) angular rotation vector

    Return:
        R - (N, 3, 3) Rotation matrices


    """
    W = skew( w )

    Rmat_list = []
    for i in range( w.shape[ 0 ] ):
        theta = torch.norm( w[ i ], p='fro' )
        if theta == 0:
            Rmat_list.append(torch.eye(3, dtype=w.dtype, device=w.device))
            continue

        # iF

        R_i = (
                torch.eye( 3, dtype=w.dtype, device=w.device )
                + torch.sin( theta ) * W[ i ] / theta
                + (1 - torch.cos( theta )) * (W[ i ] @ W[ i ]) / theta ** 2
        )
        Rmat_list.append(R_i)

    # for

    Rmat = torch.cat(
            [ R[None] for R in Rmat_list ],
            dim=0
    )

    return Rmat


# exp2r

def skew( w ):
    """ Skewify R^3 tensor

    Args:
        w - (N, 3)

    Return:
        W - (N, 3, 3) skew-symmetric matrix

    """
    W_list = []
    for i in range( w.shape[ 0 ] ):
        W_list.append( torch.tensor(
            [
                [ 0, -w[ i, 2 ], w[ i, 1 ] ],
                [ w[ i, 2 ], 0, -w[ i, 0 ] ],
                [ -w[ i, 1 ], w[ i, 0 ], 0 ],
            ],
            device=w.device,
        ))

    # for

    W = torch.cat(
            [W_i[None] for W_i in W_list],
            dim=0
    )

    return W

# skew
