import torch


def exp2r( w ):
    """ Convert R^3 tensor to SO(3) tensor

    Args:
        w - (N, 3) angular rotation vector

    Return:
        R - (N, 3, 3) Rotation matrices


    """
    # compute norms and normalized omega vectors
    thetas = torch.norm( w, p='fro', dim=1, keepdim=True )
    norms = thetas * (thetas != 0) + torch.ones_like( thetas ) * (thetas == 0)
    w_norm = w / norms
    W_norm = skew( w_norm )

    Rmat = (
            torch.eye( 3, dtype=w.dtype, device=w.device )[ None ].repeat( w.shape[ 0 ], 1, 1 )
            + torch.sin( thetas.unsqueeze( dim=-1 ) ) * W_norm
            + (1 - torch.cos( thetas.unsqueeze( dim=-1 ) )) * (W_norm @ W_norm)
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
    zeros = torch.zeros_like( w[ :, 0:1 ], dtype=w.dtype, device=w.device )
    W_half = torch.cat(
            (
                    zeros, -w[ :, 2:3 ], w[ :, 1:2 ],
                    zeros, zeros, -w[ :, 0:1 ],
                    zeros, zeros, zeros,
            ),
            dim=1
    ).reshape( -1, 3, 3 )

    return W_half - W_half.transpose( 1, 2 )

# skew
