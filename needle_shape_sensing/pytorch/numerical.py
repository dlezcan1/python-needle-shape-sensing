import torch

from needle_shape_sensing.pytorch import geometry


def integrateEP_w0(
        w_init: torch.Tensor, w0: torch.Tensor, w0prime: torch.Tensor, B: torch.Tensor,
        seq_mask: torch.Tensor, ds: float, R_init: torch.Tensor = torch.eye( 3 ),
        Binv: torch.Tensor = None, wv_only: bool = False
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
        Binv = torch.linalg.inv( B )

    # if

    # unpack the shapes
    N, M = w0.shape[ 0:2 ]

    # tensor-ify B and Binv
    B_T =  torch.tile(
                torch.reshape( B, (1, B.shape[ 0 ], B.shape[ 1 ]) ),
                (N, 1, 1)
    ).type(w0.dtype)
    Binv_T = torch.tile(
                torch.reshape( Binv, (1, B.shape[ 0 ], B.shape[ 1 ]) ),
                (N, 1, 1)
    ).type(w0.dtype)

    # prepare tensors
    wv_list = [w_init]

    for idx in range( 1, M ):
        seq_mask_idx = torch.unsqueeze( seq_mask[ :, idx:idx + 1 ], dim=-1 )

        # unpack veriables
        w0_im1      = torch.unsqueeze( w0[ :, idx - 1 ], dim=-1 ) # (N, 3, 1)
        w0prime_im1 = torch.unsqueeze( w0prime[ :, idx - 1 ], dim=-1 ) # (N, 3, 1)
        wv_im1      = torch.unsqueeze( wv_list[ idx - 1 ], dim=-1 ) # (N, 3, 1)

        if idx == 1:
            wv_i = (
                        wv_im1 + 1 * ds * (
                        w0prime_im1 - Binv_T @ torch.linalg.cross(
                            torch.squeeze( wv_im1, dim=-1 ),
                            torch.squeeze( B_T @ (wv_im1 - w0_im1), dim=-1 )
                        )[ :, :, None ]
                )
                ) * seq_mask_idx.type( w0.dtype )
            
        # if
        else:
            wv_im2 = torch.unsqueeze( wv_list[ idx - 2 ], dim=-1 ) # (N, 3, 1)
            wv_i   = (
                        wv_im2 + 2 * ds * (
                        w0prime_im1 - Binv_T @ torch.linalg.cross(
                            torch.squeeze( wv_im1, dim=-1 ),
                            torch.squeeze( B_T @ (wv_im1 - w0_im1), dim=-1 )
                        )[ :, :, None ]
                )
                ) * seq_mask_idx.type( w0.dtype )
            
        # else

        wv_list.append(torch.squeeze(wv_i, dim=-1))

    # for

    wv = torch.cat([wv_i[:, None] for wv_i in wv_list], dim=1)

    if wv_only:
        pmat, Rmat = None, None

    else:
        pmat, Rmat, seq_mask = integratePose_wv( wv, seq_mask, ds=ds, R_init=R_init, )

    return pmat, Rmat, wv, seq_mask


# integrateEP_w0

def integratePose_wv(
        wv: torch.Tensor, seq_mask: torch.Tensor, ds: float, R_init=torch.eye( 3 )
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

    # initial conditions
    pmat_list = [torch.zeros((N, 3), dtype=wv.dtype, device=wv.device)]
    Rmat_list = [R_init[None].tile((N, 1, 1)).type(wv.dtype).to(wv.device)]

    for idx in range( 1, M ):
        # unpack vars
        seq_mask_i = seq_mask[ :, idx ]  # (N, )
        Rmat_im1 = Rmat_list[idx-1]   # (N, 3, 3)
        pmat_im1 = pmat_list[idx - 1] # (N, 3)

        # integrate
        Rmat_i = Rmat_im1 @ (
                geometry.exp2r(
                        ds * torch.mean( wv[ :, idx - 1:idx ], dim=1 )
                        ) * seq_mask_i[ :, None, None ].type( wv.dtype )
        ).type(Rmat_im1.dtype)

        # add results for Rmat
        Rmat_list.append(Rmat_i)

        if idx == 1:
            pmat_i = pmat_im1 + Rmat_i[ :, :, 2 ] * ds

        else:
            e3vec = torch.cat(
                    [ R[:, None] for R in Rmat_list[0:idx + 1] ],
                    dim=1
            )[:, :, :, 2]  # grab z-directions
            pmat_i = simpson_vec_int( e3vec, ds, seq_mask[ :, :idx + 1 ] )

        # add results for pmat
        pmat_list.append(pmat_i)

    # for

    pmat = torch.cat(
            [p[:, None] for p in pmat_list],
            dim=1,
    )
    Rmat = torch.cat(
            [R[:, None] for R in Rmat_list],
            dim=1
    )

    return pmat, Rmat, seq_mask


# integratePose_wv


def simpson_vec_int( f: torch.Tensor, dx: float, seq_mask: torch.Tensor ):
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

    f_masked = f * seq_mask[ :, :, None ].type( f.dtype )

    # perform the integration
    int_res = torch.zeros( (f.shape[ 0 ], f.shape[ 2 ]), dtype=f.dtype, device=f.device )
    if num_intervals == 2:
        int_res = dx / 3 * torch.sum(
                f_masked[ :, 0:3 ]
                * torch.reshape( torch.tensor([ 1., 4., 1. ], device=f.device), (1, -1, 1) ).type( f_masked.dtype ),
                dim=1
        )

        return int_res

    # if

    elif num_intervals == 3:
        int_res = 3 / 8 * dx * torch.sum(
                f_masked[ :, 0:4 ]
                * torch.reshape( torch.tensor([ 1., 3., 3., 1. ], device=f.device), (1, -1, 1) ).type( f_masked.dtype ),
                dim=1
        )

        return int_res

    # elif

    if num_intervals % 2 != 0:
        int_res = (
                int_res +
                3 / 8 * dx * torch.sum(
                    f_masked[ :, -4: ]
                    * torch.reshape( torch.tensor([ 1., 3., 3., 1. ], device=f.device), (1, -1, 1) ).type( f_masked.dtype ),
                    dim=1
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
                    + 4 * torch.sum( f_masked[ :, 1:m:2 ], dim=1 )
            )
    )

    if m > 2:
        int_res = (
                int_res
                + 2 / 3 * dx * torch.sum( f_masked[ :, 2:m:2 ], dim=1 )
        )
    # if

    return int_res

# simpson_vec_int
