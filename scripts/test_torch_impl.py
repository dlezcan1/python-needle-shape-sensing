import needle_shape_sensing as nss
import needle_shape_sensing.pytorch as nss_torch

import torch
import torch.nn as nn

# global options
ds = 0.5
OUTPUT_SIZE = 4
NEEDLE_LENGTH = 165
TORCH_DTYPE = torch.float64

B = torch.diag( torch.tensor( [ 1., 2., 3. ], dtype=TORCH_DTYPE ) )


def singlelayer_cshape_loss( XY, y_true ):
    """ Compute the shape loss with the correct variables"""
    global ds
    y_pred, X = unpack_XY( XY )

    ins_depths = X[ :, 0 ]

    N = X.shape[ 0 ]
    M = int( torch.ceil( ins_depths / ds ).max() ) + 1

    # set-up tensors
    w0_pred_list = [ ]
    w0prime_pred_list = [ ]
    seq_mask_pred_list = [ ]

    w0_true_list = [ ]
    w0prime_true_list = [ ]
    seq_mask_true_list = [ ]

    # iterate through the samples
    for i, L in enumerate( ins_depths ):
        w0_pred_nf_i, w0prime_pred_nf_i = compute_intrinsics( y_pred[ i ], L )
        w0_true_nf_i, w0prime_true_nf_i = compute_intrinsics( y_true[ i ], L )

        # compute sequence mask
        seq_mask_pred_i = torch.tensor(
                w0_pred_nf_i.shape[ 1 ] * [ True ] + (M - w0_pred_nf_i.shape[ 1 ]) * [ False ],
                dtype=torch.bool
        )
        seq_mask_true_i = torch.tensor(
                w0_true_nf_i.shape[ 1 ] * [ True ] + (M - w0_true_nf_i.shape[ 1 ]) * [ False ],
                dtype=torch.bool
        )

        # cast to the proper dimension 
        w0_pred_i = torch.cat(
                (
                        w0_pred_nf_i,
                        torch.zeros(
                                (M - w0_pred_nf_i.shape[ 0 ], 3),
                                dtype=w0_pred_nf_i.dtype,
                                requires_grad=True,
                        )
                ),
                dim=0
        )
        w0prime_pred_i = torch.cat(
                (
                        w0prime_pred_nf_i,
                        torch.zeros(
                                (M - w0prime_pred_nf_i.shape[ 0 ], 3),
                                dtype=w0prime_pred_nf_i.dtype,
                                requires_grad=True,
                        )
                ),
                dim=0
        )

        w0_true_i = torch.cat(
                (
                        w0_true_nf_i,
                        torch.zeros(
                                (M - w0_true_nf_i.shape[ 0 ], 3),
                                dtype=w0_true_nf_i.dtype,
                                requires_grad=True,
                        )
                ),
                dim=0
        )
        w0prime_true_i = torch.cat(
                (
                        w0prime_true_nf_i,
                        torch.zeros(
                                (M - w0prime_true_nf_i.shape[ 0 ], 3),
                                dtype=w0prime_true_nf_i.dtype,
                                requires_grad=True,
                        )
                ),
                dim=0
        )

        # append results
        w0_pred_list.append( w0_pred_i )
        w0prime_pred_list.append( w0prime_pred_i )
        seq_mask_pred_list.append( seq_mask_pred_i )
        w0_true_list.append( w0_true_i )
        w0prime_true_list.append( w0prime_true_i )
        seq_mask_true_list.append( seq_mask_true_i )

    # for

    w0_pred = torch.cat(
            [ w0[ None ] for w0 in w0_pred_list ],
            dim=0
    )
    w0prime_pred = torch.cat(
            [ w0prime[ None ] for w0prime in w0prime_pred_list ],
            dim=0
    )
    seq_mask_pred = torch.cat(
            [ sm[ None ] for sm in seq_mask_pred_list ],
            dim=0
    )
    w0_true = torch.cat(
            [ w0[ None ] for w0 in w0_true_list ],
            dim=0
    )
    w0prime_true = torch.cat(
            [ w0prime[ None ] for w0prime in w0prime_true_list ],
            dim=0
    )
    seq_mask_true = torch.cat(
            [ sm[ None ] for sm in seq_mask_true_list ],
            dim=0
    )

    # compute needle shapes
    pmat_pred, Rmat_pred, wv_pred, seq_mask_pred = nss_torch.numerical.integrateEP_w0(
            y_pred[ :, 0:3 ],
            w0_pred,
            w0prime_pred,
            B,
            seq_mask_pred,
            ds,
            wv_only=False,
    )
    pmat_true, Rmat_true, wv_true, seq_mask_true = nss_torch.numerical.integrateEP_w0(
            y_true[ :, 0:3 ],
            w0_true,
            w0prime_true,
            B,
            seq_mask_true,
            ds,
            wv_only=False,
    )

    seq_mask = seq_mask_pred & seq_mask_true

    errors_pts = nn.functional.mse_loss( pmat_pred, pmat_true, reduction='none' ).sum( dim=-1 )
    errors = []
    for i in range( N ):
        errors.append(torch.masked_select( errors_pts[ i ], seq_mask[ i ] ).mean())

    # for
    errors_sum = sum(errors)
    errors_mean = errors_sum / len(errors)

    return errors_mean


# singlelayer_cshape_loss

def compute_intrinsics( y, L ):
    """ Computes single-layer single-bend shape"""
    global ds
    winit, kc = y[ 0:3 ], y[ 3 ]
    L = float( L )
    s = torch.arange( 0, L + ds, ds, dtype=y.dtype )
    k0, k0prime = nss_torch.intrinsics.SingleBend.k0_1layer(
            s,
            kc,
            L,
            return_callable=False
    )

    w0 = torch.cat( (k0[ :, None ], torch.zeros( (k0.shape[ 0 ], 2), dtype=k0.dtype, requires_grad=True )), dim=1 )
    w0prime = torch.cat( (k0prime[ :, None ], torch.zeros( (k0prime.shape[ 0 ], 2), dtype=k0prime.dtype, requires_grad=True )), dim=1 )

    return w0, w0prime


# compute_shape

def pack_XY( X, y ):
    return torch.cat( (y, X), dim=1 )


# pack_XY

def unpack_XY( XY ):
    """ Returns y, X"""
    return XY[ :, :OUTPUT_SIZE ], XY[ :, OUTPUT_SIZE: ]


# unpack_XY

def test_loss():
    """ Test whether gradients work"""

    print( "Testing if loss functions and models compile" )

    # create dataset
    N = 100
    X = torch.cat(
            (
                    torch.rand( N, 1 ) * NEEDLE_LENGTH,
                    torch.randn( N, 5 ),
            ),
            dim=1
    )

    y_true = torch.rand( N, OUTPUT_SIZE ) * 0.01

    model = nn.Linear( X.shape[ 1 ], OUTPUT_SIZE )
    y_pred = model( X )
    XY_pred = pack_XY( X, y_pred )

    loss = singlelayer_cshape_loss( XY_pred, y_true )
    loss.retain_grad()
    model.weight.retain_grad()
    loss.backward()

    print( "Model's gradient:\n", model.weight.grad )
    print( "Loss's gradient:\n", loss.grad )


# test_loss


def main():
    test_fns = [
            test_loss,
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
