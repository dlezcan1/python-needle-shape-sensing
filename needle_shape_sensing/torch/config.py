import torch


def get_device( acceleration: bool = False ):
    """ Gets the device to put the tensors on

    Args:
        acceleartion (bool, Default=False): whether to return a device (if available for acceleration)

    Returns:
        torch.device: the torch device to use

    """
    device = torch.device( "cpu" )

    if not acceleration:
        return device

    if torch.cuda.is_available():
        device = torch.device( "cuda" )

    elif torch.backends.mps.is_built():
        device = torch.device( "mps" )

    return device

# get_device
