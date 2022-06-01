import warnings

import numpy as np
from nflows.distributions import StandardNormal
from nflows.flows import Flow
from nflows.transforms import CompositeTransform
from nflows.transforms import \
    PiecewiseRationalQuadraticCouplingTransform as rq_spline
from nflows.transforms import ReversePermutation
from nflows.transforms.base import CompositeTransform




def make_mask(input_dim: int) -> list:
    """This function returns a mask for the transformer.
    The mask is a list of 1s and 0s, where 1s denote the identity and 0s denote the transform.

    Parameters
    ----------
    input_dim : int
        The input dimension/features, by default 5
    """
    n_mask = int(np.ceil(input_dim / 2))
    mask = [1] * n_mask + [0] * (input_dim - n_mask)
    return mask


def coupling_spline_transformer(
    input_dim=5,
    net_create_fn=None,
    num_stacks=4,
    tails=None,
    tail_bounds=1.0,
    num_bins=8,
    mask=None,
):
    """
    This function returns a rational quadradtic coupling spline transformer.
    The transformer is a composition of a number of coupling spline transforms, and the number of which is determined by the 'num_stacks' parameter.

    Parameters
    ----------
    input_dim : int
        The input dimension/features, by default 5
    net_create_fn : function
        The function to create the dense net required for estimating the paraemeters of the transformer, by default None
    num_stacks : int
        The number of spline transforms to stack, by default 2
    tails : string
        Function that governs the shape of the tails, by default None.
    tail_bounds : float
        Bounds on the tail; beyond this the shape would be governed by tails function, by default 1.0
    num_bins : int
        Number of spline bins, by default 8
    mask : list
        Determines which input to pass as identity and which to transform, by default None
    """

    if net_create_fn is None:
        warnings.warn("No net create function was passed.")
        exit()
    if tails is None:
        tails = "linear"
    if mask is None or len(mask) != input_dim:
        warnings.warn(
            f"mask must match the input dimension {input_dim}, but entered mask : {mask}. Adjusting mask."
        )
        mask = make_mask(input_dim)

    if not isinstance(num_stacks, int):
        warnings.warn("num_stacks must be an integer.")
        num_stacks = int(num_stacks)

    transform_list = []

    for _ in range(num_stacks):

        transform_list += [
            rq_spline(
                mask,
                net_create_fn,
                tail_bound=tail_bounds,
                num_bins=num_bins,
                tails=tails,
                apply_unconditional_transform=False,
            )
        ]
    transform_list += [ReversePermutation(input_dim)]

    return CompositeTransform(transform_list)


def coupling_flow(
    input_dim,
    net_create_fn=None,
    num_stacks=4,
    tails=None,
    tail_bounds=1.0,
    num_bins=8,
    base_density="Gaussian",
):
    """This function returns a coupling flow.
    The flow is characterised by a bijector and the base density.

    Parameters
    ----------
    input_dim : int
        The input dimension/features, by default 5
    net_create_fn : function
        The function to create the dense net required for estimating the paraemeters of the transformer, by default None
    num_stacks : int
        The number of spline transforms to stack, by default 4
    tails : string
        Function that governs the shape of the tails, by default None.
    tail_bounds : float
        Bounds on the tail; beyond this the shape would be governed by tails function, by default 1.0
    num_bins : int
        Number of spline bins, by default 8
    base_density : string
        The base density of the flow, by default 'Gaussian'
    """

    transformer = coupling_spline_transformer(
        input_dim, net_create_fn, num_stacks, tails, tail_bounds, num_bins
    )
    base_density = StandardNormal(shape=[input_dim])

    flow = Flow(transformer, base_density)
    return flow