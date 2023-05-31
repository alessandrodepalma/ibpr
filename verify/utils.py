from jax_verify.src import bound_propagation, graph_traversal, synthetic_primitives
import jax.numpy as jnp
import jax

from typing import Union
Tensor = jnp.ndarray

RELAXATION_PRIMITIVES = [
    synthetic_primitives.relu_p
]


def get_verif_net_fn(net_fn, net_params, rng_key, targets, x, ground_truth_shift=100):
    # Create a function specifying the adversarial robustness problem for verification.
    logits = net_fn(net_params, rng_key, x)
    margin = jnp.take_along_axis(logits, jnp.expand_dims(targets, 1), axis=1) - logits
    # Add a constant along the entries that will always be 0 to be on the safe side
    return margin + jax.nn.one_hot(targets, logits.shape[-1]) * ground_truth_shift


class IBSaver(bound_propagation.GraphTransform[bound_propagation.Bound]):
    # Transform function wrapper that saves the pre-activation intermediate bounds in a self.intermediate_bounds,
    # a list of Bounds
    def __init__(self, orig_transform: bound_propagation.GraphTransform[bound_propagation.Bound]):
        self.orig_transform = orig_transform
        self.intermediate_bounds = []

    def should_handle_as_subgraph(self, primitive: graph_traversal.Primitive) -> bool:
        return self.orig_transform.should_handle_as_subgraph(primitive)

    def input_transform(
            self,
            context: graph_traversal.TransformContext,
            lower_bound: Tensor,
            upper_bound: Tensor,
    ) -> bound_propagation.Bound:
        return self.orig_transform.input_transform(context, lower_bound, upper_bound)

    def primitive_transform(self, context: graph_traversal.TransformContext,
                            primitive: graph_traversal.Primitive,
                            *args: Union[bound_propagation.Bound, Tensor],
                            **params) -> bound_propagation.Bound:
        if primitive in RELAXATION_PRIMITIVES:
            bound_inp = [arg.unwrap() for arg in args if isinstance(arg, bound_propagation.Bound)][0]
            self.intermediate_bounds.append(bound_inp)
        return self.orig_transform.primitive_transform(context, primitive, *args, **params)
