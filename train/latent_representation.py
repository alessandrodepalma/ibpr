import abc
import jax
from jax import numpy as jnp
from typing import Callable, Dict, List, Union, Optional
import jax_verify

from jax_verify.src import bound_propagation, graph_traversal
from verify.utils import IBSaver

Bound = bound_propagation.Bound
Index = bound_propagation.Index
Tensor = jnp.ndarray
JittableInputBound = graph_traversal.JittableInputBound


def attack_layer(latent_representation, net_from_latent, loss_fn, n_steps, step_size, rng):
    """Find latent adversarial point
        Args:
          latent_representation: LatentRepresentation instance, defining the representation of points in the
            latent space outer-approximation.
          net_from_latent: function (two arguments: RNG, layer inputs), the network from the chosen latent space
          loss_fn: function (single argument: loss inputs. targets must have been fed as partials), loss function,
              must return a scalar
          n_steps: int, number of PGD steps to run
          step_size: float, PGD step size
        Returns:
          adv_latent: Tensor, adversarial point in the latent space
          adv_loss: Tensor, value of the adversarial loss at adv_latent
    """
    rng, key = jax.random.split(rng)
    adv_vars = latent_representation.init(key)

    def loss_from_latent_repr(adv_vars, rng):
        latent_repr = latent_representation.evaluate(adv_vars)

        adv_outs = net_from_latent(rng, latent_repr)
        ce_loss = loss_fn(adv_outs)
        return ce_loss
    grad_loss = jax.grad(loss_from_latent_repr)

    def update(vars, grad):
        return vars + step_size * jnp.sign(grad)

    for it in range(n_steps-1):
        rng, key = jax.random.split(rng)
        grad_vars = grad_loss(adv_vars, key)

        # GD step on the sign
        adv_vars = jax.tree_multimap(update, adv_vars, grad_vars)

        # clip all variables representing a point in the latent space
        adv_vars = latent_representation.project(adv_vars)

    # TODO: if the network had state, it should be updated here
    adv_latent = latent_representation.evaluate(adv_vars)
    return adv_latent


class LatentRepresentation(metaclass=abc.ABCMeta):
    """Abstract Class defining the API to represent a point in the latent representation.
    """
    @abc.abstractmethod
    def init(self, rng) -> Union[List[Tensor], Dict[Index, Tensor]]:
        # Returns a pytree (constrained to either list or dict of tensors) for the initialization variables of the
        # latent representation.
        pass

    @abc.abstractmethod
    def evaluate(self, repr_vars: Union[List[Tensor], Dict[Index, Tensor]]) -> Tensor:
        # Given the pytree for the vars of the latent representation, evaluate/compute a point in the latent space.
        pass

    @abc.abstractmethod
    def project(self, repr_vars: Union[List[Tensor], Dict[Index, Tensor]]) -> Union[List[Tensor], Dict[Index, Tensor]]:
        # Given the pytree for the vars of the latent representation, project them to their feasible region.
        pass

    @staticmethod
    @abc.abstractmethod
    def from_larger_repr(net_to_latent: Callable, latent_repr: 'LatentRepresentation') -> 'LatentRepresentation':
        # Create a representation of net_to_latent from a representation, latent_repr, of a function that is a
        # function of net_to_latent
        pass

    @abc.abstractmethod
    def get_next_preact_bounds(self) -> Optional[Bound]:
        # Return the bounds for the pre-activations following the represented latent space.
        # Returns none if the LatentRepresentation has been created using from_larger_repr (in this
        # case the shallower representation does not have access to the function pointing to the following preacts)
        pass


class IdentityRepresentation(metaclass=abc.ABCMeta):
    def __init__(self, input_bounds: Bound, net_to_latent: Callable,
                 fun_to_bound: Optional[Callable] = None, rng=None, store_ibs=False):
        """
        Used to employ attack_layer to perform PGD attacks.
        The auxiliary variables only represent the input to the network.
        Args:
          input_bounds: jax_verify.IntervalBound, bounds on the inputs of the function.
          net_to_latent: function (single argument: net inputs), the network until the chosen latent space
          fun_to_bound: function (single argument: net inputs), the network until the pre-activations
            that we want to bound
          rng: Jax random number generator key
          store_ibs: whether to store intermediate bounds for fun_to_bound for future usage
        """
        self.input_bounds = input_bounds
        self.net_to_latent = net_to_latent
        self.rng = rng
        self.represent_next_preacts = fun_to_bound is not None
        self.fun_to_bound = fun_to_bound if self.represent_next_preacts else net_to_latent
        self.store_ibs = store_ibs
        self.stored_ibs = None

    def init(self, rng) -> Tensor:
        # Returns point uniformly sampled within the input bounds.
        return jax.random.uniform(
            rng, self.input_bounds.lower.shape, minval=self.input_bounds.lower, maxval=self.input_bounds.upper)

    def evaluate(self, repr_vars: Tensor) -> Tensor:
        # Simply evaluate the network on the current auxiliaries (net input).
        return self.net_to_latent(repr_vars)

    def project(self, repr_vars: Tensor) -> Tensor:
        # Project to the input bounds.
        return jnp.clip(repr_vars, a_min=self.input_bounds.lower, a_max=self.input_bounds.upper)

    @staticmethod
    def from_larger_repr(net_to_latent: Callable, latent_repr: 'IdentityRepresentation') -> 'IdentityRepresentation':
        return IdentityRepresentation(latent_repr.input_bounds, net_to_latent)

    def get_next_preact_bounds(self) -> Optional[Bound]:
        if not self.represent_next_preacts:
            return None

        # Compute pre-activation bounds with the chosen IBs algorithm.
        ib_transform = jax_verify.ibp_transform
        if self.store_ibs:
            ib_transform = IBSaver(ib_transform)
        preact_bounds, _ = bound_propagation.bound_propagation(
            bound_propagation.ForwardPropagationAlgorithm(ib_transform), self.fun_to_bound, self.input_bounds)
        if self.store_ibs:
            self.stored_ibs = ib_transform.intermediate_bounds
        else:
            raise NotImplementedError(f"Intermediate bounds {self.ibs} not supported for IdentityRepresentation")
        return preact_bounds

    def get_stored_ibs(self) -> Optional[List[Bound]]:
        # For a non-None return value, it requires an earlier call to get_next_preact_bounds
        return self.stored_ibs
