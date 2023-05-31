from train.networks import ConvMed, FFNN, ConvMedBig, ConvMedBigger, DMLarge, FCShallow
from jax_verify.src import bound_propagation, graph_traversal
import jax.numpy as jnp
from train.loaders import get_mean_sigma
import haiku as hk
import functools
import pickle
import jax

from typing import Union
Tensor = jnp.ndarray


def cross_entropy(labels, logits):
    # Given logits and labels, compute cross entropy loss.
    labels = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(labels * jax.nn.log_softmax(logits)) / labels.shape[0]


def compute_correctly_classified(labels, logits):
    # Given logits and labels, compute fraction of outputs that are correctly classified
    return (labels == logits.argmax(axis=1)).astype(jnp.float32).mean()


def network_fn(net_name, mean, sigma, n_class, x, start=None, stop=None, train=False, dropout=False):
    # Functional view of the network, to be haiku.transform-ed.
    if net_name.startswith('ffnn_'):
        tokens = net_name.split('_')
        sizes = [int(x) for x in tokens[1:]]
        net = FFNN(mean, sigma, sizes, n_class)
    elif net_name.startswith('convmed_'):
        tokens = net_name.split('_')
        width1 = int(tokens[2])
        width2 = int(tokens[3])
        linear_size = int(tokens[4])
        net = ConvMed(mean, sigma, n_class, width1=width1, width2=width2, linear_size=linear_size)
    elif net_name.startswith('convmedbig_'):
        tokens = net_name.split('_')
        assert tokens[0] == 'convmedbig'
        width1 = int(tokens[2])
        width2 = int(tokens[3])
        width3 = int(tokens[4])
        linear_size = int(tokens[5])
        net = ConvMedBig(mean, sigma, n_class, width1, width2, width3, linear_size=linear_size)
    elif net_name.startswith('convmedbigger_'):
        tokens = net_name.split('_')
        assert tokens[0] == 'convmedbigger'
        width1 = int(tokens[2])
        width2 = int(tokens[3])
        width3 = int(tokens[4])
        linear_size = int(tokens[5])
        net = ConvMedBigger(
            mean, sigma, n_class, width1, width2, width3, linear_size=linear_size, train=train, dropout=dropout)
    elif net_name.startswith('dmlarge'):
        net = DMLarge(mean, sigma, n_class)
    elif net_name.startswith('fcshallow'):
        tokens = net_name.split('_')
        assert tokens[0] == 'fcshallow'
        lin_size = int(tokens[1])
        net = FCShallow(mean, sigma, lin_size, n_class)
    else:
        assert False, 'Unknown network!'
    # Possibly clip the network before or after a certain layer.
    return net.forward(x, start=start, stop=stop)


def get_network(net_name, dataset, input_size, input_channel, n_class, rng, start=None, stop=None, load_model=None,
                train=False, dropout=False):
    """
    Return a functional view of the network. Using the start or stop parameters, only a partial view of the network can
    be employed.
    Returns:
          params: Haiku's representation of the network parameters (either at initialization or loaded, if load_model)
          net_hk.apply: function for the network, with three arguments: the parameters, RNG, the input point
    """
    mean, sigma = get_mean_sigma(dataset)
    net_fn = functools.partial(
        network_fn, net_name, mean, sigma, n_class, start=start, stop=stop, train=train, dropout=dropout)

    if net_name.startswith('ffnn_'):
        # all fully connected architectures should enter this if
        dummy_inp = jnp.ones((1, input_size*input_size*input_channel))
    else:
        dummy_inp = jnp.ones((1, input_size, input_size, input_channel))

    if start is not None:
        # Pass input forward to the relevant latent layer if start is not None
        start_net_fn = functools.partial(network_fn, net_name, mean, sigma, n_class, start=None, stop=start)
        start_net_hk = hk.transform(start_net_fn)
        start_net_hk = hk.without_apply_rng(start_net_hk)
        params = start_net_hk.init(rng, dummy_inp)
        dummy_inp = start_net_hk.apply(params, dummy_inp)

    net_hk = hk.transform(net_fn)
    params = net_hk.init(rng, dummy_inp)

    if load_model is not None:
        assert (start is None) and (stop is None), "Only parameters of non-clamped networks can be loaded"
        # Load parameters from file rather than using the initialization values
        with open(load_model, 'rb') as f:
            params = pickle.load(f)
    return params, net_hk.apply


def network_functional_views(net_params, net_getter, curr_layer_idx, prev_layer_idx=None, next_layer_idx=None,
                             train=False):
    """Get relevant functional views (and parameters) of the network for use in layerwise training.
       NOTE that the latent space is straight after an activation function, in the COLT paper.
       NOTE-2: the network views that are meant to be fixed/used for bounding will be loaded in evaluation mode
            regardless of the train parameter
        Args:
          net_params: haiku FlatMap representing the network parameters (their current training value).
          net_getter: function to obtain initialization parameters and functional view of part of the net,
            a functools.partial view of get_network above with only start, stop, and train missing
          prev_layer_idx: index of post-act layer (numbering includes activation fns) before the current latent space
          curr_layer_idx: index of layer (numbering includes activation functions) of the current latent space
          next_layer_idx: index of post-act layer (numbering includes activation fns) after the current latent space.
            If None, the current latent space is the last layer
          train: bool indicating whether the net should be loaded in train mode (see NOTE-2 above)
        Returns:
          net_to_latent: network until the current latent space, takes as only input the original net input points
          p_net_to_latent: as net_to_latent, but stops a layer earlier
          net_to_next_preact: as net_to_latent, but stops a layer later and takes two inputs: the original net input,
            and toopt_params. Will be none if next_layer_idx is None
          net_from_latent_fn: network from the current latent space, takes three inputs
            (i) its parameters (ii) RNG (iii) latent activation
          p_net_from_latent_fn: network from the layer before the current latent space, takes three inputs
                (i) parameters of net_from_latent_fn (ii) RNG (iii) latent activation
          toopt_params: haiku FlatMap representing the network parameters (their current training value) that are not
            frozen (that is, those after the curr_layer_idx-th layer).
          fixed_params: haiku FlatMap representing the network parameters (their current training value) that are
            frozen (that is, those before the curr_layer_idx-th layer).
    """
    # Get fixed view of the network until the latent space at curr_layer_idx
    net_to_latent = get_fixed_net_view(net_params, net_getter, curr_layer_idx)

    # Get view of the network from the latent space onwards as a function of the parameters to optimize
    rnd_after_params, net_from_latent_fn = net_getter(start=curr_layer_idx, train=train)
    # select relevant parameter subset from net_params
    predicate_toopt = lambda module_name, name, value: module_name in rnd_after_params.keys()
    predicate_fixed = lambda module_name, name, value: module_name not in rnd_after_params.keys()
    toopt_params = hk.data_structures.filter(predicate_toopt, net_params)
    fixed_params = hk.data_structures.filter(predicate_fixed, net_params)

    if prev_layer_idx is not None:
        # Get fixed view of the network until the latent space at prev_layer_idx
        p_net_to_latent = get_fixed_net_view(net_params, net_getter, prev_layer_idx)

        # Get view of the network from the previous latent space onwards as a function of the parameters to optimize
        rnd_p_before_params, p_net_from_latent_fn_full = net_getter(start=prev_layer_idx, train=train)
        prev_layer_params = hk.data_structures.filter(
            lambda module_name, name, value: (module_name in rnd_p_before_params.keys()) and
                                             (module_name not in toopt_params.keys()),
            net_params)

        def p_net_from_latent_fn(var_params):
            return functools.partial(
                p_net_from_latent_fn_full, hk.data_structures.merge(prev_layer_params, var_params))
    else:
        p_net_to_latent, p_net_from_latent_fn = None, None

    net_to_next_preact = None
    if next_layer_idx is not None:
        # Get fixed view of the network until the pre-activations BEFORE the next latent space
        net_to_next_preact = _get_partly_fixed_net_view(net_params, toopt_params, net_getter, next_layer_idx-1)

    return net_to_latent, p_net_to_latent, net_to_next_preact, net_from_latent_fn, p_net_from_latent_fn, \
           toopt_params, fixed_params


def get_fixed_net_view(net_params, net_getter, layer_idx):
    # Get view of the net before layer_idx as a function of the input only, with the relevant part of net_parameters
    # applied via functools.partial
    # NOTE: it will return a network in evaluation mode
    rnd_before_params, net_to_latent_fn = net_getter(stop=layer_idx, train=False)
    # Get fixed parameters from net_params
    fixed_params = hk.data_structures.filter(
        lambda module_name, name, value: module_name in rnd_before_params.keys(),
        net_params)
    # TODO: it would be better to pass a rng for consistency
    # Get fixed view of the network until the latent space at layer_idx (None is passed as RNG, not used in eval mode)
    net_to_latent = functools.partial(net_to_latent_fn, fixed_params, None)
    return net_to_latent


def _get_partly_fixed_net_view(net_params, toopt_params, net_getter, layer_idx):
    # Get view of the net before layer_idx as a function of the parameters we optimize over only,
    # with the relevant, fixed, part of net_parameters applied via functools.partial
    # NOTE: it will return a network in evaluation mode
    rnd_before_params, net_to_latent_fn = net_getter(stop=layer_idx, train=False)
    # Get fixed parameters from net_params
    fixed_params = hk.data_structures.filter(
        lambda module_name, name, value: (module_name in rnd_before_params.keys()) and
                                         (module_name not in toopt_params.keys()),
        net_params)

    def partly_fixed_net(in_fixed_params, in_toopt_params, input):
        # Ignore in_toopt_params that do not belong to net_to_latent_fn's parameters
        filtered_toopt_params = hk.data_structures.filter(
            lambda module_name, name, value: (module_name in rnd_before_params.keys()),
            in_toopt_params)
        # (None is passed as RNG, not used in eval mode)
        return net_to_latent_fn(hk.data_structures.merge(filtered_toopt_params, in_fixed_params), None, input)

    # Get fixed view of the network until the latent space at layer_idx
    net_to_latent = functools.partial(partly_fixed_net, fixed_params)
    return net_to_latent


class DummyTransformedNode(graph_traversal.TransformedNode):
    pass


class DummyTransform(bound_propagation.GraphTransform[DummyTransformedNode]):
    # Dummy transform function that is executed to compute the index of the PropagationGraph.
    def input_transform(
            self,
            context: graph_traversal.TransformContext,
            lower_bound: Tensor,
            upper_bound: Tensor,
    ) -> DummyTransformedNode:
        return DummyTransformedNode()

    def primitive_transform(self, context: graph_traversal.TransformContext,
                            primitive: graph_traversal.Primitive,
                            *args: Union[DummyTransformedNode, Tensor],
                            **params) -> DummyTransformedNode:
        return DummyTransformedNode()


class Scheduler:
    def __init__(self, start, end, n_steps, warmup):
        self.start = start
        self.end = end
        self.n_steps = n_steps
        self.warmup = warmup
        self.curr_steps = 0

    def advance_time(self, k_steps):
        self.curr_steps += k_steps

    def get(self):
        if self.n_steps == self.warmup:
            return self.end
        if self.curr_steps < self.warmup:
            return self.start
        elif self.curr_steps > self.n_steps:
            return self.end
        return self.start + (self.end - self.start) * (self.curr_steps - self.warmup)/float(self.n_steps - self.warmup)


class Statistics:
    def __init__(self):
        self.n = 0
        self.avg = 0.0

    def update(self, x):
        self.avg = self.avg * self.n / float(self.n + 1) + x / float(self.n + 1)
        self.n += 1

    @staticmethod
    def get_statistics(k):
        return [Statistics() for _ in range(k)]
