import time
import json
import numpy as np
import os
import torch
from train.args_factory import get_args
from train.loaders import get_loaders
from train.utils import get_network, Scheduler, cross_entropy, compute_correctly_classified
from tqdm import tqdm
from train.latent_representation import attack_layer, IdentityRepresentation
from verify.utils import get_verif_net_fn
import haiku as hk
import jax.numpy as jnp
import jax_verify
import functools
import jax
import optax
import pickle


def compute_l1_loss(params, only_weights=True):
    # only_weights is true in the original PT implementation.
    if only_weights:
        # Exclude batchnorm parameters from the computation, and use only weights (no regularization on biases)
        # NOTE: assumes haiku weights are named "w", which is the case for standard hk.Conv2d and hk.Linear layers
        param_filter = lambda module_name, name, value: ('batchnorm' not in module_name) and (name == 'w')
    else:
        # Exclude batchnorm parameters from the computation
        param_filter = lambda module_name, name, value: ('batchnorm' not in module_name)
    l1_params = hk.data_structures.filter(param_filter, params)
    l1_loss = sum(jnp.sum(jnp.abs(p)) for p in jax.tree_leaves(l1_params))
    return l1_loss


def compute_relu_kw_area_loss(next_layer_idx, relu_stable, next_bounds, sum=True):
    # Given next_bounds, compute sum of areas of ReLUs KW relaxations at that layer (summed across batch entries)
    cross_relu, relu_area_loss = 0, 0
    if next_layer_idx is not None and relu_stable is not None:
        next_lb, next_ub = next_bounds.lower, next_bounds.upper
        is_cross = (next_lb < 0) & (next_ub > 0)
        cross_relu = is_cross.astype(jnp.float32).sum()
        relu_area_loss = (jnp.clip(-next_lb, a_min=0) * jnp.clip(next_ub, a_min=0))
        relu_area_loss = jnp.reshape(relu_area_loss, (relu_area_loss.shape[0], -1)).sum(axis=-1)
        if sum:
            relu_area_loss = relu_area_loss.sum()
    return cross_relu, relu_area_loss


def ibp_bounds(net_fn, net_params, input_bounds, targets):
    verif_net_fn = functools.partial(get_verif_net_fn, net_fn, net_params, None, targets, ground_truth_shift=0)
    ibp_output_bound = jax_verify.interval_bound_propagation(verif_net_fn, input_bounds)
    return ibp_output_bound.lower


def holistic_loss(net_params, net_fn, args, input_bounds, inputs,
                  targets, kappa, loss_fn, is_correctly_classified, rng):

    # Computes the loss with respect to which the parameters are differentiated
    rng, key = jax.random.split(rng)
    nat_outs = net_fn(net_params, key, inputs)
    nat_ok = is_correctly_classified(nat_outs)
    nat_loss = loss_fn(targets, nat_outs)

    rng, key1, key2 = jax.random.split(rng, num=3)
    # The latent representation itself does not depend on toopt_params. However, the class computes the bounds for
    # the following latent space, in case n_net_to_latent is provided. This computation does depend on toopt_params
    net = functools.partial(net_fn, net_params, key1)
    # train doing normal PGD + regularization of the sums of the Planet areas over all network neurons
    latent_repr = IdentityRepresentation(
        input_bounds, lambda x: x, fun_to_bound=net, rng=key, store_ibs=True)
    net_from_latent = functools.partial(net_fn, net_params)

    rng, key1, key2 = jax.random.split(rng, num=3)
    single_arg_loss_fn = functools.partial(loss_fn, targets)
    # Adversarial attack in the output space, using the latent_repr
    if args.train_att_n_steps > 0:
        adv_latent = attack_layer(
            latent_repr, net_from_latent, single_arg_loss_fn, args.train_att_n_steps, args.train_att_step_size, key1)
        adv_outs = net_from_latent(key2, adv_latent)
        adv_loss = single_arg_loss_fn(adv_outs)
        adv_ok = is_correctly_classified(adv_outs)
    else:
        # use natural loss
        adv_loss, adv_ok = nat_loss, nat_ok

    l1_loss = args.l1_reg * compute_l1_loss(net_params) if args.l1_reg > 0 else 0.
    tot_loss = (1 - kappa) * nat_loss + kappa * adv_loss + l1_loss

    if args.relu_stable is not None and args.relu_stable > 0:

        # Apply regularization on the sums of the Planet areas throughout the network
        out_bounds = latent_repr.get_next_preact_bounds()
        ibs = latent_repr.get_stored_ibs()
        bounds_list = ibs + [out_bounds] if args.relu_stable_max_ib is None else ibs[:args.relu_stable_max_ib]

        relu_reg = 0
        relu_stable = args.relu_stable
        for idx, next_bounds in enumerate(bounds_list):
            _, c_relu_reg = compute_relu_kw_area_loss(idx, relu_stable, next_bounds,
                                                      sum=(not args.relu_stable_ub_mask))
            relu_reg += relu_stable * c_relu_reg
            relu_stable *= args.relu_stable_factor

        if args.relu_stable_ub_mask:
            # ReLU-regularize only on robust points.
            logit_diffs = get_verif_net_fn(net_fn, jax.lax.stop_gradient(net_params), rng, targets, inputs)
            ub = jnp.min(logit_diffs, axis=1)  # if this is positive, the attack believes the net to be robust
            relu_reg = ((ub > 0.0) * relu_reg).sum()

        relu_reg /= len(bounds_list)  # (weighted) average of the sums over the layers
        tot_loss += kappa * relu_reg / args.train_batch

    return tot_loss, (nat_loss, nat_ok, adv_loss, adv_ok)


def get_train_step_fn(args, net_fn, loss_fn, opt, clip_to_0_1=True):
    # Define training step for the current latent space.
    @jax.jit
    def train_step(net_params, opt_state, latent_rng, inputs, eps, kappa, targets):
        # Compute fraction of outputs that are correctly classified
        is_correctly_classified = functools.partial(compute_correctly_classified, targets)

        clipper = lambda x: jnp.clip(x, 0., 1.) if clip_to_0_1 else x
        input_bounds = jax_verify.IntervalBound(clipper(inputs - eps), clipper(inputs + eps))

        # Get a function that evaluates layerwise_loss, and computes the grad of the first return value
        # w.r.t. toopt_params
        loss_val_and_grad = jax.value_and_grad(holistic_loss, argnums=0, has_aux=True)
        # Evaluate layerwise_loss, and get the gradient of the total loss w.r.t. toopt_params
        ((_, (nat_loss, nat_ok, adv_loss, adv_ok)),
         net_params_grad) = loss_val_and_grad(net_params, net_fn, args, input_bounds, inputs, targets, kappa, loss_fn,
                                              is_correctly_classified, latent_rng)

        # Step on the optimizer to minimize the loss.
        net_params_grad, opt_state = opt.update(net_params_grad, opt_state, net_params)
        net_params = optax.apply_updates(net_params, net_params_grad)

        return net_params, opt_state, (nat_loss, nat_ok, adv_loss, adv_ok)
    return train_step


def holistic_train(train_step, epoch, args, net_params, eps_sched, kappa_sched, opt_state, train_loader, rng):
    train_nat_loss, train_nat_ok, train_adv_loss, train_adv_ok, n_batches = 0, 0, 0, 0, 0

    pbar = tqdm(train_loader, dynamic_ncols=True)

    for batch_idx, (inputs, targets) in enumerate(pbar):
        # convert the current batch to jax (from numpy)
        inputs, targets = jnp.array(inputs), jnp.array(targets)

        rng, latent_rng = jax.random.split(rng)
        eps, kappa = eps_sched.get(), kappa_sched.get()

        # Perform a jitted update step over the parameters.
        net_params, opt_state, \
            (nat_loss, nat_ok, adv_loss, adv_ok) = train_step(
                net_params, opt_state, latent_rng, inputs, eps, kappa, targets
            )

        # Logging singular losses and statistics.
        train_nat_loss += nat_loss
        train_nat_ok += nat_ok
        train_adv_loss += adv_loss
        train_adv_ok += adv_ok

        n_batches += 1
        pbar.set_description(
            '[T] epoch=%d, nat_loss=%.4f, nat_ok=%.4f, adv_ok=%.4f, adv_loss=%.4f' % (
                epoch,
                train_nat_loss/n_batches,
                train_nat_ok/n_batches,
                train_adv_ok/n_batches,
                train_adv_loss/n_batches,
            )
        )
        eps_sched.advance_time(args.train_batch)
        kappa_sched.advance_time(args.train_batch)

    return net_params, opt_state, rng


def get_test_step_fn(args, net_fn, loss_fn, clip_to_0_1=True):
    # Define jitted testing step.
    @jax.jit
    def test_step(net_params, rng, inputs, targets):
        # Compute fraction of outputs that are correctly classified
        is_correctly_classified = functools.partial(compute_correctly_classified, targets)

        # Plug labels into loss function
        single_arg_loss_fn = functools.partial(loss_fn, targets)

        # Computes the natural loss and accuracy
        rng, key = jax.random.split(rng)
        nat_outs = net_fn(net_params, key, inputs)
        nat_ok = is_correctly_classified(nat_outs)
        nat_loss = loss_fn(targets, nat_outs)

        clipper = lambda x: jnp.clip(x, 0., 1.) if clip_to_0_1 else x
        input_bounds = jax_verify.IntervalBound(clipper(inputs - args.test_eps),
                                                clipper(inputs + args.test_eps))

        rng, key1, key2 = jax.random.split(rng, num=3)
        # report the standard PGD loss
        latent_repr = IdentityRepresentation(input_bounds, lambda x: x)
        net_from_latent = functools.partial(net_fn, net_params)

        rng, key1, key2 = jax.random.split(rng, num=3)
        # Adversarial attack in the latent space of layer_idx, using the latent_repr
        adv_latent = attack_layer(
            latent_repr, net_from_latent, single_arg_loss_fn, args.test_att_n_steps, args.test_att_step_size, key1)
        adv_outs = net_from_latent(key2, adv_latent)
        adv_loss = single_arg_loss_fn(adv_outs)
        adv_ok = is_correctly_classified(adv_outs)

        verified_loss = single_arg_loss_fn(-ibp_bounds(net_fn, net_params, input_bounds, targets))

        return nat_loss, nat_ok, adv_loss, adv_ok, verified_loss
    return test_step


def test(test_step, args, epoch, net_params, test_loader, rng):
    """
    Compute metrics/losses for a pre-trained network:
    - Natural loss (cross entropy by default)
    - Natural accuracy
    - PGD losses over the net's output space
    - fraction of outputs that are correctly classified from the PGD attack above
    """

    test_nat_loss, test_nat_ok, test_pgd_loss, test_pgd_ok, test_ver_loss, n_batches = 0, 0, 0, 0, 0, 0
    pbar = tqdm(test_loader)

    for inputs, targets in pbar:
        rng, latent_rng = jax.random.split(rng)
        # convert the current batch to jax (from numpy)
        inputs, targets = jnp.array(inputs), jnp.array(targets)

        nat_loss, nat_ok, adv_loss, adv_ok, ver_loss = test_step(net_params, latent_rng, inputs, targets)

        # Logging.
        test_nat_loss += nat_loss
        test_nat_ok += nat_ok
        test_pgd_loss += adv_loss
        test_pgd_ok += adv_ok
        test_ver_loss += ver_loss
        n_batches += 1
        pbar.set_description('[V] nat_loss=%.4f, nat_ok=%.4f, pgd_loss=%.4f, pgd_ok=%.4f, ver_loss={%s}' % (
            test_nat_loss/n_batches,
            test_nat_ok/n_batches,
            test_pgd_loss/n_batches,
            test_pgd_ok/n_batches,
            test_ver_loss/n_batches))

    return rng


def run_holistic_certified_training(args, clip_to_0_1=True):

    # Set random seed.
    torch.cuda.manual_seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.set_printoptions(precision=10)
    np.random.seed(args.random_seed)
    rng = jax.random.PRNGKey(args.random_seed)

    num_train, train_loader, test_loader, input_size, input_channel, n_class = get_loaders(args)
    steps_per_epoch = len(train_loader)

    rng, net_rng = jax.random.split(rng)
    net_params, net_fn = get_network(
        args.net, args.dataset, input_size, input_channel, n_class, net_rng, load_model=args.load_model, train=True)
    print('Number of parameters: ', jax.tree_util.tree_reduce(
        lambda x, y: x + y, jax.tree_util.tree_map(lambda x: x.size, net_params), 0))

    loss_fn = cross_entropy

    if args.train_mode == 'train':
        timestamp = int(time.time())
        model_dir = args.root_dir + 'models_new/%s/%s/%d/%s_%.5f/%d' % (args.dataset, args.exp_name, args.exp_id,
                                                                        args.net, args.train_eps, timestamp)
        print('Saving model to:', model_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        args_file = os.path.join(model_dir, 'args.json')
        with open(args_file, 'w') as fou:
            json.dump(vars(args), fou, indent=4)

        opt = optax.adam(args.lr) if args.opt == 'adam' else optax.sgd(args.lr, momentum=0.9)
        # Add scheduler on top of optimizer (transition_steps and transition_begin are multiplied by
        # steps_per_epoch as the scheduler acts as a function of the epoch number rather than the step number)

        if args.cont_lr_decay:
            scheduler = optax.exponential_decay(
                1.0, steps_per_epoch, args.lr_factor, staircase=True,
                transition_begin=int(args.cont_lr_mix) * args.mix_epochs * steps_per_epoch)
        else:
            # Staircase (discrete) exponential scheduler
            scheduler = optax.exponential_decay(
                1.0, args.lr_step * steps_per_epoch, args.lr_factor, staircase=True,
                transition_begin=args.mix_epochs * steps_per_epoch)

        opt = optax.chain(opt, optax.scale_by_schedule(scheduler))

        # Set utility variables employed for the layerwise training, print training progress.
        eps = args.start_eps_factor * args.train_eps
        kappa_sched = Scheduler(
            0.0, 1.0, num_train * args.mix_epochs, num_train * args.warmup_epochs)
        eps_sched = Scheduler(
            0 if args.anneal else eps, eps, num_train * args.mix_epochs, num_train * args.warmup_epochs)
        layer_dir = '{}/'.format(model_dir)

        # Initialize optimizer state
        opt_state = opt.init(net_params)
        # Get the training function along with the initialization states for the current latent training loop
        train_step = get_train_step_fn(args, net_fn, loss_fn, opt, clip_to_0_1=clip_to_0_1)
        # Get the testing function for the current latent training loop
        test_step = get_test_step_fn(args, net_fn, loss_fn, clip_to_0_1=clip_to_0_1)

        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir)
        for epoch in range(args.n_epochs):
            # Train over the current subset of the network, and return the updated net parameters (all)
            net_params, opt_state, rng = holistic_train(
                train_step, epoch, args, net_params, eps_sched, kappa_sched, opt_state, train_loader, rng)

            # Compute validation statistics.
            if (epoch+1) % args.test_freq == 0:
                rng = test(test_step, args, epoch, net_params, test_loader, rng)

        with open(os.path.join(layer_dir, 'net_%d.pt' % (epoch + 1)), 'wb') as f:
            # Save final network parameters.
            pickle.dump(net_params, f)

    elif args.train_mode == 'test':
        test_step = get_test_step_fn(args, net_fn, loss_fn, clip_to_0_1=clip_to_0_1)
        test(test_step, args, 0, net_params, test_loader, rng)
    else:
        assert False, 'Unknown mode: {}!'.format(args.train_mode)


def main():
    args = get_args()
    run_holistic_certified_training(args)


if __name__ == '__main__':

    main()
