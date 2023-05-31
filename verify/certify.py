import copy
import os
import pickle
import torch
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import jax_verify
from tqdm import tqdm
from train.loaders import get_loaders
from train.utils import get_network
from verify.args_factory import get_args
from verify.ovalbab_utils import get_pytorch_network, run_oval_bab, create_1_vs_all_verification_problem


def report(args, pbar, ver_logdir, tot_verified_corr, tot_nat_ok, tot_pgd_ok, test_idx, tot_tests, test_data):
    """ Logs evaluation statistics to standard output. """
    pbar.set_description(
        'tot_tests: %d, verified: %.5lf [%d/%d], nat_ok: %.5lf [%d/%d], pgd_ok: %.5lf [%d/%d]' % (
            tot_tests,
            tot_verified_corr/tot_tests, tot_verified_corr, tot_tests,
            tot_nat_ok/tot_tests, tot_nat_ok, tot_tests,
            tot_pgd_ok/tot_tests, tot_pgd_ok, tot_tests,
        )
    )
    out_file = os.path.join(ver_logdir, '{}.p'.format(test_idx))
    pickle.dump(test_data, open(out_file, 'wb'))


def get_input_bounds(args, inputs, clip_to_0_1=True):
    clipper = lambda x: jnp.clip(x, 0., 1.) if clip_to_0_1 else x
    return jax_verify.IntervalBound(clipper(inputs - args.test_eps),
                                    clipper(inputs + args.test_eps))


def main():
    args = get_args()

    jax.config.update('jax_platform_name', 'cpu')

    # Set random seed.
    torch.cuda.manual_seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.set_printoptions(precision=10)
    np.random.seed(args.random_seed)
    rng = jax.random.PRNGKey(args.random_seed)

    # Create folder for logs.
    ver_logdir = args.load_model[:-3] + '_ver'
    if not os.path.exists(ver_logdir):
        os.makedirs(ver_logdir)

    # Load dataset and network.
    _, _, test_loader, input_size, input_channel, n_class = get_loaders(args, test_only=True)
    rng, net_rng = jax.random.split(rng)
    net_params, net_fn = get_network(
        args.net, args.dataset, input_size, input_channel, n_class, net_rng, load_model=args.load_model)
    print('Number of parameters: ', jax.tree_util.tree_reduce(
        lambda x, y: x + y, jax.tree_util.tree_map(lambda x: x.size, net_params), 0))

    if args.pickle_results:
        # Prepare pickle for results.
        if not os.path.exists(args.pickle_results):
            os.makedirs(args.pickle_results)
        json_name = os.path.basename(args.oval_bab_config.replace(".json", ""))
        record_name = args.pickle_results + os.path.basename(
            args.load_model.replace("/", "_")) + "_" + json_name + ".pkl"
        if os.path.isfile(record_name):
            results_table = pd.read_pickle(record_name)
        else:
            dataset_len = args.end_idx if args.end_idx == -1 else len(test_loader)
            indices = list(range(dataset_len))
            results_table = pd.DataFrame(
                index=indices, columns=["prop"] + [f'BSAT_{json_name}', f'BBran_{json_name}', f'BTime_{json_name}'])
            results_table.to_pickle(record_name)
    else:
        results_table, json_name, record_name = None, None, None

    rng, net_rng = jax.random.split(rng)
    # Convert net into Pytorch for OVAL BaB use (via ONNX).
    in_shape = (input_size, input_size, input_channel)
    onnx_filename = args.load_model + ".onnx"
    torch_verif_layers = get_pytorch_network(net_params, net_fn, in_shape, onnx_filename, net_rng)
    torch_net = torch.nn.Sequential(*[copy.deepcopy(lay).cuda() for lay in torch_verif_layers])

    # Remove objects that won't be re-used in the PyTorch version of the verification loop.
    del net_params
    del net_fn

    pbar = tqdm(test_loader, dynamic_ncols=True)
    tot_verified, tot_nat_ok, tot_pgd_ok, tot_tests = 0, 0, 0, 0
    for test_idx, (inputs, targets) in enumerate(pbar):
        if test_idx < args.start_idx or (args.end_idx != -1 and test_idx >= args.end_idx):
            continue

        if results_table is not None:
            if pd.isna(results_table.loc[test_idx]["prop"]) == False:
                print(f'the {test_idx}th element is done')
                continue

        tot_tests += 1
        test_file = os.path.join(ver_logdir, '{}.p'.format(test_idx))
        test_data = pickle.load(open(test_file, 'rb')) if (not args.no_load) and os.path.isfile(test_file) else {}

        # NOTE: in order for this to work, jax needs to operate on cpu. Prepend JAX_PLATFORM_NAME=cpu at execution.
        torch_inputs = torch.tensor(inputs)
        torch_targets = torch.tensor(targets)

        # Do forward pass in PyTorch.
        nat_outs = torch_net(torch_inputs.cuda()).cpu()
        nat_ok = torch_targets.eq(nat_outs.max(dim=1)[1]).item()

        # Logging.
        tot_nat_ok += nat_ok
        test_data['ok'] = nat_ok
        if not nat_ok:
            report(args, pbar, ver_logdir, tot_verified, tot_nat_ok, tot_pgd_ok, test_idx, tot_tests, test_data)
            continue

        # Use OVAL BaB for verification instead of jax_verify
        assert inputs.shape[0] == 1, "only test_batch=1 is supported for OVAL BaB"
        torch_inputs = torch_inputs.squeeze(0)
        torch_targets = torch_targets.squeeze(0)
        torch_input_bounds = torch.stack([
            (torch_inputs - args.test_eps).clamp(0, 1),
            (torch_inputs + args.test_eps).clamp(0, 1)], dim=-1)
        # Make problem a 1vsall robustness verification problem
        torch_verif_problem = create_1_vs_all_verification_problem(
            torch_verif_layers, torch_targets, torch_input_bounds, args.ib_batch_size)
        # release some memory
        torch.cuda.empty_cache()

        verified, pgd_ok = run_oval_bab(
            torch_verif_problem, torch_input_bounds, args.oval_bab_config, timeout=args.oval_bab_timeout,
            results_table=results_table, test_idx=test_idx, json_name=json_name, record_name=record_name)

        tot_pgd_ok += int(pgd_ok)
        test_data['pgd_ok'] = int(pgd_ok)

        tot_verified += int(verified)

        report(args, pbar, ver_logdir, tot_verified, tot_nat_ok, tot_pgd_ok, test_idx, tot_tests, test_data)


if __name__ == '__main__':
    main()
