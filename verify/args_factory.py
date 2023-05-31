import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Perform greedy layerwise training.')
    parser.add_argument('--dataset', default='cifar10', help='dataset to use')
    parser.add_argument('--net', required=True, type=str, help='network to use')
    parser.add_argument('--test_eps', required=True, type=float, help='epsilon to verify')
    parser.add_argument('--load_model', type=str, help='model to load')
    parser.add_argument('--random_seed', default=0, type=int, help='random_seed')

    parser.add_argument('--test_att_n_steps', default=None, type=int, help='number of steps for the attack')
    parser.add_argument('--test_att_step_size', default=None, type=float, help='step size for the attack (relative to epsilon)')
    parser.add_argument('--attack_restarts', default=20, type=int, help='number of restarts for the attack')

    parser.add_argument('--n_valid', default=None, type=int, help='number of test samples')
    parser.add_argument('--n_train', default=None, type=int, help='number of training samples to use')
    parser.add_argument('--test_batch', default=1, type=int, help='batch size for testing')

    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--no_load', action='store_true', help='verify from scratch')
    parser.add_argument('--start_idx', default=0, type=int, help='specific index to start')
    parser.add_argument('--end_idx', default=-1, type=int, help='specific index to end or -1 to do all')

    parser.add_argument('--ib_batch_size', default=512, type=int, help='number of ibs that can be computed at once')
    parser.add_argument('--oval_bab_config', help='OVAL BaB config file')
    parser.add_argument('--oval_bab_timeout', default=60, type=int, help='number of [s] to run OVAL BaB for')
    parser.add_argument('--pickle_results', default=None, type=str,
                        help='Folder for a pickle with the verification results in the stile of OVAL BaB')

    return parser.parse_args()
