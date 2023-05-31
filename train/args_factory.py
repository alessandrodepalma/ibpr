import argparse


def get_args():
    parser = argparse.ArgumentParser()
    
    # Basic arguments
    parser.add_argument('--train_mode', default='train', type=str, help='whether to train adversarially')
    parser.add_argument('--dataset', default='cifar10', help='dataset to use')
    parser.add_argument('--net', required=True, type=str, help='network to use')
    parser.add_argument('--train_batch', default=100, type=int, help='batch size for training')
    parser.add_argument('--test_batch', default=100, type=int, help='batch size for testing')
    parser.add_argument('--layers', required=False, default=None, type=int, nargs='+', help='layer indices for training')
    parser.add_argument('--n_epochs', default=1, type=int, help='number of epochs')
    parser.add_argument('--mix_epochs', default=1, type=int, help='number of epochs to anneal schedule')
    parser.add_argument('--warmup_epochs', default=0, type=int, help='number of epochs to do standard training')
    parser.add_argument('--n_epochs_reduce', default=0, type=int, help='number of epochs to reduce each layer')
    parser.add_argument('--load_model', default=None, type=str, help='model to load')
    parser.add_argument('--n_valid', default=None, type=int, help='number of validation samples (none to use no validation)')
    parser.add_argument('--test_freq', default=50, type=int, help='frequency of testing')
    parser.add_argument('--random_seed', default=0, type=int, help='random_seed')

    # Optimizer and learning rate scheduling
    parser.add_argument('--opt', default='adam', type=str, help='optimizer to use')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_step', default=10, type=int, help='number of epochs between lr updates')
    parser.add_argument('--lr_factor', default=0.5, type=float, help='factor by which to decrease learning rate')
    parser.add_argument('--lr_layer_dec', default=0.5, type=float, help='factor by which to decrease learning rate in the next layers')
    parser.add_argument('--cont_lr_decay', action='store_true', help='whether to continuously (as opposed to stairwise-like) decay the step size')
    parser.add_argument('--cont_lr_mix', action='store_true', help='whether to postpone the lr decay for the mixing epochs when doing cont_lr_decay')
    parser.add_argument('--loop_stages', default=1, type=int, help='number of loops over the stages')
    parser.add_argument('--loop_lr_decr', default=1.0, type=float, help='LR decrease factor at each loop over stages')

    # Losses and regularizers
    parser.add_argument('--nat_factor', default=0.0, type=float, help='factor for natural loss')
    parser.add_argument('--relu_stable', required=False, type=float, default=None, help='factor for relu stability')
    parser.add_argument('--relu_stable_factor', required=False, type=float, default=1.0, help='factor for relu stability')
    parser.add_argument('--relu_stable_max_ib', required=False, type=int, default=None,
                        help='whether to limit ReLU regularization to a certain layer for holistic training')
    parser.add_argument('--relu_stable_ub_mask', action='store_true', help='whether to mask neg. UBs regularization')
    parser.add_argument('--l1_reg', default=0.0, type=float, help='l1 regularization coefficient')
    parser.add_argument('--mix', action='store_true', help='whether to mix adversarial and standard loss')
    parser.add_argument('--dropout', action='store_true', help='whether to use a dropout layer after the representation')

    # Configuration of adversarial attacks
    parser.add_argument('--train_eps', default=None, type=float, help='epsilon to train with')
    parser.add_argument('--test_eps', default=None, type=float, help='epsilon to verify')
    parser.add_argument('--anneal', action='store_true', help='whether to anneal epsilon')
    parser.add_argument('--eps_factor', default=1.05, type=float, help='factor to increase epsilon per layer')
    parser.add_argument('--start_eps_factor', default=1.0, type=float, help='factor to determine starting epsilon')
    parser.add_argument('--train_att_n_steps', default=10, type=int, help='number of steps for the attack')
    parser.add_argument('--train_att_step_size', default=0.25, type=float, help='step size for the attack (relative to epsilon)')
    parser.add_argument('--test_att_n_steps', default=None, type=int, help='number of steps for the attack')
    parser.add_argument('--test_att_step_size', default=None, type=float, help='step size for the attack (relative to epsilon)')

    # Configuration of loose LB on certified accuracy to compute at validation
    parser.add_argument('--do_not_ver', action='store_true', help='if active, avoids train-time verification (faster)')

    # Metadata
    parser.add_argument('--exp_name', default='dev', type=str, help='name of the experiment')
    parser.add_argument('--exp_id', default=1, type=int, help='name of the experiment')
    parser.add_argument('--root_dir', required=False, default='./', type=str, help='directory to store the data')

    args = parser.parse_args()

    if args.test_eps is None:
        args.test_eps = args.train_eps
    if args.test_att_n_steps is None:
        args.test_att_n_steps = args.train_att_n_steps
        args.test_att_step_size = args.train_att_step_size

    return args
