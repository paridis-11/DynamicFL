import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Total
    parser.add_argument('--algorithm', type=str, default='dyfl',
                        help='Type of algorithms')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device')
    parser.add_argument('--node_num', type=int, default=10,
                        help='Number of nodes')
    parser.add_argument('--R', type=int, default=100,
                        help='Number of rounds: R')
    parser.add_argument('--E', type=int, default=3,
                        help='Number of local epochs: E')
    parser.add_argument('--notes', type=str, default='',
                        help='Notes of Experiments')
    parser.add_argument('--max_lost', type=int, default=1,
                        help='The difference in the number of communication rounds between the fastest and slowest nodes ')
    parser.add_argument('--warmup', type=int, default=5,
                        help='The number of warmup')
    parser.add_argument('--mu', type=float, default=0.2,
                        help='Degree of non-iid')
    parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
    parser.add_argument('--capacity_values', type=list, default=[1,1,1,1,1,1,1,4,4,8],
                    help='capacity_values')
                        

    # Model
    parser.add_argument('--global_model', type=str, default='nas_model')
    parser.add_argument('--local_model', type=str, default='nas_model')
    parser.add_argument('--model-config', type=str, default='vgg16_cifar10.yaml',
                    help='Path to net config.')
    parser.add_argument('--drop', type=float, default=0.0,
                    help='Dropout rate')
    parser.add_argument('--drop-path-rate', type=float, default=0., 
                    help='Drop path rate')

    # Data
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='datasets')
    parser.add_argument('--batchsize', type=int, default=128,
                        help='batchsize')

    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='val_ratio')
    parser.add_argument('--all_data', type=bool, default=True,
                        help='use all train_set')
    parser.add_argument('--classes', type=int, default=10,
                        help='classes')
    parser.add_argument('--save_dir', type=str, default=None, help="name of save directory")
    parser.add_argument('--sampler', type=str, default='iid', help="iid, non-iid")

    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_step', type=int, default=10,
                        help='learning rate decay step size')
    parser.add_argument('--stop_decay', type=int, default=50,
                        help='round when learning rate stop decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='local ratio of data loss')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='global ratio of data loss')
    parser.add_argument('--opt-eps', default=1e-8, type=float,
                    help='Optimizer Epsilon')

    parser.add_argument('--dyrep-recal-bn-iters', type=int, default=20,
                    help='how many iterations for recalibrating the bn states in dyrep')

    parser.add_argument('--opt_no_filter', action='store_true', default=True,
                    help='disable bias and bn filter of weight decay')
    

    args = parser.parse_args()
    return args
