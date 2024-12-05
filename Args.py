import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Total
    parser.add_argument('--algorithm', type=str, default='dyfl_vit',
                        help='Type of algorithms:{fed_mutual, fed_avg, fed_coteaching, normal, parallel}')
    parser.add_argument('--wandb', default=1, type=int)
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='device: {cuda, cpu}')
    parser.add_argument('--node_num', type=int, default=9,
                        help='Number of nodes')
    parser.add_argument('--R', type=int, default=100,
                        help='Number of rounds: R')
    parser.add_argument('--E', type=int, default=5,
                        help='Number of local epochs: E')
    parser.add_argument('--notes', type=str, default='',
                        help='Notes of Experiments')
    # parser.add_argument('--mu', type=float, default=0.2,
    #                     help='Degree of non-iid')
    parser.add_argument('--seed', type=int, default=42,
                    help='random seed (default: 42)')
    parser.add_argument('--shape', type=int, default=224,
                    help='random seed (default: 224, 32)')
    # parser.add_argument('--capacity_values', type=str, default="[0,0,0,1,1,1,2,2,2]",
    #                 help='capacity_values')
    parser.add_argument('--device_ratio', type=str, default="1:1:1",
                    help='device_ratio')
                        
# [1,1,1,1,1,1,1,4,4,8],
# [1,1,1,1,1,1,1,1,1],
# [0,0,0,1,1,1,2,2,2]
# [0,0,0,0,0,0,0,0,0]

    # Model
    parser.add_argument('--model', type=str, default='vit_base')
    # parser.add_argument('--local_model', type=str, default='nas_model')
    parser.add_argument('--model-config', type=str, default='vgg16_cifar10.yaml',
                    help='Path to net config.')
    parser.add_argument('--drop', type=float, default=0.0,
                    help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path-rate', type=float, default=0., 
                    help='Drop path rate, (default: 0.)')

    # Data
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='datasets: {cifar10, CancerSlides, mnist}')
    parser.add_argument('--partition', type=str, default='iid',
                        help='partition : {iid, dir}')
    parser.add_argument('--batchsize', type=int, default=16,
                        help='batchsize')
    parser.add_argument('--dir', type=float, default=0.5,
                        help='Degree of dirichlet')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='val_ratio')
    parser.add_argument('--all_data', type=bool, default=True,
                        help='use all train_set')
    parser.add_argument('--classes', type=int, default=10,
                        help='classes')
    parser.add_argument('--save_dir', type=str, default=None, help="name of save directory")
    # parser.add_argument('--sampler', type=str, default='iid', help="iid, non-iid")

    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer: {sgd, adam}')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0,
                        help='SGD momentum')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='local ratio of data loss')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='global ratio of data loss')
    parser.add_argument('--opt-eps', default=1e-8, type=float,
                    help='Optimizer Epsilon (default: 1e-8, use opt default)')

    parser.add_argument('--dyrep-recal-bn-iters', type=int, default=20,
                    help='how many iterations for recalibrating the bn states in dyrep')

    parser.add_argument('--opt_no_filter', action='store_true', default=True,
                    help='disable bias and bn filter of weight decay')
    

    args = parser.parse_args()
    return args
