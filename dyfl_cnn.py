import os
import random
import time

import torch

from Data import Data
from models.dyrep import DyRep, build_optimizer, get_params
from Node.Node import Global_Node, Node
from Trainer import Trainer
from utils.utils import (LR_scheduler, Recorder, Summary, get_log_file_name,
                         init_args, print_memory_usage, generate_node_list)
# init 
args = init_args()
args.type = 'CNN'
args.algorithm = 'CNN'
args.model  = 'resnet18'
args.shape = 32
lr_initial = args.lr
args.capacity_values = generate_node_list(args)


if args.wandb ==1:
    import wandb
    # 构建运行名称，包含超参数信息
    run_name = f"{args.dataset}_num-{args.node_num}_lepoch-{args.E}_lr-{args.lr}_note-{args.notes}"
    wandb.init(project="DyFL", name = run_name, entity="paridis")
    config_dict = vars(args)  # 将 args 转换为字典
    wandb.config.update(config_dict)  # 更新 config 的属性

Data = Data(args)
Train = Trainer(args)
recorder = Recorder(args)
file_name  = get_log_file_name(args, directory = "logs/log2410")



# init nodes
Global_node = Global_Node(Data.test_all, args)
Edge_nodes = [Node(k, Data.train_loader[k], Data.test_loader, args, capacity=args.capacity_values[k]) for k in range(args.node_num)]
# DyFL = [Node(k, Data.train_loader[k], Data.test_loader, args) for k in range(args.node_num)]

device = args.device
print_memory_usage(device)

# train
for rounds in range(args.R): 
    Summary(args)
    print('===============The {:d}-th round==============='.format(rounds + 1))
    args.lr = LR_scheduler(lr_initial, rounds, args.R)
    for k in range(len(Edge_nodes)):
        print(f'---------- Rounds: {rounds+1}, Node: {k+1} ---------------')
        Edge_nodes[k].fork(Global_node)
        for epoch in range(args.E):
            Train(Edge_nodes[k], epoch, rounds)
        
        Edge_nodes[k].convert()
        Global_node.cnn_merge_now(Edge_nodes[k])
        Edge_nodes[k].delete_model()
    # Global_node.merge(Edge_nodes)
    
    recorder.validate(Global_node)
    recorder.printer(Global_node, file_name = file_name, rounds = rounds)




Summary(args)