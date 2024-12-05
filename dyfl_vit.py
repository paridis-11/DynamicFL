import os
import random
import time

import torch
import ast

from Data import Data
from models.dyrep import DyRep, build_optimizer, get_params
from Node.Node import Global_Node, Node
from Trainer import Trainer
from utils.utils import (LR_scheduler, Recorder, Summary, get_log_file_name,
                         init_args, print_memory_usage, generate_node_list)

# init 
args = init_args()
lr_initial = args.lr
args.type = 'VIT'
args.shape = 224  # For Vit
args.capacity_values = generate_node_list(args)



if args.wandb ==1:
    import wandb
    run_name = f"{args.dataset}_num-{args.node_num}_lepoch-{args.E}_lr-{args.lr}_note-{args.notes}"
    wandb.init(project="DyFL", name = run_name, entity="paridis")
    config_dict = vars(args)
    wandb.config.update(config_dict)


Data = Data(args)
Train = Trainer(args)
recorder = Recorder(args)
file_name  = get_log_file_name(args, directory = "logs/log2410")


# init nodes
Global_node = Global_Node(Data.test_all, args)
Edge_nodes = [Node(k, Data.train_loader[k], Data.test_loader, args, capacity=args.capacity_values[k]) for k in range(args.node_num)]
device = args.device


# train
for rounds in range(args.R): 
    Summary(args)
    print('===============The {:d}-th round==============='.format(rounds + 1))
    args.lr = LR_scheduler(lr_initial, rounds, args.R)
    for k in range(len(Edge_nodes)):
        print(f'---------- Rounds: {rounds+1}, Node: {k+1}, Notes: {args.notes} ---------------')
        Edge_nodes[k].fork(Global_node)          # edge_node get global model
        Edge_nodes[k].model.expand(num_branches = args.capacity_values[k])  # expand branches according to capacity_values[k]
        for epoch in range(args.E):
            # Edge_nodes[k].model.expand(num_branches = args.capacity_values[k])  # expand branches according to capacity_values[k]
            Train(Edge_nodes[k], epoch, rounds)
        
        Edge_nodes[k].model.merge()     # merge the expand branches
        Global_node.merge_now(Edge_nodes[k])
        Edge_nodes[k].delete_model()
    
    recorder.validate(Global_node)
    recorder.printer(Global_node, file_name = file_name, rounds = rounds)




Summary(args)