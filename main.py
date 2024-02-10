import torch
import time
import os
import random
from logger import Logger
from Node import Node, Global_Node
from Data import Data
from utils import LR_scheduler, Recorder, Summary, init_args
from Trainer import Trainer
from dyrep import DyRep, build_optimizer, get_params



# init 
args = init_args()
Data = Data(args)
Train = Trainer(args)
recorder = Recorder(args)

logger_pre = Logger(args)


# init nodes
Global_node = Global_Node(Data.test_all, args)
Edge_nodes = [Node(k, Data.train_loader[k], Data.test_loader, args, capacity=args.capacity_values[k]) for k in range(args.node_num)]
DyFL = [Node(k, Data.train_loader[k], Data.test_loader, args) for k in range(args.node_num)]


# train
for rounds in range(args.R): 
    print('===============The {:d}-th round==============='.format(rounds + 1))
    LR_scheduler(rounds, Edge_nodes, args)
    for k in range(len(Edge_nodes)):
        print(f'---------- Rounds: {rounds+1}, Node: {k+1} ---------------')
        Edge_nodes[k].fork(Global_node)
        for epoch in range(args.E):
            Train(Edge_nodes[k], epoch, rounds)
        
        Edge_nodes[k].convert()

    Global_node.merge(Edge_nodes)
    
    recorder.validate(Global_node)
    recorder.printer(Global_node)
    logger_pre.write(rounds=rounds + 1, test_acc=recorder.val_acc[str(Global_node.num)][rounds])

recorder.finish()
logger_pre.close()

Summary(args)

