import copy
# from torch.cuda import random
import random
from re import S

import torch
import torch.nn as nn
from numpy import s_

import models.Model as Model
from models.builder import build_model
from models.dbb_block import DiverseBranchBlock
from models.dyrep import DyRep, build_optimizer
from models.recal_bn import recal_bn


def init_model(model_type):
    model = []
    if model_type == 'LeNet5':
        model = Model.LeNet5()
    elif model_type == 'MLP':
        model = Model.MLP()
    elif model_type == 'ResNet18':
        model = Model.ResNet18()
    elif model_type == 'CNN':
        model = Model.CNN()
    return model


def init_optimizer(model, args):
    optimizer = []
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-2)
    return optimizer





class Node(object):
    def __init__(self, num, train_loader, test_data, args, capacity=1):
        self.args = args
        self.num = num + 1
        self.capacity = capacity
        self.device = self.args.device
        self.train_data = train_loader
        self.test_data = test_data
        # self.model = build_model(args, args.model).to(args.device)
        # if self.args.type == 'CNN':
        #     self.interaction_model = copy.deepcopy(self.model).to(self.device)
        # self.optimizer = init_optimizer(self.model, self.args)

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.args.R)



    def fork(self, global_node):
        self.model = copy.deepcopy(global_node.model).to(self.device)
        self.optimizer = init_optimizer(self.model, self.args)
        if self.args.type == 'CNN':
            self.interaction_model = copy.deepcopy(self.model).to(self.device)

    def delete_model(self):
        # 删除模型以释放显存
        del self.model
        del self.optimizer
        torch.cuda.empty_cache()  # 清空未使用的显存缓存


    def convert(self):
        self.model = copy.deepcopy(self.interaction_model).to(self.device)
        for m in self.model.modules():
            if isinstance(m, DiverseBranchBlock):
                m.switch_to_deploy()


    def adjust(self):
        self.model = copy.deepcopy(self.model).to(self.device)
        self.optimizer = init_optimizer(self.model, self.args)



class Select_Node(object):
    def __init__(self, args):
        self.args = args
        self.s_list = []   
        self.c_list = []   
        self.node_list = list(range(args.node_num))
        self.max_lost = args.max_lost   

        for j in range(self.max_lost):
            self.s_list.extend(self.node_list)
    

    def random_select(self):
        index = random.randrange(len(self.s_list))      
        chosen_number = self.s_list.pop(index)          
        self.c_list.append(chosen_number)               
        print(self.c_list)

        if len(set(self.c_list)) == self.args.node_num :
            self.s_list.extend(self.node_list)          
            [self.c_list.remove(i) for i in range(self.args.node_num)]     
        return chosen_number


class Global_Node(object):
    def __init__(self, test_data, args):
        self.num = 0
        self.args = args
        self.device = self.args.device
        self.model = build_model(args, args.model).to(args.device)

        self.test_data = test_data  
        self.Dict = self.model.state_dict()

        # self.edge_node = [build_model(args, args.local_model).to(args.device) for k in range(args.node_num)]
        self.init = False
        self.save = []


    def merge(self, Edge_nodes):
        Node_State_List = [copy.deepcopy(Edge_nodes[i].model.state_dict()) for i in range(len(Edge_nodes))]
        # 局部模型切换回全局模型结构后，其模型内部依然含有之前的那些层，需要把那些层删掉。
        for i in range(len(Edge_nodes)):
            # keys_to_remove = [key for key in Node_State_List[i].keys() if "conv.dbb" in key]
            keys_to_remove = [key for key in Node_State_List[i].keys() if "dbb" in key]
            for key in keys_to_remove:
                # 删除扩增层
                del Node_State_List[i][key]
            # 更改合并的deployed卷积的名字
            Node_State_List[i] = {key.replace('conv_deployed.', ''): value for key, value in Node_State_List[i].items()}

        self.Dict = Node_State_List[0]
        # print('Edge_nodes[i]):', self.Dict.keys())
        # 聚合
        for key in self.Dict.keys():
            for i in range(1, len(Edge_nodes)):
                self.Dict[key] += Node_State_List[i][key]
            self.Dict[key] = self.Dict[key].float()
            self.Dict[key] /= len(Edge_nodes)

        self.model.load_state_dict(self.Dict)

    def merge_now(self, Edge_node):
        # 为了节省显存，因此在每次局部更新结束后就执行融合，从而可以释放掉客户端模型所占显存
        Edge_node_State_List = Edge_node.model.state_dict()
        current_num = Edge_node.num
        if current_num == 1:
            for key in self.Dict:
                self.Dict[key] = torch.zeros_like(self.Dict[key])
            print('第一个节点，零初始化')

        for key in self.Dict.keys():
            self.Dict[key] += Edge_node_State_List[key]
            self.Dict[key] = self.Dict[key].float()

            if current_num == self.args.node_num:
                self.Dict[key] /= self.args.node_num


        if current_num == self.args.node_num:
            self.model.load_state_dict(self.Dict)
            print('更新至全局模型')


    def cnn_merge_now(self, Edge_node):
        # 为了节省显存，因此在每次局部更新结束后就执行融合，从而可以释放掉客户端模型所占显存
        Edge_node_State_List = Edge_node.model.state_dict()

        keys_to_remove = [key for key in Edge_node_State_List.keys() if "dbb" in key]
        for key in keys_to_remove:
            # 删除扩增层
            del Edge_node_State_List[key]
        # 更改合并的deployed卷积的名字
        Edge_node_State_List = {key.replace('conv_deployed.', ''): value for key, value in Edge_node_State_List.items()}


        current_num = Edge_node.num
        if current_num == 1:
            for key in self.Dict:
                self.Dict[key] = torch.zeros_like(self.Dict[key])
            print('第一个节点，零初始化')

        for key in self.Dict.keys():
            self.Dict[key] += Edge_node_State_List[key]
            self.Dict[key] = self.Dict[key].float()

            if current_num == self.args.node_num:
                self.Dict[key] /= self.args.node_num


        if current_num == self.args.node_num:
            self.model.load_state_dict(self.Dict)
            print('更新至全局模型')







        



    # def update(self, Edge_node):

    #     self.edge_node[Edge_node.num-1] = Edge_node.model

    # def init_processing(self):
    #     assert self.init
    #     ## warmup
    #     Node_State_List = [copy.deepcopy(self.edge_node[i].state_dict()) for i in self.save]
    #     self.Dict = Node_State_List[0]
    #     for key in self.Dict.keys():
    #         if 'num_batches_tracked' in key:
    #             continue

    #         for i in range(1, len(Node_State_List)):
    #             self.Dict[key] += Node_State_List[i][key]

    #         self.Dict[key] /= float(len(Node_State_List))

    #     self.model.load_state_dict(self.Dict)

    # def processing(self):

    #     Node_State_List = [copy.deepcopy(self.edge_node[i].state_dict()) for i in range(self.args.node_num)]
    #     self.Dict = Node_State_List[0]
    #     for key in self.Dict.keys():
    #         if 'num_batches_tracked' in key:
    #             continue

    #         for i in range(1, self.args.node_num):
    #             self.Dict[key] += Node_State_List[i][key]

    #         self.Dict[key] /= self.args.node_num
    #     self.model.load_state_dict(self.Dict)

        




