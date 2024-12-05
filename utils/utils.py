import os
import random
from datetime import datetime

import numpy as np
import torch
import math

from Args import args_parser

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()



def get_log_file_name(args, directory="logs"):
    # 获取当前时间，格式为 YYYYMMDD_HHMMSS
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 生成文件名，格式为 dataset_model_YYYYMMDD_HHMMSS.log
    file_name = f"{args.dataset}_{args.model}_{current_time}.log"
    
    # 确保日志保存的目录存在，如果不存在则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 返回完整的文件路径
    return os.path.join(directory, file_name)

class Recorder(object):
    def __init__(self, args):
        self.args = args
        self.counter = 0
        self.tra_loss = {}
        self.tra_acc = {}
        self.val_loss = {}
        self.val_acc = {}
        for i in range(self.args.node_num + 1):
            self.val_loss[str(i)] = []
            self.val_acc[str(i)] = []
            # self.val_loss[str(i)] = []
            # self.val_acc[str(i)] = []
        self.acc_best = torch.zeros(self.args.node_num + 1)
        self.get_a_better = torch.zeros(self.args.node_num + 1)

    def validate(self, node):
        self.counter += 1
        node.model.to(node.device).eval()
        total_loss = 0.0
        correct = 0.0
        pred_res = []
        target_res = []

        with torch.no_grad():
            for idx, (data, target) in enumerate(node.test_data):
                # data, target = data.to(node.device), target.to(node.device).squeeze(dim=1)
                data = data.to(node.device)
                if target.dim() > 1 and target.size(1) == 1:
                    target = target.long().to(node.device).squeeze(dim=1)
                else:
                    target = target.long().to(node.device)
                if node.args.dataset.lower() == 'chestxray':
                    target = (target.sum(dim=1) > 0).long()
                output = node.model(data)
                total_loss += torch.nn.CrossEntropyLoss()(output, target)
                pred = output.argmax(dim=1)
                correct += pred.eq(target.view_as(pred)).sum().item()
                pred_res.append(pred)
                target_res.append(target)

            total_loss = total_loss / (idx + 1)

            # acc = correct / len(node.test_data.dataset) * 100
            acc = round(correct / len(node.test_data.dataset) * 100, 2)
            pred_res = torch.cat(pred_res)
            target_res = torch.cat(target_res)
            prec = []
            for i in range(10):
                mask = target_res == i
                idx = np.where(mask.cpu().numpy())[0]
                c_ac = sum(pred_res[idx] == target_res[idx])/sum(mask)
                prec.append(float(c_ac.cpu().numpy()))
            #print(prec)

        self.val_loss[str(node.num)].append(total_loss.item())
        self.val_acc[str(node.num)].append(acc)

        if self.val_acc[str(node.num)][-1] > self.acc_best[node.num]:
            self.get_a_better[node.num] = 1
            self.acc_best[node.num] = self.val_acc[str(node.num)][-1]
            # torch.save(node.model.state_dict(),
            #            './saves/model/Node{:d}_{:s}.pth'.format(node.num, node.args.local_model))

    def printer(self, node, file_name, rounds):
        if self.get_a_better[node.num] == 1:
            # print('Node{:d}: A Better Accuracy: {:.2f}%! Model Saved!'.format(node.num, self.acc_best[node.num]))

            self.get_a_better[node.num] = 0
        if node.num == 0:
            print(f'中央服务器的准确率: {self.val_acc[str(node.num)]}')
            print(f'中央服务器的loss: {self.val_loss[str(node.num)]}')
            print('中央服务器的Best Accuracy = {:.2f}%'.format(self.acc_best[node.num]))
            with open(file_name, 'a') as log_file:
                log_file.write(f"当前回合: {rounds}, Global_Node Acc: {self.val_acc[str(node.num)][-1]}, 历史精度：{self.val_acc[str(node.num)]}\n")
            if node.args.wandb ==1:
                import wandb
                wandb.log({node.args.dataset + "_loss": self.val_loss[str(node.num)][rounds]}, step = rounds)
                wandb.log({node.args.dataset + "_acc": self.val_acc[str(node.num)][rounds]}, step = rounds)


        
        else:
            print(f'节点 {node.num} 准确率: {self.val_acc[str(node.num)]}')
            print(f'节点 {node.num} loss: {self.val_loss[str(node.num)]}')



    def finish(self, file_name):
        # torch.save([self.val_loss, self.val_acc],
                #    './saves/record/loss_acc_{:s}_{:s}.pt'.format(self.args.algorithm, self.args.notes))
        print('Finished!\n')
        for i in range(self.args.node_num + 1):
            print('Node{}: Best Accuracy = {:.2f}%'.format(i, self.acc_best[i]))
            with open(file_name, 'a') as log_file:
                # log_file.write(f"Node{i}: Best Accuracy = {round(self.acc_best[i].item(), 2)}\n")
                log_file.write(f"Node{i}: Final Accuracy = {self.val_acc[str(i)][-1]}\n")


# def LR_scheduler(rounds, Edge_nodes, args):

#     for i in range(len(Edge_nodes)):
#         Edge_nodes[i].args.lr = args.lr
#         Edge_nodes[i].args.alpha = args.alpha
#         Edge_nodes[i].args.beta = args.beta
#         Edge_nodes[i].optimizer.param_groups[0]['lr'] = args.lr
    
#     # print('Learning rate={:.4f}'.format(args.lr))


# def LR_scheduler2(rounds, Edge_nodes, args):
#     trigger = int(args.R / 3)
#     if rounds != 0 and rounds % trigger == 0 and rounds < args.stop_decay:
#         args.lr *= 0.1
#         # args.alpha += 0.2
#         # args.beta += 0.4
#         for i in range(len(Edge_nodes)):
#             Edge_nodes[i].args.lr = args.lr
#             Edge_nodes[i].args.alpha = args.alpha
#             Edge_nodes[i].args.beta = args.beta
#             Edge_nodes[i].optimizer.param_groups[0]['lr'] = args.lr
    
#     print('Learning rate={:.4f}'.format(args.lr))



def get_params(model, ignore_auxiliary_head=True):
    if not ignore_auxiliary_head:
        params = sum([m.numel() for m in model.parameters()])
    else:
        params = sum([m.numel() for k, m in model.named_parameters() if 'auxiliary_head' not in k])
    return params

def get_flops(model, input_shape=(3, 224, 224)):
    if hasattr(model, 'flops'):
        return model.flops(input_shape)
    else:
        return get_flops_hook(model, input_shape)

def get_flops_hook(model, input_shape=(3, 224, 224)):
    is_training = model.training
    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        assert self.in_channels % self.groups == 0

        kernel_ops = self.kernel_size[0] * self.kernel_size[
            1] * (self.in_channels // self.groups)
        params = output_channels * kernel_ops
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement()

        flops = batch_size * weight_ops
        list_linear.append(flops)

    def foo(net, hook_handle):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                hook_handle.append(net.register_forward_hook(conv_hook))
            if isinstance(net, torch.nn.Linear):
                hook_handle.append(net.register_forward_hook(linear_hook))
            return
        for c in childrens:
            foo(c, hook_handle)

    hook_handle = []
    foo(model, hook_handle)
    input = torch.rand(*input_shape).unsqueeze(0).to(next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        out = model(input)
    for handle in hook_handle:
        handle.remove()

    total_flops = sum(sum(i) for i in [list_conv, list_linear])
    model.train(is_training)
    return total_flops


def LR_scheduler(lr_initial, current_epoch, total_epochs):

    lr_new = lr_initial * 0.5 * (1 + math.cos(math.pi * current_epoch / total_epochs))
    
    return lr_new



def init_args():
    args = args_parser()
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('Running on', args.device)
    return args


def Summary(args):
    print("Summary：\n")
    print("dataset:{}\t batchsize:{}\n".format(args.dataset, args.batchsize))
    print("node_num:{}，\t local model:{},\n".format(args.node_num, args.model))
    # print("iid:{},\tequal:{},\n".format(args.iid == 1, args.unequal == 0))
    print("global epochs:{},\t local epochs:{},\n".format(args.R, args.E))
    print("lr:{}，\t optimizer:{},\n".format(args.lr, args.optimizer))


def print_memory_usage(device):
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    print(f"GPU {device}:")
    print(f"    Allocated memory: {allocated:.2f} MB")
    print(f"    Reserved memory : {reserved:.2f} MB")


def generate_node_list(args):
    device_ratio = args.device_ratio
    ratios = list(map(int, device_ratio.split(":")))
    
    total_ratio = sum(ratios)
    
    if args.node_num % total_ratio != 0:
        raise ValueError(f"args.node_num must be a multiple of {total_ratio} to satisfy the specified ratio {device_ratio}")
    
    base_count = args.node_num // total_ratio
    counts = [r * base_count for r in ratios]
    
    capacity_values = []
    for i, count in enumerate(counts):
        capacity_values.extend([i] * count)
    
    random.shuffle(capacity_values)
    
    return capacity_values