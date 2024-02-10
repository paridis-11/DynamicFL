import torch.nn as nn
import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
from misc import AverageMeter, accuracy
# from losses import GCELoss
from measure import  get_flops, get_params
import time
from recal_bn import recal_bn
from dyrep import DyRep, build_optimizer

from builder import build_model
from dbb_block import DiverseBranchBlock
from dyrep import DyRep
CE_Loss = nn.CrossEntropyLoss()




def train_avg(node, epoch):
    node.model.to(node.device).train()
    train_loader = node.train_data
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    description = "Node{:d}: loss={:.4f} acc={:.2f}%"
    with tqdm(train_loader) as epochs:
        for idx, (data, target) in enumerate(epochs):
            node.optimizer.zero_grad()
            epochs.set_description(description.format(node.num, avg_loss, acc))
            data, target = data.to(node.device), target.to(node.device)
            output = node.model(data)
            loss = CE_Loss(output, target)
            loss.backward()
            node.optimizer.step()
            total_loss += loss
            avg_loss = total_loss / (idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100





def train_dyfl(node, epoch, round):
    node.interaction_model.to(node.device).train()
    train_loader = node.train_data
    loss_fn = nn.CrossEntropyLoss()
    loss_m = AverageMeter()
    val_loss_fn = loss_fn
    total_loss = 0.0
    avg_loss = 0.0
    correct = 0.0
    acc = 0.0
    dyrep = DyRep( node.interaction_model,
                    node.optimizer,
                    recal_bn_fn=lambda m: recal_bn(node.interaction_model, train_loader,
                    node.args.dyrep_recal_bn_iters, m),
                    filter_bias_and_bn=not node.args.opt_no_filter)
    description = "Node{:d}: loss={:.4f} acc={:.2f}%"
    with tqdm(train_loader) as epochs:
        for batch_idx, (data, target) in enumerate(epochs):
            epochs.set_description(description.format(node.num, avg_loss, acc))
            data = data.float().to(node.device)
            target = target.long().to(node.device).squeeze()
            for p in node.interaction_model.parameters():
                p.grad = None
            output = node.interaction_model(data)

            loss = loss_fn(output, target)
            loss.backward()
            dyrep.record_metrics()
            node.optimizer.step()
            loss_m.update(loss.item(), n=data.size(0))
            total_loss += loss
            avg_loss = total_loss / (batch_idx + 1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum()
            acc = correct / len(train_loader.dataset) * 100
    

    if epoch ==0:
        device_capacity = node.capacity 
        dyrep.adjust_model(device_capacity)


def validate(args, epoch, model, loader, loss_fn, log_suffix=''):
    loss_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    batch_time_m = AverageMeter()
    start_time = time.time()

    model.eval()
    for batch_idx, (input, target) in enumerate(loader):
        input = input.float().to(args.device)
        target = target.long().to(args.device)
        with torch.no_grad():
            output = model(input)
            loss = loss_fn(output, target)

        top1, top5 = accuracy(output, target, topk=(1, 5))
        loss_m.update(loss.item(), n=input.size(0))
        top1_m.update(top1 * 100, n=input.size(0))
        top5_m.update(top5 * 100, n=input.size(0))

        batch_time = time.time() - start_time
        batch_time_m.update(batch_time)

        start_time = time.time()

    return {'test_loss': loss_m.avg, 'top1': top1_m.avg, 'top5': top5_m.avg}






class Trainer(object):

    def __init__(self, args):
        if args.algorithm == 'fed_avg':
            self.train = train_avg
        elif args.algorithm == 'dyfl':
            self.train = train_dyfl

    def __call__(self, node, epoch, round):
        self.train(node, epoch, round)







