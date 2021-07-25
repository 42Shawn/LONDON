#use WRN28-4 as teacher to train WRN-16-2, baseline is 75.92%
#LONDON 23.78
from __future__ import print_function

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

import distiller_london as distiller_mine
import load_settings

from tqdm import tqdm

parser = argparse.ArgumentParser(description='CIFAR-100 training')
parser.add_argument('--data_path', type=str, default='/home/user/shang/data/')
parser.add_argument('--paper_setting', default='c', type=str)
parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
parser.add_argument('--batch_size', default=64*2, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--alpha', default=0.25, type=float, help='alpha in loss function')
parser.add_argument('--is_distill_loss_applied', default=True, type=bool, help='adjuct the loss function')
parser.add_argument('--is_spectral_loss_applied', default=True, type=bool, help='adjust the loss function')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--gpu_id', default='1', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

gpu_num = 0
use_cuda = torch.cuda.is_available()
transform_train = transforms.Compose([
    transforms.Pad(4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                         np.array([63.0, 62.1, 66.7]) / 255.0)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                         np.array([63.0, 62.1, 66.7]) / 255.0),
])

trainset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
testset = torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

# Model
t_net, s_net, args = load_settings.load_paper_settings(args)

if torch.cuda.device_count() > 1:
    t_net = nn.DataParallel(t_net).to(device)
    s_net = nn.DataParallel(s_net).to(device)
else:
    t_net = t_net.to(device)
    s_net = s_net.to(device)

# t_net_mine = torchvision.models.resnet50(pretrained=True)

# Module for distillation
d_net = distiller_mine.Distiller(t_net, s_net)

print('the number of teacher model parameters: {}'.format(sum([p.data.nelement() for p in t_net.parameters()])))
print('the number of student model parameters: {}'.format(sum([p.data.nelement() for p in s_net.parameters()])))

if use_cuda:
    torch.cuda.set_device(0)
    d_net.cuda()
    s_net.cuda()
    t_net.cuda()
    # t_net_mine.cuda()
    cudnn.benchmark = True

# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model).to(device)
# else:
#     model = model.to(device)

criterion_CE = nn.CrossEntropyLoss()

# Training
def train_with_distill(d_net, epoch, args):
    epoch_start_time = time.time()
    print('\nDistillation epoch: %d' % epoch)

    d_net.train()
    d_net.s_net.train()
    d_net.t_net.train()

    train_loss = 0
    correct = 0
    total = 0

    global optimizer
    with tqdm(total=len(trainloader)) as t:
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()

            batch_size = inputs.shape[0]
            outputs, loss_distill, loss_spectral = d_net(inputs)
            loss_CE = criterion_CE(outputs, targets)
            # loss_classic_kd = criterion_CE()

            # loss = loss_CE + loss_distill.sum() / batch_size / 1000 + loss_spectral.sum() / batch_size / 1000/ 2 + loss_largest_eigenvalue.sum()/200
            if args.is_distill_loss_applied and args.is_spectral_loss_applied:
                loss = loss_CE + loss_distill.sum() / batch_size / 1000  + args.alpha*loss_spectral.sum() / 200
                # print('loss mode:',1)
            elif args.is_distill_loss_applied and not args.is_spectral_loss_applied:
                loss = loss_CE + loss_distill.sum() / batch_size / 1000
                # print('loss mode:',2)
            elif args.is_spectral_loss_applied and not args.is_distill_loss_applied:
                loss = loss_CE + args.alpha * loss_spectral.sum() / 200
                print('loss mode:',3)

            # print(loss_CE/loss)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().float().item()

            b_idx = batch_idx

            t.update()

        print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
        print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))

    return train_loss / (b_idx + 1)

def test(net):
    epoch_start_time = time.time()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = criterion_CE(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    print('Test \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (b_idx + 1), 100. * correct / total, correct, total))
    return test_loss / (b_idx + 1), correct / total


print('Performance of teacher network')
test(t_net)

print('Performance of pretrained-student network')
test(s_net)

for epoch in range(args.epochs):
    if epoch == 0:
        optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
                              lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif epoch == (args.epochs * 1// 2):
        optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
                              lr=args.lr / 10, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif epoch == (args.epochs * 3// 4):
        optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
                              lr=args.lr / 100, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif epoch == (args.epochs * 7// 8):
        optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
                              lr=args.lr / 500, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    train_loss = train_with_distill(d_net, epoch, args)
    test_loss, accuracy = test(s_net)