'''
This is PyTorch 1.0 implementation for training baseline model on CIFAR-10/100 and ImageNet.
更改scheduler策略ReduceLROnPlateau
'''
import argparse
import logging
import os
import random
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms

import utils_for_training_cifar as utils

import load_settings
import models
# import models.data_loader as data_loader
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True

# Fix the random seed for reproducible experiments
# random.seed(97)
# np.random.seed(97)
# torch.manual_seed(97)
# if torch.cuda.is_available(): torch.cuda.manual_seed(97)
# torch.backends.cudnn.deterministic = True

# Set parameters
parser = argparse.ArgumentParser()

print(models.__dict__)
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# parser.add_argument('--model', metavar='ARCH', default='vgg16')
parser.add_argument('--paper_setting', default='test', type=str)
parser.add_argument('--dataset', default='CIFAR100', type=str, help='Input the name of dataset: default(CIFAR100)')
parser.add_argument('--data_path', type=str, default='/home/user/shang/data')
# parser.add_argument('--paper_setting', default='a', type=str)
parser.add_argument('--num_epochs', default = 300, type=int, help='Input the number of epoches: default(300)')
parser.add_argument('--batch_size', default=128 * 2, type=int, help='Input the batch size: default(128)')
parser.add_argument('--lr', default=0.1, type=float, help='Input the learning rate: default(0.1)')
# parser.add_argument('--schedule', type=int, nargs='+', default=[30, 80, 150, 200],
#                     help='Decrease learning rate at these epochs.')
parser.add_argument('--efficient', action='store_true',
                    help='Decide whether or not to use efficient implementation(only of densenet): default(False)')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Input the weight decay rate: default(5e-4)')
parser.add_argument('--dropout', default=0.005, type=float, help='Input the dropout rate: default(0.0)')#resnet因为有bn layer，所以不需要dropout
parser.add_argument('--resume', default='', type=str, help='Input the path of resume model: default('')')
parser.add_argument('--version', default='V0', type=str, help='Input the version of current model: default(V0)')
parser.add_argument('--num_workers', default=8, type=int, help='Input the number of works: default(8)')
parser.add_argument('--gpu_id', default='1', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(args)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(train_loader, model, optimizer, criterion, accuracy, args):
    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    loss_avg = utils.RunningAverage()
    accTop1_avg = utils.RunningAverage()
    accTop5_avg = utils.RunningAverage()
    end = time.time()

    # Use tqdm for progress bar
    with tqdm(total=len(train_loader)) as t:
        for _, (train_batch, labels_batch) in enumerate(train_loader):
            # _,占位符。之后不会用到，省下起变量名称。
            train_batch = train_batch.cuda(non_blocking=True)
            labels_batch = labels_batch.cuda(non_blocking=True)

            # compute model output and loss
            output_batch = model(train_batch)
            loss = criterion(output_batch, labels_batch)
            loss = loss/args.batch_size
            #训练技巧：要做梯度归一化,即算出来的梯度除以minibatch size
            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Update average loss and accuracy
            metrics = accuracy(output_batch, labels_batch, topk=(1, 5))
            accTop1_avg.update(metrics[0].item())
            accTop5_avg.update(metrics[1].item())
            loss_avg.update(loss.item())
            # optimizer.step()
            t.update()

    # compute mean of all metrics in summary
    train_metrics = {'train_loss': loss_avg.value(),
                     'train_accTop1': accTop1_avg.value(),
                     'train_accTop5': accTop5_avg.value(),
                     'time': time.time() - end}

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in train_metrics.items())
    logging.info("- Train metrics: " + metrics_string)
    return train_metrics


def evaluate(test_loader, model, criterion, accuracy, args):
    # set model to evaluation mode
    model.eval()
    loss_avg = utils.RunningAverage()
    accTop1_avg = utils.RunningAverage()
    accTop5_avg = utils.RunningAverage()
    end = time.time()

    with torch.no_grad():
        for test_batch, labels_batch in test_loader:
            test_batch = test_batch.cuda(non_blocking=True)
            labels_batch = labels_batch.cuda(non_blocking=True)

            # compute model output
            output_batch = model(test_batch)
            loss = criterion(output_batch, labels_batch)

            # Update average loss and accuracy
            metrics = accuracy(output_batch, labels_batch, topk=(1, 5))
            # only one element tensors can be converted to Python scalars
            accTop1_avg.update(metrics[0].item())
            accTop5_avg.update(metrics[1].item())
            loss_avg.update(loss.item())

    # compute mean of all metrics in summary
    test_metrics = {'test_loss': loss_avg.value(),
                    'test_accTop1': accTop1_avg.value(),
                    'test_accTop5': accTop5_avg.value(),
                    'time': time.time() - end}

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in test_metrics.items())
    logging.info("- Test  metrics: " + metrics_string)
    return test_metrics


def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, accuracy, model_dir, args):
    start_epoch = 0
    best_acc = 0.0
    # learning rate schedulers for different models:
    # scheduler = MultiStepLR(optimizer, milestones=args.schedule, gamma=0.33333)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, verbose=True, patience=10)

    # TensorboardX setup
    writer = SummaryWriter(log_dir=model_dir)
    # Save best accTop1
    choose_accTop1 = True

    # Save the parameters for export
    result_train_metrics = list(range(args.num_epochs))
    result_test_metrics = list(range(args.num_epochs))

    # If the training is interruptted
    if args.resume:
        # Load checkpoint.
        logging.info('Resuming from checkpoint..')
        resumePath = os.path.join(args.resume, 'last.pth')
        assert os.path.isfile(resumePath), 'Error: no checkpoint directory found!'

        checkpoint = torch.load(resumePath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim_dict'])
        # resume from the last epoch
        start_epoch = checkpoint['epoch']
        # scheduler.step(start_epoch - 1)
        if choose_accTop1:
            best_acc = checkpoint['test_accTop1']
        else:
            best_acc = checkpoint['test_accTop5']
        result_train_metrics = torch.load(os.path.join(args.resume, 'train_metrics'))
        result_test_metrics = torch.load(os.path.join(args.resume, 'test_metrics'))

    for epoch in range(start_epoch, args.num_epochs):



        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, args.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_metrics = train(train_loader, model, optimizer, criterion, accuracy, args)

        scheduler.step(train_metrics['train_loss'])

        writer.add_scalar('Train/Loss', train_metrics['train_loss'], epoch + 1)
        writer.add_scalar('Train/AccTop1', train_metrics['train_accTop1'], epoch + 1)
        writer.add_scalar('Train/AccTop5', train_metrics['train_accTop5'], epoch + 1)

        # Evaluate for one epoch on validation set
        test_metrics = evaluate(test_loader, model, criterion, accuracy, args)

        # Find the best accTop1 model.
        if choose_accTop1:
            test_acc = test_metrics['test_accTop1']
        else:
            test_acc = test_metrics['test_accTop5']

        writer.add_scalar('Test/Loss', test_metrics['test_loss'], epoch + 1)
        writer.add_scalar('Test/AccTop1', test_metrics['test_accTop1'], epoch + 1)
        writer.add_scalar('Test/AccTop5', test_metrics['test_accTop5'], epoch + 1)

        result_train_metrics[epoch] = train_metrics
        result_test_metrics[epoch] = test_metrics

        # Save latest train/test metrics
        torch.save(result_train_metrics, os.path.join(model_dir, 'train_metrics'))
        torch.save(result_test_metrics, os.path.join(model_dir, 'test_metrics'))

        # last_path = os.path.join(model_dir, 'last.pth')
        # Save latest model weights, optimizer and accuracy
        state = {'model': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'acc': test_metrics['test_accTop1'],
                    'test_accTop5': test_metrics['test_accTop5']}
        if not os.path.isdir('/home/user/shang/prg_kd/checkpoints/'):
            os.mkdir('/home/user/shang/prg_kd/checkpoints/')
        torch.save(state, '/home/user/shang/prg_kd/checkpoints/' + args.version+ '_WRN28-4_21.pt')

        # # save checkpoint
        # if epoch == 1:
        #     print('Saving..')
        #     state = {
        #         'net': model.state_dict(),
        #         'acc': test_metrics['test_accTop1'],
        #         'epoch': epoch,
        #     }
        #     if not os.path.isdir('checkpoint'):
        #         os.mkdir('checkpoint')
        #     torch.save(state, './checkpoint/pruned-' + args.model + '-ckpt.t7')

        # If best_eval, best_save_path
        is_best = test_acc >= best_acc
        if is_best:
            logging.info("- Found better accuracy")
            best_acc = test_acc
            # Save best metrics in a json file in the model directory
            test_metrics['epoch'] = epoch + 1
            utils.save_dict_to_json(test_metrics, os.path.join(model_dir, "test_best_metrics.json"))

            # Save model and optimizer
            # shutil.copyfile(last_path, os.path.join(model_dir, 'best.pth'))
    writer.close()


if __name__ == '__main__':

    begin_time = time.time()
    # Set the model directory
    model_dir = os.path.join('/home/user/shang/prg_kd/checkpoints', args.dataset, str(args.num_epochs), args.paper_setting + args.version)
    if not os.path.exists(model_dir):
        print("Directory does not exist! Making directory {}".format(model_dir))
        os.makedirs(model_dir)

    # Set the logger
    utils.set_logger(os.path.join(model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load data
    train_loader, test_loader = trainloader, testloader
    logging.info("- Done.")

    # Training from scratch
    model, s_net, args = load_settings.load_paper_settings(args)

    print('the number of teacher model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    num_params = (sum(p.numel() for p in model.parameters()) / 1000000.0)
    logging.info('Total params: %.2fM' % num_params)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    accuracy = utils.accuracy
    # 定义方法的时候加了self，那么在调用这个类中的方法时就必需要实例化一个对象，即：类（对象）.方法（参数）
    # 定义方法的时候没有加self，那么调用这个类的方法时就可以直接调用方法，即：类.方法（参数）
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(),lr=args.lr, betas=(0.9,0.99))
    #一般将参数设为0.5,0.9，或者0.99，分别表示最大速度2倍，10倍，100倍于SGD的算法。

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(args.num_epochs))
    train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, accuracy, model_dir, args)


    logging.info('Total time: {:.2f} minutes'.format((time.time() - begin_time) / 60.0))
    state['Total params'] = num_params
    # params_json_path = os.path.join(model_dir, "parameters.json")  # save parameters
    # utils.save_dict_to_json(state, params_json_path)