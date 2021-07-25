import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage() #实例化
    loss_avg.update(2) #调用实例
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def value(self):
        return self.total/float(self.steps)
# loss_avg = RunningAverage() #实例化
# loss_avg.update(2) #调用实例
# loss_avg.update(4)
# print(loss_avg.value())

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)

def load_json_to_dict(json_path):
    """Loads json file to dict 

    Args:
        json_path: (string) path to json file
    """
    with open(json_path, 'r') as f:
        params = json.load(f)
    return params

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim= 1, largest= True, sorted= True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class kd_loss_fn(nn.Module):
    def __init__(self, num_classes, args):
        super(kd_loss_fn, self).__init__()
        self.num_classes = num_classes
        self.alpha = args.alpha
        self.T = args.temperature
        
    def forward(self, output_batch, labels_batch, teacher_outputs):
        # output_batch  -> B X num_classes
        # labels_batch  -> B, LongTensor
        # teacher_outputs -> B X num_classes
        
        # torch.save(output_batch, './output_batch')
        # torch.save(labels_batch,'./labels_batch')
        # torch.save(teacher_outputs,'./teacher_outputs')
    
        # zero-mean, and small value
        # teacher_outputs = (teacher_outputs - torch.mean(teacher_outputs, dim=1).view(-1,1))/100.0
        # output_batch = (output_batch - torch.mean(output_batch, dim=1).view(-1,1))/100.0
    
        teacher_outputs=F.softmax(teacher_outputs/self.T,dim=1)
        output_batch=F.log_softmax(output_batch/self.T,dim=1)    
    
        #CE_teacher = -torch.sum(torch.sum(torch.mul(teacher_outputs,output_batch)))/teacher_outputs.size(0)
        #CE_teacher.requires_grad_(True)
        KL_teacher = nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs) * self.T
        CE_true = nn.CrossEntropyLoss()(output_batch, labels_batch) 
        loss = KL_teacher * self.alpha + CE_true * (1 - self.alpha)    
        return loss

       

class Att_Loss(nn.Module):
    def __init__(self, temperature = 1, loss = 'CE'):
        super(Att_Loss, self).__init__()        
        self.T = temperature
        self.loss = loss
    def forward(self, output_batch, labels_batch, attention):
        # output_batch  -> B X num_classes X num_student
        # attention     -> B X num_student X num_student
        # teacher_outputs -> B X num_classes
        
        batch_size, num_classes, num_student = output_batch.size()
        labels_batch = labels_batch.view(-1,1).repeat(1, num_student)  # B X num_student
        loss_true = nn.CrossEntropyLoss()(output_batch, labels_batch) * num_student
        # teacher_outputs = teacher_outputs.repeat(args.num_student, 1, 1).view(-1, num_classes, args.num_student) # B X num_classes X num_student    
        
        attention_label = torch.bmm(output_batch, attention.permute(0,2,1))     # B X num_classes X num_student
        
        if self.loss == 'CE':
            output_batch = F.log_softmax(output_batch/self.T, dim=1)
            attention_outputs = F.softmax(attention_label/self.T, dim=1)            # B X num_classes X num_student
            loss_att = -torch.sum(torch.mul(output_batch, attention_outputs))/batch_size
        elif self.loss == 'MSE':
            # calculate the average distance between attention and identity
            output_batch = F.softmax(output_batch, dim=1)    
            attention_outputs = F.softmax(attention_label, dim=1)            # B X num_classes X num_student
            loss_att = torch.sum((output_batch - attention_outputs) ** 2) / batch_size 
        # calculate the log angle 
        identity = torch.eye(num_student).reshape(1, num_student, num_student).repeat(batch_size, 1, 1).cuda()
        # calculate the average distance between attention and identity
        scale = torch.Tensor([batch_size * num_student]).sqrt().cuda()
        dist_att = torch.norm(attention - identity, p='fro')/scale
        # dist_p = torch.norm(output_batch, p='fro')
        # angle = torch.log(loss_att) - torch.log(dist) - torch.log(dist_p)
        # angle = loss_att/(dist * dist_p)
        return loss_true, loss_att, dist_att
        
class KL_Loss(nn.Module):
    def __init__(self, temperature = 1):
        super(KL_Loss, self).__init__()
        self.T = temperature
    def forward(self, output_batch, teacher_outputs):
        # output_batch  -> B X num_classes            
        # teacher_outputs -> B X num_classes
        
        # loss_2 = -torch.sum(torch.sum(torch.mul(F.log_softmax(teacher_outputs,dim=1), F.softmax(teacher_outputs,dim=1)+10**(-7))))/teacher_outputs.size(0)
        # print('loss H:',loss_2)
        
        output_batch = F.log_softmax(output_batch/self.T, dim = 1)    
        teacher_outputs = F.softmax(teacher_outputs/self.T, dim = 1) + 10**(-7)
    
        loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)
        #print(1111111111111111110000000000000000000001111111111111111111111)
        
        # Same result KL-loss implementation
        # loss = T * T * torch.sum(torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch)))/teacher_outputs.size(0)
        return loss

        
class CE_Loss(nn.Module):
    #继承自nn.Module的子类。
    def __init__(self, temperature = 1):
        super(CE_Loss, self).__init__() #super函数，实现父类与子类的关联。如父类有参数，则需要传递参数
        self.T = temperature
    def forward(self, output_batch, teacher_outputs):
    
        # output_batch      -> B X num_classes 
        # teacher_outputs   -> B X num_classes
        
        output_batch = F.log_softmax(output_batch/self.T,dim=1)    
        teacher_outputs = F.softmax(teacher_outputs/self.T,dim=1)
        
        # Same result CE-loss implementation torch.sum -> sum of all element
        loss = -self.T*self.T*torch.sum(torch.mul(output_batch, teacher_outputs))/teacher_outputs.size(0)
        
        return loss

class MSE_Loss(nn.Module):
    def __init__(self):
        super(MSE_Loss, self).__init__()
        
    def forward(self, output_batch, teacher_outputs):
    
        # output_batch      -> B X num_classes 
        # teacher_outputs   -> B X num_classes
        
        batch_size = output_batch.size(0)
        output_batch = F.softmax(output_batch, dim = 1)
        teacher_outputs = F.softmax(teacher_outputs, dim = 1)
        # Same result MSE-loss implementation torch.sum -> sum of all element
        loss = torch.sum((output_batch - teacher_outputs) ** 2) / batch_size 
        
        return loss

class E_Loss(nn.Module):
    #继承自nn.Module的类
    def __init__(self, temperature = 1):
        super(E_Loss, self).__init__() #
        self.T = temperature
    def forward(self, output_batch, teacher_outputs):
    
        # output_batch      -> B X num_classes 
        # teacher_outputs   -> B X num_classes
        
        output_batch = F.log_softmax(output_batch/self.T,dim=1)    
        self_outputs = F.softmax(output_batch/self.T,dim=1)
        
        # Same result CE-loss implementation torch.sum -> sum of all element
        loss = -self.T*self.T*torch.sum(torch.mul(output_batch, self_outputs))/output_batch.size(0)
        
        return loss

def lookup(model_name):
    if model_name == "resnet8" or model_name == "resnet14" or model_name == "resnet20" or model_name == "resnet32":
        input_channel = 64
    elif model_name == "densenetd40k12":
        input_channel = 132
    elif model_name == "densenetd100k12":
        input_channel = 342
    elif model_name == "densenetd100k40":
        input_channel = 1126
    elif model_name == "resnet110":
        input_channel = 256
    elif model_name == "vgg16" or model_name == "resnet34":
        input_channel = 512
    elif model_name == "wide_resnet20_8" or model_name == "wide_resnet28_10":
        input_channel = 256
    # imagenet
    elif model_name == "shufflenet_v2_x1_0": 
        input_channel = 1024
    return input_channel

class prune_WILTON(nn.Module):
    # 继承自nn.Module的类
    def __init__(self, temperature=1):
        super(E_Loss, self).__init__()  #
        self.T = temperature
    def prune_wilton(self,module,pruning_method,pruning_rate=0.3):
        model_pruned =  nn.Module.structed_prune(module,pruning_rate)
        return model_pruned

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f