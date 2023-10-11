import csv
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import shutil
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Recorder(object):
    def __init__(self, settings, save_path):
        print('>> Init a recoder at ', save_path)
        self.save_path = save_path
        self.settings = settings
        self.code_path = os.path.join(self.save_path, 'code')
        #self.log_path = os.path.join(self.save_path, 'log')
        #self.code_file_extension = ['.py', '.yml', '.yaml', '.sh']
        self.code_file_extension = ['.py', '.json']
        self.ignore_file_extension = ['.pyc']
        self.excluded_strings = ['Video_data', 'MiDaS', 'frame', 'pretrained_weight', 'results']
        #self.checkpoint_path = os.path.join(self.save_path, 'checkpoint')
        

        # make directories
        if not os.path.isdir(self.code_path):
            os.makedirs(self.code_path)
        #if not os.path.isdir(self.log_path):
        #    os.makedirs(self.log_path)
        #if not os.path.isdir(self.checkpoint_path):
        #    os.makedirs(self.checkpoint_path)

        # init logger
        #self.logger = self._initLogger()
        # save code and settings
        self._saveConfig()

    def _checkFileExtension(self, file_name):
        for k in self.code_file_extension:
            if k in file_name:
                for kk in self.ignore_file_extension:
                    if kk in file_name:
                        return False
                return True
        return False

    def _copyFiles(self, root_path, target_path):
        file_list = os.listdir(root_path)
        for file_name in file_list:
            file_path = os.path.join(root_path, file_name)
            #if os.path.isdir(file_path) and 'log_' not in file_path:
            if os.path.isdir(file_path) and all(s not in file_path for s in self.excluded_strings):
                dst_path = os.path.join(target_path, file_path)
                self._copyFiles(file_path, dst_path)
            else:
                if self._checkFileExtension(file_name):
                    if not os.path.isdir(target_path):
                        os.makedirs(target_path)
                    dst_file = os.path.join(target_path, file_name)
                    shutil.copyfile(file_path, dst_file)
                
    def _saveConfig(self):
        # copy code files
        self._copyFiles(root_path='./', target_path=self.code_path)

        # write settings to file
        #with open(os.path.join(self.log_path, 'settings.log'), 'w') as f:
        #    for k, v in self.settings.__dict__.items():
        #        f.write('{}: {}\n'.format(k, v))


class Logger(object):

    def __init__(self, path, header):

        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()



class Queue:
    # Constructor creates a list
    def __init__(self, max_size, n_classes):
        self.queue = list(np.zeros((max_size, n_classes), dtype=float).tolist())
        self.max_size = max_size
        self.median = None
        self.ma = None
        self.ewma = None

    # Adding elements to queue
    def enqueue(self, data):
        self.queue.insert(0, data)
        self.median = self._median()
        self.ma = self._ma()
        self.ewma = self._ewma()
        return True

    # Removing the last element from the queue
    def dequeue(self):
        if len(self.queue) > 0:
            return self.queue.pop()
        return ("Queue Empty!")

    # Getting the size of the queue
    def size(self):
        return len(self.queue)

    # printing the elements of the queue
    def printQueue(self):
        return self.queue

    # Average
    def _ma(self):
        return np.array(self.queue[:self.max_size]).mean(axis=0)

    # Median
    def _median(self):
        return np.median(np.array(self.queue[:self.max_size]), axis=0)

    # Exponential average
    def _ewma(self):
        weights = np.exp(np.linspace(-1., 0., self.max_size))
        weights /= weights.sum()
        average = weights.reshape(1, self.max_size).dot(np.array(self.queue[:self.max_size]))
        return average.reshape(average.shape[1], )


def LevenshteinDistance(a, b):
    # This is a straightforward implementation of a well-known algorithm, and thus
    # probably shouldn't be covered by copyright to begin with. But in case it is,
    # the author (Magnus Lie Hetland) has, to the extent possible under law,
    # dedicated all copyright and related and neighboring rights to this software
    # to the public domain worldwide, by distributing it under the CC0 license,
    # version 1.0. This software is distributed without any warranty. For more
    # information, see <http://creativecommons.org/publicdomain/zero/1.0>
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)
    if current[n]<0:
        return 0
    else:
        return current[n]


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        #correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:k].reshape(-1).float().sum(0) #수정
        res.append(correct_k.mul_(100.0 / batch_size)) #맞춘 개수 / 전체 개수 * 100
    return res




def calculate_precision(outputs, targets):

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()

    targets_np = targets.view(-1).cpu().numpy()
    pred_np = pred.view(-1).cpu().numpy()

    return  precision_score(targets_np, pred_np, average = 'macro', zero_division=0)


def calculate_recall(outputs, targets):

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    #print('target type: ', type(targets))
    #targets = targets.cpu().numpy()
    #print('device: ', targets.device)
    #return  recall_score(targets.view(-1), pred.view(-1), average = 'macro')

    # Move tensors to CPU and convert them to NumPy arrays
    targets_np = targets.view(-1).cpu().numpy()
    pred_np = pred.view(-1).cpu().numpy()

    return recall_score(targets_np, pred_np, average='macro', zero_division=0)


def save_checkpoint(state, is_best, opt):
    torch.save(state, '%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name),'%s/%s_best.pth' % (opt.result_path, opt.store_name))

def save_checkpoint_prec(state, is_best, opt):
    torch.save(state, '%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name),'%s/%s_best_prec.pth' % (opt.result_path, opt.store_name))

def save_checkpoint_recall(state, is_best, opt):
    torch.save(state, '%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name),'%s/%s_best_recall.pth' % (opt.result_path, opt.store_name))

def save_checkpoint_precision(state, is_best, opt):
    torch.save(state, '%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth' % (opt.result_path, opt.store_name),'%s/%s_best_precision.pth' % (opt.result_path, opt.store_name))


def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_new = opt.learning_rate * (0.1 ** (sum(epoch >= np.array(opt.lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new
        #param_group['lr'] = opt.learning_rate


class FocalLoss(nn.Module):
    def __init__(self, alpha=1., gamma=2.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        targets_one_hot = F.one_hot(targets, num_classes=2).float()
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets_one_hot, reduction='none')
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()

