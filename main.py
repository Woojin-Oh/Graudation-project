import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from mean import get_mean, get_std
from spatial_transforms import *
from temporal_transforms import *
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose
from dataset import get_training_set, get_validation_set, get_test_set
from utils import *
from train import train_epoch
from validation import val_epoch, test_epoch
import test

import torch.multiprocessing as mp

from models.resnetl import get_fine_tuning_parameters

if __name__ == '__main__':

    mp.set_start_method('spawn') 

    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}'.format(opt.model)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.width_mult) + 'x',
                               opt.modality, str(opt.sample_duration)])
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    #코드 저장
    code_record = Recorder(opt, opt.result_path)

    model, parameters = generate_model(opt)
    model = model.cuda()
    print('model: ',model)
    print('model device: ', next(model.parameters()).device)
    print('loaded model conv1 weight: ', model.state_dict()['module.conv1.weight'].shape) # torch.Size([16, 5, 7, 7, 7])

    for i in model.modules():
        print(i)
        break

    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f'NaN found in model parameters: {name}')
            print('param: ', param.shape)

    # Egogesture, with "no-gesture" training, weighted loss
    # class_weights = torch.cat((0.012*torch.ones([1, 83]), 0.00015*torch.ones([1, 1])), 1)
    #criterion = nn.CrossEntropyLoss()

    # # nvgesture, with "no-gesture" training, weighted loss
    # class_weights = torch.cat((0.04*torch.ones([1, 25]), 0.0008*torch.ones([1, 1])), 1)
    # criterion = nn.CrossEntropyLoss(weight=class_weights, size_average=False)

    #criterion = nn.CrossEntropyLoss()
    #class_weights = torch.tensor([1.0, 3.0]) #기존 1,3
    #class_weights = torch.tensor([3.0, 1.0])
    #class_weights = torch.tensor([1.0, 3.0]) #train class 비율: 3:1(true가 더 많음)
    #class_weights = torch.tensor([0.1, 1.0]) #train class 비율: 20:1(true가 더 많음)
    #criterion = nn.CrossEntropyLoss(weight=class_weights) #자체적으로 softmax 적용

    #focal loss
    criterion = FocalLoss(alpha=0.25, gamma=2)
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            #RandomHorizontalFlip(),
            #RandomRotate(),
            #RandomResize(),
            crop_method,
            #MultiplyValues(),
            #Dropout(),
            #SaltImage(),
            #Gaussian_blur(),
            #SpatialElasticDisplacement(),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
        target_transform = ClassLabel()
        training_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        
        train_logger = Logger(
            os.path.join(opt.result_path, opt.store_name + '_train.log'),
            #['epoch', 'loss', 'prec1', 'prec5', 'lr'])
            ['epoch', 'loss', 'prec1', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            #['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])
            ['epoch', 'batch', 'iter', 'loss', 'prec1', 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        optimizer = optim.SGD(
            model.parameters(),#parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)
    if not opt.no_val:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        #temporal_transform = LoopPadding(opt.sample_duration)
        temporal_transform = TemporalCenterCrop(opt.sample_duration, opt.downsample)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=8,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            #os.path.join(opt.result_path, opt.store_name + '_val.log'), ['epoch', 'loss', 'prec1', 'prec5'])
            os.path.join(opt.result_path, opt.store_name + '_val.log'), ['epoch', 'loss', 'prec1', 'recall'])

    best_prec1 = 0
    best_recall = 0
    best_precision = 0
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']
        best_prec1 = checkpoint['best_prec1']
        best_recall = checkpoint['best_recall']
        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])


    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
    # for i in range(opt.begin_epoch, opt.begin_epoch + 10):
        if not opt.no_train:
            adjust_learning_rate(optimizer, i, opt)
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
                'best_recall': best_recall,
                'best_precision': best_precision
                }
            save_checkpoint(state, False, opt)
            
        if not opt.no_val:
            validation_loss, prec1, recall, precision = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)
            is_best_prec1 = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            is_best_recall = recall > best_recall
            best_recall = max(recall, best_recall)
            is_best_precision = precision > best_precision
            best_precision = max(precision, best_precision)
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1,
                'best_recall': best_recall,
                'best_precision': best_precision
                }
            save_checkpoint_prec(state, is_best_prec1, opt)
            save_checkpoint_recall(state, is_best_recall, opt)
            save_checkpoint_precision(state, is_best_precision, opt)
            '''
            validation_loss, prec1 = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
                }
            save_checkpoint(state, is_best, opt)
            '''


    if opt.test:
        spatial_transform = Compose([
            Scale(int(opt.sample_size / opt.scale_in_test)),
            CornerCrop(opt.sample_size, opt.crop_position_in_test),
            ToTensor(opt.norm_value), norm_method
            #ToTensor(opt.norm_value)
        ])
        # temporal_transform = LoopPadding(opt.sample_duration, opt.downsample)
        temporal_transform = TemporalRandomCrop(opt.sample_duration, opt.downsample)
        #target_transform = VideoID()
        target_transform = ClassLabel()

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=8,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        #test.test(test_loader, model, opt, test_data.class_names)

        test_logger = Logger(
            #os.path.join(opt.result_path, opt.store_name + '_val.log'), ['epoch', 'loss', 'prec1', 'prec5'])
            os.path.join(opt.result_path, opt.store_name + '_test.log'), ['total accuracy', 'total recall', 'total precision'])
        prec1 = test_epoch(0, test_loader, model, opt, test_logger)
        print('total accuracy: ', prec1)




