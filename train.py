import torch
from torch.autograd import Variable
import time
import os
import sys

from utils import *


def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        '''
        import torchvision.transforms as transforms
        from PIL import Image
        # inputs: torch.Size([40, 3, 16, 112, 112])
        transform = transforms.ToPILImage()
        #print('targets shape: ', targets.shape) #[40]

        # 디렉토리 생성 (없으면)
        import os
        if not os.path.exists('frames_HOLO'):
            os.makedirs('frames_HOLO')

        
        if i <4:
            for k in range(inputs.size(0)): # batch size loop
                for j in range(inputs.size(2)): # frame loop
                    img = inputs[k,:,j,:,:] # get image tensor 
                    img = transform(img) # convert tensor to PIL image
                    trg = targets[k]
                    img.save(f'frames_HOLO/frame_{i}_{k}_{j}_{trg}.jpg') # save image 
        '''

        if not opt.no_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()
        #inputs = Variable(inputs)
        #targets = Variable(targets)
        #print('mmodel: ', model)
        #for name, param in model.named_parameters():
        #    print(name, param.shape)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        #print('output: ', outputs)
        #print('targets: ', targets)
        #print('loss: ',loss)

        #for name, param in model.named_parameters():
        #    print('name, requires_grad: ',name, param.requires_grad)

        losses.update(loss.data, inputs.size(0))
        #prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
        prec1 = calculate_accuracy(outputs.data, targets.data)[0]
        top1.update(prec1, inputs.size(0))
        #top5.update(prec5, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val.item(),
            'prec1': top1.val.item(),
            #'prec5': top5.val.item(),
            'lr': optimizer.param_groups[0]['lr']
        })
        if i % 10 ==0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'.format( ##'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'
                      epoch,
                      i,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      #top5=top5,
                      lr=optimizer.param_groups[0]['lr']))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'prec1': top1.avg.item(),
        #'prec5': top5.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })

    #if epoch % opt.checkpoint == 0:
    #    save_file_path = os.path.join(opt.result_path,
    #                                  'save_{}.pth'.format(epoch))
    #    states = {
    #        'epoch': epoch + 1,
    #        'arch': opt.arch,
    #        'state_dict': model.state_dict(),
    #        'optimizer': optimizer.state_dict(),
    #    }
    #    torch.save(states, save_file_path)
