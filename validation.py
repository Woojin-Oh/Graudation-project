import torch
from torch.autograd import Variable
import time
import sys

from utils import *

def val_epoch(epoch, data_loader, model, criterion, opt, logger):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    #top5 = AverageMeter()
    recall = AverageMeter()
    precision = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda()
        with torch.no_grad():
            inputs = Variable(inputs)
            targets = Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        #prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,2))
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,2))
        recall_1 = calculate_recall(outputs.data, targets.data)
        precision_1 = calculate_precision(outputs.data, targets.data)
        top1.update(prec1, inputs.size(0))
        #top5.update(prec5, inputs.size(0))
        recall.update(recall_1, inputs.size(0))
        precision.update(precision_1, inputs.size(0))

        losses.update(loss.data, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % 10 ==0:
          print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
              'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
              'Recall {recall.val:.5f} ({recall.avg:.5f})\t'
              'Precision {precision.val:.5f} ({precision.avg:.5f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  top1=top1,
                  #top5=top5))
                  recall=recall,
                  precision = precision))

    logger.log({'epoch': epoch,
                'loss': losses.avg.item(),
                'prec1': top1.avg.item(),
                #'prec5': top5.avg.item()})
                'recall': recall.avg.item(),
                'precision': precision.avg.item()})

    return losses.avg.item(), top1.avg.item(), recall.avg.item(), precision.avg.item()


def test_epoch(epoch, data_loader, model, opt, logger):
    print('test at epoch {}'.format(epoch))

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    #top5 = AverageMeter()
    recall = AverageMeter()
    precision = AverageMeter()

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
        if not os.path.exists('frames'):
            os.makedirs('frames')

        
        if i <4:
            for k in range(inputs.size(0)): # batch size loop
                for j in range(inputs.size(2)): # frame loop
                    img = inputs[k,:,j,:,:] # get image tensor
                    trg = targets[k]
                    #print('img shape: ', img.shape)
                    
                    img = transform(img) # convert tensor to PIL image
                    img = img.convert("RGB")
                    #print('img type: ',type(img))
                    #img.save(f'frames/frame_{i}_{k}_{j}_{trg}.jpg') # save image 
                    
                    # Get RGB and depth images
                    rgb_img = img[:3,:,:]
                    depth_img = img[3,:,:]

                    # Convert tensors to PIL images
                    rgb_img = transform(rgb_img)
                    depth_img = transform(depth_img)

                    # Convert depth image to grayscale mode
                    depth_img = depth_img.convert("L")

                    # Save images
                    rgb_img.save(f'frames/rgb_frame_{i}_{k}_{j}_{trg}.jpg')
                    depth_img.save(f'frames/depth_frame_{i}_{k}_{j}_{trg}.jpg')

        '''
        

        if not opt.no_cuda:
            targets = targets.cuda()
        with torch.no_grad():
            outputs = model(inputs)
        #prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
        #print('outputs type: ', type(outputs))
        #print('outputs.data shape: ', outputs.data.shape) #[40,2]
        #print('targets.data shape: ', targets.data.shape) #[40]
        prec1= calculate_accuracy(outputs.data, targets.data)[0]
        recall_1 = calculate_recall(outputs.data, targets.data)
        precision_1 = calculate_precision(outputs.data, targets.data)
        top1.update(prec1, inputs.size(0))
        #top5.update(prec5, inputs.size(0))
        recall.update(recall_1, inputs.size(0))
        precision.update(precision_1, inputs.size(0))

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if i % 10 ==0:
          print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
              'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
              'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
              'Recall {recall.val:.5f} ({recall.avg:.5f})\t'
              'Precision {precision.val:.5f} ({precision.avg:.5f})\t'.format( #'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  top1=top1,
                  recall=recall,
                  precision=precision))
                  
    logger.log({'total accuracy': top1.avg, 'total recall': recall.avg, 'total precision': precision.avg})

    print('total accuracy: ', top1.avg)
    print('total recall: ', recall.avg)
    print('total preicison: ', precision.avg)
    return top1.avg.item()