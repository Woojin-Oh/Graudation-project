import torch
import torch.utils.data as data
from PIL import Image
from spatial_transforms import *
import os
import math
import functools
import json
import copy
from numpy.random import randint
import numpy as np
import random

from utils import load_value_file
import pdb

import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from MiDaS import run, utils
from MiDaS.midas.model_loader import default_models, load_model

#MiDaS 적용 버전

def pil_loader(path, modality):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        #print(path)
        with Image.open(f) as img:
            if modality == 'RGB':
                return img.convert('RGB')
            elif modality == 'Depth':
                return img.convert('L') # 8-bit pixels, black and white check from https://pillow.readthedocs.io/en/3.0.x/handbook/concepts.html


def accimage_loader(path, modality):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, modality, sample_duration, device, midas_model, midas_model_type, transform, net_w, net_h, image_loader):

        
    video = []
    if modality == 'RGB':
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(i))
            if os.path.exists(image_path):
                
                video.append(image_loader(image_path, modality))
            else:
                print(image_path, "------- Does not exist")
                return video
    elif modality == 'Depth':

        for i in frame_indices:
            image_path = os.path.join(video_dir_path.replace('color','depth'), '{:05d}.jpg'.format(i) )
            if os.path.exists(image_path):
                video.append(image_loader(image_path, modality))
            else:
                print(image_path, "------- Does not exist")
                return video
    elif modality == 'RGB-D':
        for i in frame_indices: # index 35 is used to change img to flow
            image_path = os.path.join(video_dir_path, '{:05d}.jpg'.format(i))

            
            #image_path_depth = os.path.join(video_dir_path.replace('color','depth'), '{:05d}.jpg'.format(i) )

            
            image = image_loader(image_path, 'RGB') #[0~255], uint8
            

            # input
            #print('image type: ', type(image))
            original_image_rgb = utils.read_image(image_path)
            
            midas_image = transform({"image": original_image_rgb})["image"]

            #print('original_image_rgb: ', original_image_rgb.shape) # 320, 240, 3

            # compute
            with torch.no_grad():
                prediction = run.process(device, midas_model, midas_model_type, midas_image, (net_w, net_h), original_image_rgb.shape[1::-1], #[1] 인덱스부터 역순 -> 240, 320
                                     optimize=False, use_camera = False)
                #print('midas depth output: ', prediction.shape) (240, 320)
                prediction = prediction.T #float 32
                depth_pil = Image.fromarray(prediction.astype(np.uint8))
                
        

            if os.path.exists(image_path):
                video.append(image)
                video.append(depth_pil)
            else:
                print(image_path, "------- Does not exist")
                return video
        #print('frame indices: ', frame_indices)
        #print('video len: ', len(video))
        #print('video[1].shape: ', video[1].shape)
        #print('video[1] type: ', type(video[1]))
        #print('video[0] shape: ', video[0].shape)
    
    return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            #video_names.append('{}/{}'.format(label, key))
            video_names.append(key.split('^')[0])
            annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    print("[INFO]: NV Dataset - " + subset + " is loading...")
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        
        if not os.path.exists(video_path):
            continue

        

        begin_t = int(annotations[i]['start_frame'])
        end_t = int(annotations[i]['end_frame'])
        n_frames = end_t - begin_t + 1
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            #'video_id': video_names[i].split('/')[1]
            'video_id': i
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(begin_t, end_t + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class


class NV(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 midas_input_path,
                 midas_output_path,
                 midas_weights,
                 midas_model_type,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 modality='RGB',
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.modality = modality
        self.sample_duration = sample_duration
        self.loader = get_loader()

        self.midas_input_path = midas_input_path
        self.midas_output_path = midas_output_path
        self.midas_weights = midas_weights
        self.midas_model_type = midas_model_type

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas_model, self.transform, self.net_w, self.net_h= load_model(self.device, self.midas_weights, self.midas_model_type,
                                                                     optimize=False, height= None, square= True)
        print(f"    Input resized to {self.net_w}x{self.net_h} before entering the encoder")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']

        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices, self.modality, self.sample_duration, self.device, self.midas_model, self.midas_model_type, self.transform, self.net_w, self.net_h)
        oversample_clip =[]
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
    

        
        im_dim = clip[0].size()[-2:]
        clip = torch.cat(clip, 0).view((self.sample_duration, -1) + im_dim).permute(1, 0, 2, 3)

        print('clip shape: ', clip.shape)
        
     
        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)


