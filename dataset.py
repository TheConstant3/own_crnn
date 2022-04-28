#!/usr/bin/python
# encoding: utf-8
import json
import os
import random

import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import albumentations.augmentations as A
import lmdb
import six
import sys
from PIL import Image
import numpy as np

from generate_plate import get_char_images, get_bkg_plate_images, get_random_data, generate_plate


class lprDataset(Dataset):
    def __init__(self, stage=0, n_samples=128000, n_channels=1, debug=False):
        self.nSamples = n_samples
        self.debug = debug
        if stage == 0:
            P = 0.1
        elif stage == 1:
            P = 0.2
        elif stage == 2:
            P = 0.3
        else:
            P = 0.4

        print(f'augmentation prob is {P}')

        augmentations = []
        augmentations.append(A.PiecewiseAffine(scale=(0.01, 0.02), p=P))
        augmentations.append(A.MotionBlur(blur_limit=15, p=P+0.2))
        augmentations.append(A.GlassBlur(max_delta=2, p=P))
        augmentations.append(A.RandomBrightness(limit=(-0.6, 0.2), p=1))
        augmentations.append(A.RandomSunFlare(flare_roi=(0, 0, 1, 1), num_flare_circles_lower=1, src_radius=100, p=P))
        augmentations.append(A.RandomFog(p=P))
        augmentations.append(A.RandomSnow(p=P))
        augmentations.append(A.JpegCompression(quality_lower=1, quality_upper=100, p=P+0.2))
        augmentations.append(A.ImageCompression(quality_lower=1, quality_upper=100, p=P-0.1))
        augmentations.append(A.GaussNoise(var_limit=(50, 100), p=P))
        augmentations.append(A.OpticalDistortion(distort_limit=0.1, shift_limit=0.5, p=P))
        augmentations.append(A.Perspective(scale=0.05, p=P))
        augmentations.append(A.Rotate(limit=5, p=P))
        augmentations.append(A.Affine(rotate=1, shear={"x": (-10, 10), "y": (-5, 5)}, p=P))

        self.available_transforms = augmentations
        self.target_transforms = self.available_transforms if stage > 0 else None

        self.alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZaou0123456789 _'
        self.char_image_dict = get_char_images()
        self.plate_bkgs = get_bkg_plate_images()
        self.n_channels = n_channels

    @staticmethod
    def preprocess_label(label: str):
        return label.replace(' ', '').replace('_', 's')

    def generate_sample(self):
        bkg_plate, number = get_random_data(self.alphabet, self.plate_bkgs)
        label = ''
        while label == '':
            generated_plate, label = generate_plate(number, self.char_image_dict, bkg_plate)

        return generated_plate, self.preprocess_label(label)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1

        img_orig, label = self.generate_sample()
        img = img_orig.copy()
        if self.target_transforms is not None:
            img = {'image': img}
            random.shuffle(self.target_transforms)
            for target_transform in self.target_transforms:
                img = target_transform(**img)
            img = img['image']

        if self.n_channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.debug:
            return img, label, img_orig
        return img, label


class imageDataset(Dataset):
    def __init__(self, root=None, transform=None, target_transform=None, debug=False):
        self.images_path = f'{root}/img'
        self.labels_path = f'{root}/ann'
        self.samples = []
        self.debug = debug

        for image_file in os.listdir(self.images_path):
            img_path = f'{self.images_path}/{image_file}'
            lbl_path = f'{self.labels_path}/{image_file.replace(".png", ".json")}'
            with open(lbl_path) as f:
                label = json.load(f)["description"]
            self.samples.append({
                'image': img_path,
                'label': label
            })

        self.nSamples = len(self.samples)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index < len(self), 'index range error'
        sample = self.samples[index]
        index += 1

        image_file = sample['image']
        img_orig = cv2.imread(image_file, 0)
        img = img_orig.copy()
        label = sample['label']

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.debug:
            return img, label, img_orig
        return img, label


class lmdbDataset(Dataset):
    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key))

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        # img = img.resize(self.size, self.interpolation)
        # print(img)
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)

        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1, debug=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
