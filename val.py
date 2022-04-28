from __future__ import print_function
from __future__ import division

import argparse
import random
from collections import OrderedDict

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utils
import dataset

import models.crnn as crnn

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, help='path to data', required=True)
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=40, help='number of epochs to train for')
# TODO(meijieru): epoch -> iter
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--alphabet', type=str, default='ABCDEFGHIJKLMNOPQRSTUVWXYZaou0123456789s')
parser.add_argument('--displayInterval', type=int, default=100, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
parser.add_argument('--stage', type=int, default=0, help='stage augmentation')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
parser.add_argument('--debug', action='store_true', help='debug images')
opt = parser.parse_args()
print(opt)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print(f'torch.cuda.is_available {torch.cuda.is_available()}')


test_dataset = dataset.imageDataset(opt.root)
# test_dataset = dataset.lprDataset()
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=opt.batchSize,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))

nclass = len(opt.alphabet) + 1
nc = 1

converter = utils.strLabelConverter(opt.alphabet, ignore_case=False)
criterion = CTCLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


model = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
model.apply(weights_init)

if opt.pretrained != '':
    print('loading pretrained model from %s' % opt.pretrained)
    if opt.pretrained == 'data/crnn.pth':
        model_pretrained = crnn.CRNN(opt.imgH, nc, 37, opt.nh)
        model_pretrained.load_state_dict(torch.load(opt.pretrained))
        for W, W_pre in zip(model.parameters(), model_pretrained.parameters()):
            try:
                w = [x for x in W.size()]
                w_pre = [x for x in W_pre.size()]
                if w == w_pre:
                    W.data.copy_(W_pre)
            except Exception as e:
                print(e)

        # for W, W_pre in zip(model.parameters(), model_pretrained.parameters()):
        #     if np.array_equal(W.numpy(), W_pre.numpy()):
        #         print('weights transferred!!')
        #     break

    else:
        state_dict = torch.load(opt.pretrained)
        state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            state_dict_rename[name] = v
        model.load_state_dict(state_dict_rename)

# print(model)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(opt.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg = utils.averager()


# def val(model, test_loader, criterion):
print('Start val')

for p in model.parameters():
    p.requires_grad = False

model.eval()

val_iter = iter(test_loader)

i = 0
n_correct = 0
loss_avg = utils.averager()

max_iter = len(test_loader)
for i in range(max_iter):
    data = val_iter.next()
    i += 1
    # if opt.debug:
    #     cpu_images, cpu_texts, orig_imgs = data
    # else:
    cpu_images, cpu_texts = data

    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = model(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    loss_avg.add(cost)

    _, preds = preds.max(2)
    # preds = preds.squeeze(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
    i = 0
    for pred, target in zip(sim_preds, cpu_texts):
        if opt.debug:
            print(np.swapaxes(cpu_images[i].numpy(), 0, 2).shape)
            cv2.imshow(f'pred: {pred}, gt: {target}', np.moveaxis(cpu_images[i].numpy(), 0, -1))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            i += 1
        if pred == target:
            n_correct += 1

raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
    print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

accuracy = n_correct / float(max_iter * opt.batchSize)
print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    # return accuracy

# accuracy = val(model, test_loader, criterion)

