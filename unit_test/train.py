
<<<<<<< HEAD

import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append("..")
# Add lib to PYTHONPATH
# lib_path = os.path.join(os.path.abspath(
#     os.path.join(root_dir, '../libs')))
# sys.path.append(lib_path)

import time
import argparse
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from libs.utils.augmentations import SSDAugmentation
from libs.networks.vgg_refinedet import VGGRefineDet
from libs.networks.resnet_refinedet import ResNetRefineDet
from libs.dataset.config import voc320, coco320, MEANS
from libs.dataset.coco import COCO_ROOT, COCODetection, COCO_CLASSES
from libs.dataset.voc0712 import VOC_ROOT, VOCDetection, \
    VOCAnnotationTransform
from libs.dataset import *
import cPickle

import pdb

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')
=======
import os
import sys
sys.path.append('../')

from data import *
from utils.augmentations import SSDAugmentation
from layers.modules.refinedet_loss import BiBoxLoss, MultiBoxLoss
from refinedet import build_refinedet
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import cPickle

from pycallgraph import PyCallGraph, Config, GlobbingFilter
from pycallgraph.output import GraphvizOutput


import pdb

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")
>>>>>>> 3efa668f4283428ec27558c51bf55872947c7de6


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
<<<<<<< HEAD
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default='/root/dataset/voc/VOCdevkit/',
                    help='Dataset root directory path')
# parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
#                     help='Pretrained base model')
parser.add_argument('--basenet', default='resnet101.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
=======
parser.add_argument('--dataset', default='COCO', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default='/root/dataset/coco',
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=8, type=int,
>>>>>>> 3efa668f4283428ec27558c51bf55872947c7de6
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
<<<<<<< HEAD
parser.add_argument('--num_workers', default=0, type=int,
=======
parser.add_argument('--num_workers', default=4, type=int,
>>>>>>> 3efa668f4283428ec27558c51bf55872947c7de6
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
<<<<<<< HEAD
parser.add_argument('--save_folder', default='../weights/resnet101',
=======
parser.add_argument('--save_folder', default='weights/',
>>>>>>> 3efa668f4283428ec27558c51bf55872947c7de6
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if torch.cuda.is_available():
<<<<<<< HEAD
    print('CUDA devices: ', torch.cuda.device)
    print('GPU numbers: ', torch.cuda.device_count())

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print('WARNING: It looks like you have a CUDA device, but are not' +
              'using CUDA.\nRun with --cuda for optimal training speed.')
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

def main():
    images_path = './images.pkl'
    targets_path = './targets.pkl'
    
    args.dataset_root = COCO_ROOT
    cfg = coco320
    # data
    dataset = COCODetection(root=args.dataset_root,
                            transform=SSDAugmentation(cfg['min_dim'], MEANS))
    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=False, collate_fn=detection_collate,
                                  pin_memory=True)
    ## load train data
    batch_iterator = iter(data_loader)
    # data_loader[0]
    images_list, targets_list = next(batch_iterator)
    pdb.set_trace()
    # images = torch.stack(images_list, 0)
    images_array_list = [Variable(cur_image).data.numpy() for cur_image in images_list]
    targets_array_list = [Variable(cur_targets).data.numpy() for cur_targets in targets_list]
    # store data
    fw_images = open(images_path, 'wb')
    cPickle.dump(images_array_list, fw_images)
    fw_images.close()
    fw_targets = open(targets_path, 'wb')
    cPickle.dump(targets_array_list, fw_targets)
    fw_targets.close()
=======
  print('CUDA devices: ', torch.cuda.device)
  print('GPU numbers: ', torch.cuda.device_count())

if torch.cuda.is_available():
  if args.cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
  if not args.cuda:
    print("WARNING: It looks like you have a CUDA device, but aren't " +
          "using CUDA.\nRun with --cuda for optimal training speed.")
    torch.set_default_tensor_type('torch.FloatTensor')
else:
  torch.set_default_tensor_type('torch.FloatTensor')




def xavier(param):
  init.xavier_uniform(param)


def weights_init(m):
  if isinstance(m, nn.Conv2d):
    xavier(m.weight.data)
    m.bias.data.zero_()

def main():

  
  # run(
  #   'include',
  #   trace_filter=trace_filter,
  #   comment='Should show secret_function.'
  # )
  
  graphviz = GraphvizOutput()
  graphviz.output_file = 'basic.svg'
  graphviz.output_type = 'svg'
  config = Config()
  trace_filter = GlobbingFilter(include=[
    'layers.*',
    'utils.*',
    'data.*',
    'refinedet.*'
  ])
  # trace_filter = GlobbingFilter(exclude=[
  #   'pdb.*',
  #   'pycallgraph.*'
  # ])
  
  config.trace_filter = trace_filter
  with PyCallGraph(output=graphviz, config=config):
    # pdb.set_trace()
    
    images_path = './images.pkl'
    targets_path = './targets.pkl'
    
    
    args.dataset_root = COCO_ROOT
    cfg = coco
    
    # data
    # dataset = COCODetection(root=args.dataset_root,
    #                        transform=SSDAugmentation(cfg['min_dim'],
    #                                                  MEANS))
    # step_index = 0
    #
    # data_loader = data.DataLoader(dataset, args.batch_size,
    #                               num_workers=args.num_workers,
    #                               shuffle=False, collate_fn=detection_collate,
    #                               pin_memory=True)
    # ## load train data
    # batch_iterator = iter(data_loader)
    # # data_loader[0]
    # images_list, targets_list = next(batch_iterator)
    # # images = torch.stack(images_list, 0)
    # images_array_list = [Variable(cur_image).data.numpy() for cur_image in images_list]
    # targets_array_list = [Variable(cur_targets).data.numpy() for cur_targets in targets_list]
    # # store data
    # cPickle.dump(images_array_list, open(images_path, 'wb'))
    # cPickle.dump(targets_array_list, open(targets_path, 'wb'))
>>>>>>> 3efa668f4283428ec27558c51bf55872947c7de6
    
    # load data
    images_array_list = cPickle.load(open(images_path, 'rb'))
    targets_array_list = cPickle.load(open(targets_path, 'rb'))
    
    images = torch.stack([torch.Tensor(cur_image) for cur_image in
                          images_array_list], 0)
    targets = [torch.Tensor(cur_target) for cur_target in targets_array_list]
    
    print(type(images))
    print(type(targets))
    
<<<<<<< HEAD
    refinedet = ResNetRefineDet(cfg['num_classes'], cfg)

    refinedet.create_architecture(
        os.path.join(args.save_folder, args.basenet), pretrained=True,
        fine_tuning=True)
    net = refinedet
    if args.cuda:
        net = torch.nn.DataParallel(refinedet).cuda()
        cudnn.benchmark = True

    params = net.state_dict()
    # for k, v in params.items():
    #     print(k)
    #     print(v.shape)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    net.train()
    print('Using the specified args:')
    print(args)

=======
    
    # pdb.set_trace()
    refinedet_net = build_refinedet('train', cfg, cfg['min_dim'],
                                    cfg['num_classes'])
    
    if args.cuda:
      net = refinedet_net.cuda()
      # net = torch.nn.DataParallel(refinedet_net).cuda()
    else:
      net = refinedet_net
      
    device_ids = [0]
    cudnn.benchmark = True
    weights_path = '../weights/vgg16_reducedfc.pth'
    vgg_weights = torch.load(weights_path)
    refinedet_net.vgg.load_state_dict(vgg_weights)
    
    
    print('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    net.extras.apply(weights_init)
    net.bi_loc.apply(weights_init)
    net.bi_conf.apply(weights_init)

    net.back_pyramid.apply(weights_init)
    net.multi_loc.apply(weights_init)
    net.multi_conf.apply(weights_init)
    
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    # neg_pos ratio 3:1
    bi_criterion = BiBoxLoss(0.5, True, 0, True, 3, 0.5,
                             args.cuda)
    
    multi_criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True,
                                   3, 0.5, 0.6, args.cuda)
    
    net.train()
    # loss counters
    print('Loading the dataset...')
    print('Using the specified args:')
    print(args)
    
      
>>>>>>> 3efa668f4283428ec27558c51bf55872947c7de6
    if args.cuda:
        images = Variable(images.cuda())
        targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
    else:
        images = Variable(images)
        targets = [Variable(ann, volatile=True) for ann in targets]
<<<<<<< HEAD

    # forward
    t0 = time.time()
    optimizer.zero_grad()
    bi_loss_loc, bi_loss_conf, multi_loss_loc, multi_loss_conf = \
        net(images, targets)
    loss = bi_loss_loc + bi_loss_conf + multi_loss_loc + multi_loss_conf
    loss.backward()
    optimizer.step()
    t1 = time.time()
=======
        
    # forward
    t0 = time.time()
    bi_out, multi_out, priors = net(images)
    # backprop
    optimizer.zero_grad()
    # pdb.set_trace()
    bi_loss_l, bi_loss_c = bi_criterion(bi_out, priors, targets)
    multi_loss_l, multi_loss_c = multi_criterion(bi_out, multi_out, priors, targets)
    loss = bi_loss_l + bi_loss_c + multi_loss_l + multi_loss_c
    print(loss.data[0])
    
    loss.backward()
    optimizer.step()
    t1 = time.time()
    
    bi_loc_loss = 0
    bi_conf_loss = 0
    multi_loc_loss = 0
    multi_conf_loss = 0
    
    bi_loc_loss += bi_loss_l.data[0]
    bi_conf_loss += bi_loss_c.data[0]
    multi_loc_loss += multi_loss_l.data[0]
    multi_conf_loss += multi_loss_c.data[0]
    
>>>>>>> 3efa668f4283428ec27558c51bf55872947c7de6
    print('timer: %.4f sec.' % (t1 - t0))


if __name__ == '__main__':
<<<<<<< HEAD
    main()
=======
  main()
>>>>>>> 3efa668f4283428ec27558c51bf55872947c7de6

