import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset

from torch.autograd import Variable
from model import NetworkCIFAR as Network

import faulthandler
import torch.onnx

# example
# python train.py --auxiliary --layers 8

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../../data', help='location of the data corpus')
parser.add_argument('--onnx_export_path', type=str, default='../model/darts.proto', help='locatin of the exported model onnx file')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


def main():
  faulthandler.enable()

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  logging.info("args = %s", args)

  # load trained model, without trained weights..
  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  
  # export to onnx
  dummy_input = Variable(torch.randn(96, 3, 32, 32))
  torch.onnx.export(model, dummy_input, args.onnx_export_path, verbose=True)
  


if __name__ == '__main__':
  main() 

