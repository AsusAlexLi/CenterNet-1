from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

from opts import opts
from detectors.detector_factory import detector_factory

import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from net.CenterNet import CenterNet

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


import scipy.spatial
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def extract_pt_weights(model_path, debug):
    # load model from pytorch files
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage.cpu())
    state_dict = checkpoint['state_dict']
    weights = {}

    for key, value in state_dict.items():
        weight = np.array(value)
        if weight.ndim == 4:
            weight = np.moveaxis(weight, [0, 1, 2, 3], [2, 3, 1, 0])
            weights[key] = np.swapaxes(weight, 2, 3)
        else:
            weights[key] = weight
    if debug:
        for key, value in weights.items():
            print('[Layer] ' + key + ', [Shape] ', value.shape)
    return weights


def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    ret = detector.run(opt.demo)

  
if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
