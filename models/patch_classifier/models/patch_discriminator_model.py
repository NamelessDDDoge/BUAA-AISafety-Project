import torch
import torch.nn as nn
from collections import OrderedDict
from ..utils import renormalize, imutil
from .base_model import BaseModel
from .networks import networks
import numpy as np
import logging
import cv2
from PIL import Image
from collections import namedtuple


class PatchDiscriminatorModel(BaseModel):
    def name(self):
        return "PatchDiscriminatorModel"

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # load/define networks
        self.net_D = networks.define_patch_D(opt.arch, None, opt.gpu_ids)

    def forward(self, input):
        return self.net_D(input)

    def to(self, device):
        self.net_D = self.net_D.to(device)
        self.device = device
        return self

    def parameters(self):
        return self.net_D.parameters()

    def __call__(self, *args, **kwargs):
        return self.net_D(*args, **kwargs)

    def train(self):
        self.net_D.train()
        return self

    def eval(self):
        self.net_D.eval()
        return self

    def state_dict(self, *args, **kwargs):
        return self.net_D.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.net_D.load_state_dict(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.net_D, name)

    def reset(self):
        # for debugging .. clear all the cached variables
        self.loss_D = None
        self.acc_D = None
        self.acc_D_raw = None
        self.acc_D_voted = None
        self.acc_D_avg = None
        self.ims = None
        self.labels = None
        self.pred_logit = None
