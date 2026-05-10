import os
import time

import torch
import torch.nn as nn
from torch.nn import init

from models import get_model  # 假设 models.py 在你的路径中


def init_weights(net, init_type="normal", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


class Trainer(nn.Module):
    def name(self):
        return "Trainer"

    def __init__(self, opt):
        super(Trainer, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.device = (
            torch.device(f"cuda:{opt.gpu_ids[0]}")
            if opt.gpu_ids
            else torch.device("cpu")
        )

        self.model = get_model(opt.arch)

        init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)

        if opt.fix_backbone:
            params = []
            for name, p in self.model.named_parameters():
                if name == "fc.weight" or name == "fc.bias":
                    params.append(p)
                else:
                    p.requires_grad = False
        else:
            print(
                "Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun"
            )
            import time

            time.sleep(3)
            params = self.model.parameters()

        if opt.optim == "adam":
            self.optimizer = torch.optim.AdamW(
                params,
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay,
            )
        elif opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay
            )
        else:
            raise ValueError("Optim should be [adam, sgd]")

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.model.to(self.device)

    def save_networks(self, save_filename):
        save_path = os.path.join(self.save_dir, save_filename)
        state_dict = self.model.fc.state_dict()
        torch.save(state_dict, save_path)
        print(f"FC layer weights saved to {save_path}")

    def eval(self):
        self.model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] /= 10.0
            if param_group["lr"] < min_lr:
                return False
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

    def forward(self):
        self.output = self.model(self.input)
        self.output = self.output.view(-1).unsqueeze(1)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
