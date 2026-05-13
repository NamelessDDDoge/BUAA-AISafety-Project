import argparse
import os
import time
from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import init

from utils.earlystop import EarlyStopping

"""Currently assumes jpg_prob, blur_prob 0 or 1"""


class Trainer(nn.Module):
    def name(self):
        return "Trainer"

    def __init__(self, opt):
        from models import get_model

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
            time.sleep(3)
            params = self.model.parameters()

        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Trainable parameters: {trainable_params}")

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


def get_train_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default="binary")
    parser.add_argument("--name", type=str, default="experiment_name")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--dataroot", type=str, default=None)
    parser.add_argument("--arch", type=str, default="res50")

    parser.add_argument("--fix_backbone", action="store_true")
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--new_optim", action="store_true")
    parser.add_argument("--init_gain", type=float, default=0.02)
    parser.add_argument("--init_type", type=str, default="normal")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--suffix", default="", type=str)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--niter", type=int, default=100)
    parser.add_argument("--earlystop_epoch", type=int, default=5)
    parser.add_argument("--epoch_count", type=int, default=1)
    parser.add_argument("--last_epoch", type=int, default=-1)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")

    parser.add_argument("--loss_freq", type=int, default=400)
    parser.add_argument("--save_epoch_freq", type=int, default=1)

    parser.add_argument("--data_aug", action="store_true")
    parser.add_argument("--no_resize", action="store_true")
    parser.add_argument("--no_crop", action="store_true")
    parser.add_argument("--no_flip", action="store_true")
    parser.add_argument("--resize_or_crop", type=str, default="scale_and_crop")
    parser.add_argument("--rz_interp", type=str, nargs="+", default="bilinear")
    parser.add_argument("--loadSize", type=int, default=256)
    parser.add_argument("--cropSize", type=int, default=224)
    parser.add_argument("--blur_sig", type=str, nargs="+", default="0.0,3.0")
    parser.add_argument("--blur_prob", type=float, default=0.5)
    parser.add_argument("--jpg_qual", type=str, nargs="+", default="30,100")
    parser.add_argument("--jpg_prob", type=float, default=0.5)
    parser.add_argument("--jpg_method", type=str, nargs="+", default="cv2,pil")

    parser.add_argument("--serial_batches", action="store_true")
    parser.add_argument("--data_label", type=str, default="train")

    parser.add_argument("--class_bal", action="store_true")
    parser.add_argument("--num_threads", type=int, default=4)

    parser.add_argument("--gpu_ids", type=str, default="0")

    parser.add_argument(
        "--data_mode",
        type=str,
        default="ours",
        choices=["ours", "wang2020", "ours_wang2020"],
    )
    parser.add_argument("--wang2020_data_path", type=str, default=None)
    parser.add_argument(
        "--real_list_path", type=str, default=None
    )  # only used when data_mode='ours'
    parser.add_argument(
        "--fake_list_path", type=str, default=None
    )  # only used when data_mode='ours'

    opt = parser.parse_args()
    opt.isTrain = True

    def split_csv_values(value, convert=str):
        if isinstance(value, str):
            values = value.split(",")
        else:
            values = []
            for item in value:
                values.extend(str(item).split(","))
        return [convert(item) for item in values if item != ""]

    if opt.suffix:
        opt.name = opt.name + "_" + opt.suffix.format(**vars(opt))

    opt.rz_interp = split_csv_values(opt.rz_interp)
    opt.blur_sig = split_csv_values(opt.blur_sig, float)
    opt.jpg_method = split_csv_values(opt.jpg_method)
    opt.jpg_qual = split_csv_values(opt.jpg_qual, int)
    if len(opt.jpg_qual) == 2:
        opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
    elif len(opt.jpg_qual) > 2:
        raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

    opt.gpu_ids = [
        int(gpu_id)
        for gpu_id in str(opt.gpu_ids).split(",")
        if gpu_id != "" and int(gpu_id) >= 0
    ]

    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(os.path.join(expr_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(expr_dir, "val"), exist_ok=True)

    return opt


def get_val_opt(train_opt):
    val_opt = deepcopy(train_opt)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = "val"
    val_opt.jpg_method = ["pil"]
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


if __name__ == "__main__":
    opt = get_train_opt()
    val_opt = get_val_opt(opt)

    from data import create_dataloader
    from tensorboardX import SummaryWriter
    from utils.evaluation import validate

    model = Trainer(opt)

    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    early_stopping = EarlyStopping(
        patience=opt.earlystop_epoch, delta=-0.001, verbose=True
    )
    start_time = time.time()
    print("Length of training data loader: %d" % (len(data_loader)))
    print("Length of validation data loader: %d" % (len(val_loader)))
    for epoch in range(opt.niter):
        for i, data in enumerate(data_loader):
            model.total_steps += 1

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print(
                    "Train loss: {} at step: {}".format(model.loss, model.total_steps)
                )
                train_writer.add_scalar("loss", model.loss, model.total_steps)
                print("Iter time: ", ((time.time() - start_time) / model.total_steps))

            # if model.total_steps in [
            #     10,
            #     30,
            #     50,
            #     100,
            #     1000,
            #     5000,
            #     10000,
            # ]:  # save models at these iters
            #     model.save_networks("model_iters_%s.pth" % model.total_steps)

        if epoch % opt.save_epoch_freq == 0:
            print("saving the model at the end of epoch %d" % (epoch))
            model.save_networks("model_epoch_best.pth")
            model.save_networks("model_epoch_%s.pth" % epoch)

        model.eval()
        ap, r_acc, f_acc, acc = validate(model.model, val_loader, model.device)
        val_writer.add_scalar("accuracy", acc, model.total_steps)
        val_writer.add_scalar("ap", ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(
                    patience=opt.earlystop_epoch, delta=-0.002, verbose=True
                )
            else:
                print("Early stopping.")
                break
        model.train()
