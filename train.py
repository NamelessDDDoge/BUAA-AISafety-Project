import argparse
import os
import time

from earlystop import EarlyStopping
from tensorboardX import SummaryWriter

from data import create_dataloader
from models import get_model
from networks.trainer import Trainer

from .test import validate

"""Currently assumes jpg_prob, blur_prob 0 or 1"""


def get_train_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="experiment_name")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints")
    parser.add_argument("--dataroot", type=str, default="./datasets/progan_train")
    parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")

    parser.add_argument("--fix_backbone", action="store_true", default=True)
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--init_gain", type=float, default=0.02)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0001)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--niter", type=int, default=100)
    parser.add_argument("--earlystop_epoch", type=int, default=5)

    parser.add_argument("--loss_freq", type=int, default=400)
    parser.add_argument("--save_epoch_freq", type=int, default=1)

    parser.add_argument("--no_resize", action="store_true")
    parser.add_argument("--no_crop", action="store_true")
    parser.add_argument("--no_flip", action="store_true")
    parser.add_argument("--rz_interp", type=str, nargs="+", default=["bilinear"])
    parser.add_argument("--loadSize", type=int, default=256)
    parser.add_argument("--cropSize", type=int, default=224)
    parser.add_argument("--blur_sig", type=float, nargs="+", default=[0.0, 3.0])
    parser.add_argument("--blur_prob", type=float, default=0.5)
    parser.add_argument("--jpg_qual", type=int, nargs="+", default=[30, 100])
    parser.add_argument("--jpg_prob", type=float, default=0.5)
    parser.add_argument("--jpg_method", type=str, nargs="+", default=["pil"])

    parser.add_argument("--serial_batches", action="store_true")
    parser.add_argument("--data_label", type=str, default="train")

    parser.add_argument("--class_bal", action="store_true")
    parser.add_argument("--num_threads", type=int, default=4)

    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0])

    parser.add_argument(
        "--data_mode",
        type=str,
        default="wang2020",
        choices=["ours", "wang2020", "ours_wang2020"],
    )
    parser.add_argument(
        "--wang2020_data_path", type=str, default="./datasets/progan_train"
    )
    parser.add_argument(
        "--real_list_path", type=str, default=""
    )  # data_mode='ours' 时才用
    parser.add_argument(
        "--fake_list_path", type=str, default=""
    )  # data_mode='ours' 时才用

    opt = parser.parse_args()
    opt.isTrain = True

    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(os.path.join(expr_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(expr_dir, "val"), exist_ok=True)

    return opt


def get_val_opt(train_opt):
    from copy import deepcopy

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

    model = Trainer(opt)

    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    early_stopping = EarlyStopping(
        patience=opt.earlystop_epoch, delta=-0.001, verbose=True
    )
    start_time = time.time()
    print("Length of data loader: %d" % (len(data_loader)))
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

        # Validation
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
