import argparse
import os
import torch
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from models import get_model
from copy import deepcopy
import random
import shutil
from data.test_data import DEFAULT_CONFIG_PATH, TestDataset, TestDatasetSpec, load_test_dataset_specs

SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    if y_pred[0:N//2].max() <= y_pred[N//2:N].min(): # perfectly separable case
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2 

    best_acc = 0 
    best_thres = 0 
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp>=thres] = 1 
        temp[temp<thres] = 0 

        acc = (temp == y_true).sum() / N  
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc 
    
    return best_thres
        

def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc    


def validate(model, loader, device, find_thres=False):

    with torch.no_grad():
        y_true, y_pred = [], []
        print ("Length of dataset: %d" %(len(loader)))
        for img, label in loader:
            in_tens = img.to(device)

            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # ================== save this if you want to plot the curves =========== # 
    # torch.save( torch.stack( [torch.tensor(y_true), torch.tensor(y_pred)] ),  'baseline_predication_for_pr_roc_curve.pth' )
    # exit()
    # =================================================================== #
    
    # Get AP 
    ap = average_precision_score(y_true, y_pred)

    # Acc based on 0.5
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    if not find_thres:
        return ap, r_acc0, f_acc0, acc0


    # Acc based on the best thres
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres




if __name__ == '__main__':


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--real_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--fake_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--data_mode', type=str, default=None, help='wang2020 or ours')
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG_PATH), help='datasets.yaml path')
    parser.add_argument('--max_sample', type=int, default=1000, help='only check this number of images for both fake/real')

    parser.add_argument('--arch', type=str, default='CLIP:ViT-L/14')
    parser.add_argument('--ckpt', type=str, default='./weights/fc_weights.pth')

    parser.add_argument('--result_folder', type=str, default='result', help='')
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--jpeg_quality', type=int, default=None, help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None, help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")


    opt = parser.parse_args()

    
    if os.path.exists(opt.result_folder):
        shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(opt.arch)
    state_dict = torch.load(opt.ckpt, map_location='cpu')
    model.fc.load_state_dict(state_dict)
    print(f"Model loaded on {device}..")
    model.eval()
    model.to(device)

    if (opt.real_path == None) or (opt.fake_path == None) or (opt.data_mode == None):
        dataset_paths = [
            dict(key=spec.key, real_path=spec.real_path, fake_path=spec.fake_path, data_mode=spec.data_mode)
            for spec in load_test_dataset_specs(opt.config)
        ]
    else:
        dataset_paths = [ dict(key='custom', real_path=opt.real_path, fake_path=opt.fake_path, data_mode=opt.data_mode) ]



    for dataset_path in (dataset_paths):
        set_seed()

        spec = TestDatasetSpec(
            key=dataset_path['key'],
            real_path=dataset_path['real_path'],
            fake_path=dataset_path['fake_path'],
            data_mode=dataset_path['data_mode'],
        )
        dataset = TestDataset(  spec,
                                max_sample=opt.max_sample, 
                                arch=opt.arch,
                                jpeg_quality=opt.jpeg_quality, 
                                gaussian_sigma=opt.gaussian_sigma,
                                )

        loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
        ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(model, loader, device, find_thres=True)

        with open( os.path.join(opt.result_folder,'ap.txt'), 'a') as f:
            f.write(dataset_path['key']+': ' + str(round(ap*100, 2))+'\n' )

        with open( os.path.join(opt.result_folder,'acc0.txt'), 'a') as f:
            f.write(dataset_path['key']+': ' + str(round(r_acc0*100, 2))+'  '+str(round(f_acc0*100, 2))+'  '+str(round(acc0*100, 2))+'\n' )
