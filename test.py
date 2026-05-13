import argparse
import os
import shutil
import torch
import torch.utils.data

from data.test_data import DEFAULT_CONFIG_PATH, TestDataset, TestDatasetSpec, load_test_dataset_specs
from models import get_model
from utils.evaluation import validate
from utils.nn_evaluation import load_or_build_feature_bank, validate_nn
from utils.reproducibility import set_seed

SEED = 0

if __name__ == '__main__':


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--real_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--fake_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--data_mode', type=str, default=None, help='wang2020 or ours')
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG_PATH), help='datasets.yaml path')
    parser.add_argument('--max_sample', type=int, default=1000, help='only check this number of images for both fake/real')

    parser.add_argument('--method', type=str, default='linear', choices=['linear', 'nn'], help='linear uses the learned fc head; nn uses the paper nearest-neighbor feature bank')
    parser.add_argument('--arch', type=str, default='CLIP:ViT-L/14')
    parser.add_argument('--ckpt', type=str, default='./weights/fc_weights.pth')
    parser.add_argument('--nn_k', type=int, default=1, help='number of nearest features per class to average for NN')
    parser.add_argument('--nn_bank_path', type=str, default='datasets/train', help='training feature-bank root used for both real and fake when class folders contain 0_real/1_fake')
    parser.add_argument('--nn_bank_real_path', type=str, default=None, help='optional real-image feature-bank path')
    parser.add_argument('--nn_bank_fake_path', type=str, default=None, help='optional fake-image feature-bank path')
    parser.add_argument('--nn_bank_data_mode', type=str, default='wang2020', choices=['wang2020', 'ours'])
    parser.add_argument('--nn_bank_max_sample', type=int, default=None, help='optional max feature-bank images per class')
    parser.add_argument('--nn_bank_cache', type=str, default=None, help='optional .pth cache for extracted NN feature bank')
    parser.add_argument('--nn_bank_chunk_size', type=int, default=8192, help='bank chunk size for cosine nearest-neighbor search')

    parser.add_argument('--result_folder', type=str, default='result', help='')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--jpeg_quality', type=int, default=None, help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None, help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")


    opt = parser.parse_args()

    
    if os.path.exists(opt.result_folder):
        shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(opt)
    if opt.method == 'linear':
        state_dict = torch.load(opt.ckpt, map_location='cpu')
        model.fc.load_state_dict(state_dict)
    print(f"Model loaded on {device}..", flush=True)
    model.eval()
    model.to(device)

    nn_bank = None
    if opt.method == 'nn':
        nn_bank = load_or_build_feature_bank(model, opt, device)
        print(f"NN feature bank: real={len(nn_bank.real)} fake={len(nn_bank.fake)} k={opt.nn_k}", flush=True)

    if (opt.real_path == None) or (opt.fake_path == None) or (opt.data_mode == None):
        dataset_paths = [
            dict(key=spec.key, real_path=spec.real_path, fake_path=spec.fake_path, data_mode=spec.data_mode)
            for spec in load_test_dataset_specs(opt.config)
        ]
    else:
        dataset_paths = [ dict(key='custom', real_path=opt.real_path, fake_path=opt.fake_path, data_mode=opt.data_mode) ]



    print(f"Evaluating {len(dataset_paths)} dataset(s)...", flush=True)

    for dataset_idx, dataset_path in enumerate(dataset_paths, start=1):
        set_seed(SEED)
        print(
            f"[{dataset_idx}/{len(dataset_paths)}] Preparing dataset: {dataset_path['key']}",
            flush=True,
        )

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

        loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
        print(
            f"[{dataset_idx}/{len(dataset_paths)}] Dataset {dataset_path['key']}: "
            f"{len(dataset)} images, {len(loader)} batches",
            flush=True,
        )
        if opt.method == 'nn':
            # Paper uses the natural 0.5 threshold (closer class wins); no threshold tuning.
            ap, r_acc0, f_acc0, acc0 = validate_nn(
                model,
                loader,
                nn_bank,
                device,
                k=opt.nn_k,
                bank_chunk_size=opt.nn_bank_chunk_size,
                find_thres=False,
                progress_prefix=f"[{dataset_idx}/{len(dataset_paths)}] {dataset_path['key']}",
            )
            print(
                f"[{dataset_idx}/{len(dataset_paths)}] Finished {dataset_path['key']}: "
                f"AP={ap * 100:.2f}, ACC@0.5={acc0 * 100:.2f}",
                flush=True,
            )
        else:
            ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(model, loader, device, find_thres=True)
            print(
                f"[{dataset_idx}/{len(dataset_paths)}] Finished {dataset_path['key']}: "
                f"AP={ap * 100:.2f}, ACC@0.5={acc0 * 100:.2f}, "
                f"best_thres={best_thres:.6f}, ACC@best={acc1 * 100:.2f}",
                flush=True,
            )

        with open( os.path.join(opt.result_folder,'ap.txt'), 'a') as f:
            f.write(dataset_path['key']+': ' + str(round(ap*100, 2))+'\n' )

        with open( os.path.join(opt.result_folder,'acc0.txt'), 'a') as f:
            f.write(dataset_path['key']+': ' + str(round(r_acc0*100, 2))+'  '+str(round(f_acc0*100, 2))+'  '+str(round(acc0*100, 2))+'\n' )
