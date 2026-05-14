"""
Run all experiments in sequence.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None, help='Training data root (wang2020 format)')
    parser.add_argument('--bank_path', type=str, default=None, help='ProGAN data root for NN bank')
    parser.add_argument('--max_sample', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--skip', nargs='*', default=[], choices=['robustness', 'backbone', 'datasize', 'diversity', 'clip_layers'])
    args = parser.parse_args()

    if 'robustness' not in args.skip:
        print('\n' + '='*60)
        print('EXPERIMENT 1/5: Robustness (Sec 6.7)')
        print('='*60)
        from experiments import exp_robustness
        import argparse as ap
        r_args = ap.Namespace(max_sample=args.max_sample)
        exp_robustness.run(r_args)

    if 'backbone' not in args.skip:
        print('\n' + '='*60)
        print('EXPERIMENT 2/5: Backbone comparison (Sec 6.3)')
        print('='*60)
        from experiments import exp_backbone
        import argparse as ap
        b_args = ap.Namespace(data_path=args.data_path, max_sample=args.max_sample,
                              batch_size=args.batch_size, epochs=30)
        exp_backbone.run(b_args)

    if 'datasize' not in args.skip:
        print('\n' + '='*60)
        print('EXPERIMENT 3/5: Data size effect (Sec 6.5)')
        print('='*60)
        from experiments import exp_datasize
        import argparse as ap
        d_args = ap.Namespace(data_path=args.data_path, max_sample=args.max_sample,
                              batch_size=args.batch_size, epochs=30, retrain=False)
        exp_datasize.run(d_args)

    if 'diversity' not in args.skip:
        print('\n' + '='*60)
        print('EXPERIMENT 4/5: Dataset diversity (Appendix C.2)')
        print('='*60)
        from experiments import exp_diversity
        import argparse as ap
        dv_args = ap.Namespace(data_path=args.bank_path or args.data_path,
                               max_sample=args.max_sample, max_bank_per_class=2000)
        exp_diversity.run(dv_args)

    if 'clip_layers' not in args.skip:
        print('\n' + '='*60)
        print('EXPERIMENT 5/5: CLIP layer ablation (Appendix C.1)')
        print('='*60)
        from experiments import exp_clip_layers
        import argparse as ap
        cl_args = ap.Namespace(bank_path=args.bank_path or args.data_path,
                               max_sample=args.max_sample, max_bank=5000, batch_size=args.batch_size)
        exp_clip_layers.run(cl_args)

    print('\n' + '='*60)
    print('All experiments complete. Results in experiments/results/')
    print('='*60)

if __name__ == '__main__':
    main()
