import re
import numpy as np
import torch
import torch.distributed as dist
import collections
import logging
import random
import argparse



class LossMeter(object):
    def __init__(self, maxlen=100):
        """Computes and stores the running average"""
        self.vals = collections.deque([], maxlen=maxlen)

    def __len__(self):
        return len(self.vals)

    def update(self, new_val):
        self.vals.append(new_val)

    @property
    def val(self):
        return sum(self.vals) / len(self.vals)

    def __repr__(self):
        return str(self.val)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_state_dict(state_dict_path, loc='cpu'):
    state_dict = torch.load(state_dict_path, map_location=loc)
    # Change Multi GPU to single GPU
    original_keys = list(state_dict.keys())
    for key in original_keys:
        if key.startswith("module."):
            new_key = key[len("module."):]
            state_dict[new_key] = state_dict.pop(key)
    return state_dict


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

def parse_args():
    parser = argparse.ArgumentParser()
    # Model Loading
    parser.add_argument('--model', 
                        default='t5-large', 
                        type=str, help='path to pretrained model')
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--output', type=str, 
                        default='boxbart_checkpoint',
                        help='Save the model (usually the fine-tuned model).')

    # Training Hyper-parameters
    parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
    parser.add_argument('-b', '--batch_size', default=8, type=int, #32
                        help='mini-batch size (default: 256)')
    parser.add_argument('--valid_batch_size', type=int, default=8) #32
    parser.add_argument('--beam_size', type=int, default=5
                        )
    parser.add_argument('--num_predictions',type=int, default=10)

    # CPU/GPU
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--distributed", action='store_true')

    # Data Splits
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--copy', action='store_true')
    parser.add_argument("--dataset_dir", 
                        default='chemner_filter_cleaned_data', 
                        type=str,  help='which dataset')

    # Quick experiments
    parser.add_argument('--train_topk', type=int, default=-1)
    parser.add_argument('--valid_topk', type=int, default=-1)

    parser.add_argument('--neg_num', default=5, type=int,
                        help='number of context entities used for negatives')

    # Training configuration
    parser.add_argument('--clip_grad_norm', type=float, default=10.0)
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_eps", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lr-scheduler', default='linear', type=str,
                    help='Lr scheduler to use')
    parser.add_argument('--patient', type=int, default=4)
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run') #10
    parser.add_argument('-j', '--workers', default=8, type=int,
                    help='number of data loading workers')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--warmup', default=400, type=int,
                        help='warmup steps')


    args = parser.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    return args