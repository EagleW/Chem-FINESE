import argparse
import random
from ast import literal_eval
import numpy as np
import torch



def parse_args():
    parser = argparse.ArgumentParser()
    # Data Splits
    parser.add_argument('--test_only', action='store_true')

    # Quick experiments
    parser.add_argument('--train_topk', type=int, default=-1)
    parser.add_argument('--valid_topk', type=int, default=-1)

    # Training Hyper-parameters
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--valid_batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--patient', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--neg_num', default=5, type=int,
                        help='number of context entities used for negatives')

    parser.add_argument('--wandb', action='store_true')

    # Debugging
    parser.add_argument('--output', type=str, default='boxbart_checkpoint')

    # Model Loading
    parser.add_argument('--load', type=str, default=None,
                        help='Load the model (usually the fine-tuned model).')
    parser.add_argument('--load_cons', type=str, default=None,
                        help='Load the construction model (usually the fine-tuned model).')
   

    
    # Optimization
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--lr_mul', type=int, default=1)
    parser.add_argument('--lr-scheduler', default='cosine', type=str,
                    help='Lr scheduler to use')
    parser.add_argument("--warmup_steps", default=400, type=int)

    # Pre-training Config
    parser.add_argument("--dataset_dir", default='chemner_filter_cleaned_data', type=str)
    
    # CPU/GPU
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument("--distributed", action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('-j', '--workers', default=8, type=int,
                    help='number of data loading workers')

    # Inference
    parser.add_argument('--num_beams', type=int, default=5)


    # Training configuration

    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_eps", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)

    parser.add_argument("--start_epoch", default=0, type=int)

    parser.add_argument("--pos_num", default = 25, type = int)
    # Parse the arguments.
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    return args


if __name__ == '__main__':
    args = parse_args()