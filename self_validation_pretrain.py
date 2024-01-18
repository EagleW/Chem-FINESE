from pretrain_dataset import get_loader
import json
from tqdm import tqdm
import random
import torch
import argparse
from transformers import  get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, AutoTokenizer, AutoModelForSeq2SeqLM
import wandb
import numpy as np
from torch.cuda.amp import autocast
from torch.optim import AdamW
from utils_ import parse_args, LossMeter
import os
import math
from collections import defaultdict
from time import time
from statistics import mean


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Trainer:
    def __init__(self, args, train_loader=None, valid_loader=None, test_loader=None, tokenizer=None, sampler=None, train=True):
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.model = AutoModelForSeq2SeqLM.from_pretrained("cogint/in-boxbart")


        if args.load is not None:
            ckpt_path = args.load
            self.load_checkpoint(ckpt_path)
            print('Load pretrained model')
        
        if torch.cuda.is_available():
            print(f'Model Launching at GPU ')
            self.model = self.model.cuda()
        
        if train:
            if self.args.fp16:
                print('Run in half precision')
                self.scaler = torch.cuda.amp.GradScaler()
        self.create_optimizer_and_scheduler()
        self.tokenizer = tokenizer
        
        self.device =  next(self.model.parameters()).device

    def create_optimizer_and_scheduler(self):
        
        self.model.model.shared.requires_grad = False
        self.optim = AdamW([p for p in self.model.parameters() if p.requires_grad],
                            lr=self.args.lr, eps=self.args.adam_eps, betas=(0.9, 0.98))
        num_training_steps = self.args.epochs * len(self.train_loader)
        self.lr_scheduler = self._create_lr_scheduler(num_training_steps)
        

    def _create_lr_scheduler(self, num_training_steps):
        self.args.warmup = min(self.args.warmup, num_training_steps // 10)
        if self.args.lr_scheduler == 'linear':
            return get_linear_schedule_with_warmup(optimizer=self.optim,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        elif self.args.lr_scheduler == 'cosine':
            return get_cosine_schedule_with_warmup(optimizer=self.optim,
                                                   num_warmup_steps=self.args.warmup,
                                                   num_training_steps=num_training_steps)
        else:
            assert False, 'Unknown lr scheduler: {}'.format(self.args.scheduler)
        

    def load_checkpoint(self, ckpt_path):
        print("Load model from %s" % ckpt_path)
        pretrained_dict = torch.load("%s.pth" % ckpt_path)

        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=False)

    def save(self, name):
        if not os.path.isdir(self.args.output):
            os.makedirs(self.args.output, exist_ok=True)
        torch.save(self.model.state_dict(),
            os.path.join(self.args.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict) 

    def train(self):
        loss_meter = LossMeter()
        best_score = 0
        best_epoch = 0
        
        for epoch in range(self.args.epochs):
            self.model.train()
            epoch_results = {
                'loss': 0.
            }

            pbar = tqdm(total=len(self.train_loader), ncols=150)
            for batch in self.train_loader:
                self.model.train()
                self.model.zero_grad(set_to_none=True)

                if self.args.fp16:
                    with autocast():
                        input_ids = batch['input_ids'].to(self.device)
                        lm_labels = batch["target_ids"].to(self.device)
                        attention_mask = batch["attention_masks"].to(self.device)

                        results = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=lm_labels,
                            return_dict=True
                        )
                        loss = results.loss
                        self.scaler.scale(loss).backward()
                else:
                    input_ids = batch['input_ids'].to(self.device)
                    lm_labels = batch["target_ids"].to(self.device)
                    attention_mask = batch["attention_masks"].to(self.device)

                    results = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=lm_labels,
                        return_dict=True
                    )
                    loss = results.loss
                    loss.backward()# Update Parameters
                if self.args.clip_grad_norm > 0:
                    if self.args.fp16:

                        self.scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)

                        self.scaler.step(self.optim)

                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.clip_grad_norm)
                        self.optim.step()
                else:

                    if self.args.fp16:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                    else:
                        self.optim.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()
                for param in self.model.parameters():
                    param.grad = None

                l = results.loss.detach().item()
                epoch_results['loss'] += l
                lr=self.optim.param_groups[0]["lr"] 

                loss_meter.update(l)
                desc_str = f'Epoch {epoch} | LR {lr:.10f}'
                desc_str += f' | Loss {loss_meter.val:6f}'

                pbar.set_description(desc_str)
                pbar.update(1)

            pbar.close()
            score_dict = self.predict(False)

            len_train_loader = len(self.train_loader)
            epoch_results['loss'] /= len_train_loader

            if score_dict > best_score or epoch == 0:
                best_score = score_dict
                self.save("BEST")
                update_epoch  = epoch
                best_epoch = epoch

            wandb_log_dict = {}
            wandb_log_dict['Train/Loss'] = epoch_results['loss'] 
            wandb_log_dict['Valid/BLEU'] = score_dict

            log_str = ''
            log_str += "\nEpoch %d: Best Loss %0.2f\n" % (best_epoch, best_score)
            print("\nEpoch %d: Train loss %0.4f Valid Score  %0.4f\n" % (epoch, wandb_log_dict['Train/Loss'], score_dict))
            print(log_str)
            if epoch - update_epoch > self.args.patient:
                break        

        torch.cuda.empty_cache()
        best_path = os.path.join(self.args.output, 'BEST')
        self.load(best_path)
        self.predict()

    def predict(self, test=True):
        # start_time = time()
        self.model.eval()
        if test:
            loader = self.test_loader
        else:
            loader = self.valid_loader


        quesid2ans = {}
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(self.device)
            batch_size = input_ids.size(0)
            output = self.model.generate(
                input_ids = input_ids, 
                num_beams=self.args.beam_size, 
                max_length=250,
                return_dict_in_generate=True)

            pred_ans = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
            tgts =  batch["targets"]
            cids = batch['sent_ids']
            sources =  batch["sources"]
            for qid, src, ans, tgt in zip(cids, sources, pred_ans, tgts):
                quesid2ans[qid] = (ans, src, tgt)

        topk_score = loader.evaluator.evaluate(quesid2ans)

           


        if test:

            os.makedirs(self.args.output, exist_ok=True)

            dump_path = '{}/eval_results.json'.format(self.args.output)

            loader.evaluator.dump_result(quesid2ans, dump_path)
            metrics = {'bleu':topk_score['bleu'],'rouge':topk_score['rouge']}
            with open('{}/metrics.json'.format(self.args.output), 'w', encoding='utf-8') as writer:
                writer.write('metrics: {}\n'.format(json.dumps(metrics)))
        return topk_score['score']

args = parse_args()


tokenizer = AutoTokenizer.from_pretrained("cogint/in-boxbart")

print('Building train loader')
if args.test_only:
    test_loader, sampler, test_dataset = get_loader(
        args,
        split='test', 
        mode='test', 
        tokenizer=tokenizer,
        batch_size=args.valid_batch_size,
        workers=args.workers,
        topk=args.valid_topk,
    )
    train_loader = None

    trainer = Trainer(args, train_loader, test_loader, tokenizer, sampler, train=False)
    trainer.predict()
else:
    train_loader, sampler, train_dataset = get_loader(
        args,
        split='train', 
        mode='train', 
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        workers=args.workers,
        topk=args.train_topk,
    )
    valid_loader, sampler, valid_dataset = get_loader(
        args,
        split='valid', 
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        workers=args.workers,
        topk=args.train_topk,
    )
    test_loader, sampler, test_dataset = get_loader(
        args,
        split='test', 
        mode='test', 
        tokenizer=tokenizer,
        batch_size=args.valid_batch_size,
        workers=args.workers,
        topk=args.valid_topk,
    )
    trainer = Trainer(args, train_loader, valid_loader, test_loader, tokenizer, sampler, train=False)
    trainer.train()

