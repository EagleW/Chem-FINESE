from dataset_nu_cl import get_loader
import torch.backends.cudnn as cudnn
import os
from pathlib import Path
from transformers import BartTokenizer, BartForConditionalGeneration
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
import numpy as np
from param import parse_args
from utils_ import set_global_logging_level, LossMeter
import wandb
from torch.cuda.amp import autocast
from trainer_base import TrainerBase
import json
import math
from cycle_model import CycleV1_cl
from nereval import Entity, precision, recall, f1, count_correct
from nltk import word_tokenize

class Trainer(TrainerBase):
    def __init__(self, args, train_loader=None, valid_loader=None, test_loader=None, tokenizer=None, train=True):
        super().__init__(
            args,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            tokenizer=tokenizer,)

        config = self.create_config()
        self.model = CycleV1_cl(config, args.gpu)
        fpath = '../' + args.dataset_dir + '/types.json'
        file_object = open(fpath, 'r')
        self.types = [x.replace('_', ' ') for x in json.load(file_object)]
        self.t2u = {}
        for x in self.types:
            self.t2u[x.lower()] = x
        
        # load self_validation
        self.load_v(args.load_cons)


        if args.load is not None:
            ckpt_path = args.load
            self.load_checkpoint(ckpt_path)
            print('Load pretrained model')

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        from time import time
        start = time()
        self.model = self.model.to(args.gpu)

        # Optimizer
        if train:
            self.model.model2.requires_grad_(False)
            if self.args.fp16:
                print('Run in half precision')
                self.scaler = torch.cuda.amp.GradScaler()
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

        print(f'It took {time() - start:.1f}s')
        self.device =  args.gpu

        if args.wandb:
            wandb.watch(self.model)

    def save(self, name):
        if not os.path.isdir(self.args.output):
            os.makedirs(self.args.output, exist_ok=True)
        torch.save(self.model.state_dict(),
                   os.path.join(self.args.output, "%s.pth" % name))
        
    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)

    def load_v(self, path):
        print("Load self-validation model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.model2.load_state_dict(state_dict)

    def train(self):
        loss_meters = [LossMeter() for _ in range(7)]
        LOSSES_NAME = [
            'loss',
            'sup_loss_f',
            'rec_loss_f',
            'cl_loss_decoder',
            ]
        best_score = 0
        best_epoch = 0
        
        for epoch in range(self.args.epochs):
            self.model.train()
            epoch_results = {
                'loss': 0.,
                'sup_loss_f': 0.,
                'rec_loss_f': 0.,
                'cl_loss_decoder': 0.,
            }

            pbar = tqdm(total=len(self.train_loader), ncols=150)
            for batch in self.train_loader:
                self.model.train()
                self.model.zero_grad(set_to_none=True)

                if self.args.fp16:
                    with autocast():
                        results = self.model(
                            batch=batch)
                        loss = results['loss']
                        self.scaler.scale(loss).backward()
                else:
                    results = self.model(
                            batch=batch)
                    loss = results['loss']
                    loss.backward()
                
                # Update Parameters
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

                l = loss.item()
                for k, v in results.items():
                    epoch_results[k] += v.item()
                lr=self.optim.param_groups[0]["lr"] 

                desc_str = f'Epoch {epoch} | LR {lr:.10f}'

                for i, (loss_name, loss_meter) in enumerate(zip(LOSSES_NAME, loss_meters)):
                            
                    loss_meter.update(results[f'{loss_name}'].item())
                    desc_str += f' {loss_name} {loss_meter.val:.2f}'

                pbar.set_description(desc_str)
                pbar.update(1)

            pbar.close()
            score_dict = self.predict_t(False)

            wandb_log_dict = {}
            len_train_loader = len(self.train_loader)
            for loss_name in LOSSES_NAME:
                wandb_log_dict['Train/%s' % loss_name] = epoch_results[loss_name] / len_train_loader

            if score_dict['f1'] > best_score or epoch == 0:
                best_score = score_dict['f1']
                self.save("BEST")
                update_epoch  = epoch
                best_epoch = epoch

            wandb_log_dict['Train/lr'] = lr
            wandb_log_dict['Valid/p'] = score_dict['p']
            wandb_log_dict['Valid/r'] = score_dict['r']
            wandb_log_dict['Valid/f1'] = score_dict['f1']
            wandb_log_dict['Train/best_epoch'] = best_epoch
            if self.args.wandb:
                wandb.log(wandb_log_dict, step=epoch)

            log_str = ''
            log_str += "\nEpoch %d: Best Loss %0.2f\n" % (best_epoch, best_score)
            print("\nEpoch %d: Train loss %0.4f  Valid F1  %0.4f\n" % (epoch, wandb_log_dict['Train/loss'], score_dict['f1']))
            print(log_str)
            if epoch - update_epoch > self.args.patient:
                break        

        torch.cuda.empty_cache()
        best_path = os.path.join(self.args.output, 'BEST')
        self.load(best_path)
        self.predict_t()

    @torch.no_grad()
    def predict_t(self, test=True):
        # start_time = time()
        self.model.eval()
        if test:
            loader = self.test_loader
        else:
            loader = self.valid_loader

        # precisions = []
        # recalls = []
        # f1s = []
        y_trues = []
        y_preds = []

        results = []
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(self.device)
            batch_size = input_ids.size(0)
            output = self.model.model1.generate(
                input_ids = input_ids, 
                num_beams=self.args.num_beams, 
                return_dict_in_generate=True,
                max_length=256
                )

            predictions = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
            references =  batch["targets"]
            ques_ids = batch['sent_ids']
            sources =  batch["sources"]


            for source, pred, ref, sentid in zip(sources, predictions, references, ques_ids):
                input_ = source['input']
                entities = source['entities']
                sent_tokens = source['sent_tokens']
                pred_results = []

                y_true = []
                y_pred = []
                n_pred = pred[:].strip()
                for text,type_,start in entities:
                    y_true.append(Entity(text, type_, start))
                
                used = []

                for chunk in pred.split('>, '):
                    if '>' in chunk:
                        chunk = chunk.split('>')[0]
                    if len(chunk.split(' <')) == 2:
                        text, type_ = chunk.split(' <')
                        type_ = type_.strip()
                        if text in input_:
                            if type_.lower() in self.t2u:
                                start = self.get_start(text, sent_tokens, used)
                                if start is not None:
                                    used.append(text)
                                    pred_results.append((text, self.t2u[type_.lower()], start))
                                    y_pred.append(Entity(text, self.t2u[type_.lower()], start))
                        else:
                            tokenized_t = word_tokenize(text)
                            n_text = ' '.join(tokenized_t)
                            if n_text in input_:
                                # print(n_text)
                                n_pred = n_pred.replace(text, n_text)
                                if type_.lower() in self.t2u:
                                    start = self.get_start(n_text, sent_tokens, used)
                                    if start is not None:
                                        used.append(n_text)
                                        pred_results.append((n_text, self.t2u[type_.lower()], start))
                                        y_pred.append(Entity(n_text, self.t2u[type_.lower()], start))
                    elif len(chunk.split(' <')) > 2:
                        text, type_ = chunk.split(' <')[-2:]
                        type_ = type_.strip()
                        if text in input_:
                            if type_.lower() in self.t2u:
                                start = self.get_start(text, sent_tokens, used)
                                if start is not None:
                                    used.append(text)
                                    pred_results.append((text, self.t2u[type_.lower()], start))
                                    y_pred.append(Entity(text, self.t2u[type_.lower()], start))
                        else:
                            tokenized_t = word_tokenize(text)
                            n_text = ' '.join(tokenized_t)
                            if n_text in input_:
                                # print(n_text)
                                n_pred = n_pred.replace(text, n_text)
                                if type_.lower() in self.t2u:
                                    start = self.get_start(n_text, sent_tokens, used)
                                    if start is not None:
                                        used.append(n_text)
                                        pred_results.append((n_text, self.t2u[type_.lower()], start))
                                        y_pred.append(Entity(n_text, self.t2u[type_.lower()], start))


                tmp = {
                    'sent_id': sentid,
                    'input': source['input'],
                    'prediction': pred_results,
                    'ground_truth': entities,
                    'output': n_pred

                }
                results.append(tmp)
                y_trues.append(y_true)
                y_preds.append(y_pred)
        print(results[-1]['input'])
        print(results[-1]['output'])
        print(results[-1]['ground_truth'])
        print(results[-1]['prediction'])
        p_, r_, f1_ = self.eval_f1(y_trues, y_preds)
        score_dict = {
            'p': p_,
            'r': r_,
            'f1': f1_,
        }
        if test:

            os.makedirs(self.args.output, exist_ok=True)

            with open('{}/eval_results.json'.format(self.args.output), 'w', encoding='utf-8') as writer:
                writer.write(json.dumps(results, ensure_ascii=False, indent=4))
            metrics = {'p':p_,'r':r_, 'f1': f1_}

            with open('{}/metrics.json'.format(self.args.output), 'w', encoding='utf-8') as writer:
                writer.write(json.dumps(metrics))
        return score_dict
    
    def find_sub_list(self, sl,l):
        results=[]
        sll=len(sl)
        for ind in (i for i,e in enumerate(l) if e==sl[0]):
            if l[ind:ind+sll]==sl:
                results.append(ind)

        return results
    
    def get_start(self, text, sent_tokens, used):
        txt_tmp = text.split(' ')
        offset = used.count(text)

        results = self.find_sub_list(txt_tmp, sent_tokens)
        if len(results) > offset:
            return results[offset]
        else:
            return None 

    def eval_f1(self, y_true, y_pred):

        correct, actual, possible = 0, 0, 0

        for x, y in zip(y_true, y_pred):
            correct += sum(count_correct(x, y))
            # multiply by two to account for both type and text
            possible += len(x) * 2
            actual += len(y) * 2
        
        p = precision(correct, actual)
        r = recall(correct, possible)

        return p,r,f1(p, r)



def main_worker(gpu, args):
    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')
    tokenizer = BartTokenizer.from_pretrained("cogint/in-boxbart")
    


    if args.test_only:
    
        print(f'Building submit test loader at GPU {gpu}')

        split = f'submit_{gpu}'
        print('Loading', split)

        test_loader, _, test_dataset = get_loader(
            args,
            split='test', 
            mode='val', 
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            workers=4,
            topk=args.valid_topk,
        )
        train_loader = None
        valid_loader = None

        trainer = Trainer(args, train_loader, valid_loader, test_loader, tokenizer, train=False)
        trainer.predict_t(True)

    else:

        print(f'Building train loader at GPU {gpu}')
        train_loader, _, train_dataset = get_loader(
            args,
        # split='new_test', 
            split='train', 
            mode='train', 
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            workers=4,
            topk=args.train_topk,
        )

        if args.valid_batch_size is not None:
            valid_batch_size = args.valid_batch_size
        else:
            valid_batch_size = args.batch_size
        print(f'Building val loader at GPU {gpu}')
        valid_loader, _, valid_dataset = get_loader(
            args,
            split='valid', 
            mode='valid', 
            tokenizer=tokenizer,
            batch_size=valid_batch_size,
            workers=4,
            topk=args.valid_topk,
        )

        print(f'Building test loader at GPU {gpu}')
        test_loader, _, test_dataset = get_loader(
            args,
            split='test', 
            mode='test', 
            tokenizer=tokenizer,
            batch_size=valid_batch_size,
            workers=4,
            topk=args.valid_topk,
        )

        trainer = Trainer(args, train_loader, valid_loader, test_loader, tokenizer, train=True)

        trainer.train()
if __name__ == "__main__":
    cudnn.benchmark = True
    args = parse_args()
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node

    project_name = "ChemicalIE"


    comments = []
    if args.load is not None:
        ckpt_str = "_".join(args.load.split('/')[-3:])
        comments.append(ckpt_str)
    comment = '_'.join(comments)


    if args.wandb:
        wandb.init(project=project_name,  resume="allow")
        wandb.config.update(args)
        config = wandb.config
    else:
        config=args
    main_worker(0, config)