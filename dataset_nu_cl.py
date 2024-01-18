
from torch.utils.data import DataLoader, Dataset
import json
import os
from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy
import statistics
from bleu.bleu import Bleu
from rouge.rouge import Rouge
import random
import itertools
from bisect import bisect
from torch.utils.data.distributed import DistributedSampler


class ChemDataset(Dataset):
    def __init__(self, split='train', dataset='chemner_filter_cleaned_data', topk=-1, tokenizer=None, neg_num=10):
        self.topk = topk
        self.tokenizer = tokenizer
        
        fpath = '../' + dataset + '/'

        fname = fpath + '%s.json' % split
        
        self.input_temp = "Instruction: Definition: In this task, you are given a small paragraph as input, and your task is to identify all the named chemical entities from the given input and also provide type of the each entity. Generate the output in this format: entity1 <type_of_entity1>, entity2 <type_of_entity2>. Instance: input: %s, output: ?"

        self.neg_num = neg_num
        if 'test' in split:
            self.data = self.loadData(fname)
        else:
            self.data = self.loadData1(fname)

    def __len__(self):
        return len(self.data)
    
    def create_negs(self, sent_tokens, type_dict, pos_set, pos_list, entities):
        pos_list.append(0)
        pos_list.append(len(sent_tokens))
        neg_sample = []

        t_d_keys = list(sorted(type_dict.keys()))
        for s_e_tuple in itertools.combinations(pos_list,2):
            if s_e_tuple in pos_set:
                continue

            s, e = sorted(s_e_tuple)
            idx_e = bisect(t_d_keys, e)
            idx_s = bisect(t_d_keys, s)
            if idx_e - idx_s > 3: 
                continue

            idx_ee = t_d_keys[idx_e - 1]
            type_e = type_dict[idx_ee]

            text = ' '.join(sent_tokens[s:e])
            
            negs_ = []
            negs_ = entities[:idx_s] + [text + ' <%s>' % type_e]
            negs_t = ', '.join(negs_)
            neg_id_ = self.tokenizer(negs_t, truncation=True, max_length=512).input_ids
            if len(neg_id_) > 256:
                negs_ = [text + ' <%s>' % type_e] +  entities[idx_e:]
            else:
                negs_ = entities[:idx_s] + [text + ' <%s>' % type_e] +  entities[idx_e:]
            
            negs_t = ', '.join(negs_)



            neg_id = self.tokenizer(negs_t, truncation=True, max_length=256).input_ids
            neg_length = len(neg_id)
            neg_sample.append((torch.LongTensor(neg_id), neg_length))

        return neg_sample

        

    def loadData(self, filename):
        data = []
        with open(filename, 'r') as f:
            for line in tqdm(f):
                cur_data = json.loads(line)
                sentid = cur_data["sentid"]
                tmp_entities = cur_data["entities"]
                sent_tokens = cur_data["sent_tokens"]
                input_ = ' '.join(sent_tokens)
                entities = []
                new_entity = []

                # old_e = []
                type_dict = {}
                pos_set = set()
                pos_list = []

                for entity in sorted(tmp_entities, key=lambda d: d['start']):
                    # type_ = entity["type"]
                    type_ = entity["type"].replace('_', ' ')
                    text = entity["text"]
                    start = entity["start"]
                    end = entity["end"]
                    type_dict[end] = type_
                    pos_set.add((start,end))

                    pos_list.append(start)
                    pos_list.append(end)

                    # if text not in old_e:
                        # old_e.append(text)
                    entities.append(text + ' <%s>' % type_)
                    new_entity.append((text,type_,start))
                pos_set.add((0, min(pos_list)))
                pos_set.add((max(pos_list), len(sent_tokens)))
                pos_set.add((0, len(sent_tokens)))

                neg_sample = self.create_negs(sent_tokens, type_dict, pos_set, pos_list, entities)

                output = ', '.join(entities)
                input_1 = self.input_temp % input_

                source_id = self.tokenizer(input_1, truncation=True, max_length=512).input_ids

                target_id = self.tokenizer(output, truncation=True, max_length=256).input_ids

                input_ids_i = self.tokenizer(input_, truncation=True, max_length=512).input_ids
                target_ids_i = self.tokenizer(output, truncation=True, max_length=512).input_ids

                out_dict = {
                    'source': {'input':input_, 'entities':new_entity, 'sent_tokens':sent_tokens},
                    'sent_id': sentid,
                    'target': output,
                    'input_length': len(source_id),
                    'input_ids': torch.LongTensor(source_id),
                    'target_ids': torch.LongTensor(target_id),
                    'target_length': len(target_id),

                    'neg_sample':neg_sample,

                    'input_i_length': len(input_ids_i),
                    'input_ids_i': torch.LongTensor(input_ids_i),
                    'target_ids_i': torch.LongTensor(target_ids_i),
                    'target_i_length': len(target_ids_i),
                }


                data.append(out_dict)
                if len(data) > self.topk and self.topk != -1:
                    return data
        return data

        

    def loadData1(self, filename):
        data = []
        with open(filename, 'r') as f:
            for line in tqdm(f):
                cur_data = json.loads(line)
                sentid = cur_data["sentid"]
                tmp_entities = cur_data["entities"]
                sent_tokens = cur_data["sent_tokens"]
                input_ = ' '.join(sent_tokens)
                tmp_ids = self.tokenizer(input_).input_ids
                l_sent = len(sent_tokens)
                if len(tmp_ids) > 450:
                    input_ = self.tokenizer.decode(tmp_ids[:450], skip_special_tokens=True) 
                    l_sent = len(input_.split())
                entities = []
                new_entity = []

                # old_e = []
                type_dict = {}
                pos_set = set()
                pos_list = []

                for entity in sorted(tmp_entities, key=lambda d: d['start']):
                    # type_ = entity["type"]
                    type_ = entity["type"].replace('_', ' ')
                    text = entity["text"]
                    start = entity["start"]
                    end = entity["end"]
                    if end > l_sent:
                        break 

                    type_dict[end] = type_

                    pos_set.add((start,end))

                    pos_list.append(start)
                    pos_list.append(end)

                    # if text not in old_e:
                        # old_e.append(text)
                    entities.append(text + ' <%s>' % type_)
                    new_entity.append((text,type_,start))
                if len(new_entity) == 0:
                    continue
                pos_set.add((0, min(pos_list)))
                pos_set.add((max(pos_list), len(sent_tokens)))
                pos_set.add((0, len(sent_tokens)))

                neg_sample = self.create_negs(sent_tokens, type_dict, pos_set, pos_list, entities)

                output = ', '.join(entities)
                input_1 = self.input_temp % input_

                source_id = self.tokenizer(input_1, truncation=True, max_length=512).input_ids

                target_id = self.tokenizer(output, truncation=True, max_length=256).input_ids

                input_ids_i = self.tokenizer(input_, truncation=True, max_length=512).input_ids
                target_ids_i = self.tokenizer(output, truncation=True, max_length=512).input_ids

                out_dict = {
                    'source': {'input':input_, 'entities':new_entity, 'sent_tokens':sent_tokens},
                    'sent_id': sentid,
                    'target': output,
                    'input_length': len(source_id),
                    'input_ids': torch.LongTensor(source_id),
                    'target_ids': torch.LongTensor(target_id),
                    'target_length': len(target_id),

                    'neg_sample':neg_sample,

                    'input_i_length': len(input_ids_i),
                    'input_ids_i': torch.LongTensor(input_ids_i),
                    'target_ids_i': torch.LongTensor(target_ids_i),
                    'target_i_length': len(target_ids_i),
                }


                data.append(out_dict)
                if len(data) > self.topk and self.topk != -1:
                    return data
        return data

    def __getitem__(self, idx):
        datum = self.data[idx]

        tmp = random.sample(datum['neg_sample'], k= min(self.neg_num-1,len(datum['neg_sample'])))

        neg_lengths = []
        neg_ids = []

        cur_neg = len(tmp)

        while cur_neg < self.neg_num:
            n_tmp = random.sample(datum['neg_sample'], k= min(self.neg_num-cur_neg,len(datum['neg_sample'])))
            tmp.extend(n_tmp)
            cur_neg = len(tmp)

        for neg_id, neg_length in tmp:
            neg_lengths.append(neg_length)
            neg_ids.append(neg_id)

        datum['neg_length'] = neg_lengths
        datum['neg_ids'] = neg_ids
        return datum
        
    def collate_fn(self, batch):
        batch_entry = {}
        B = len(batch)
        targets = []
        sources= []
        sent_ids= []

        S_L = max(entry['input_length'] for entry in batch)
        input_ids = torch.ones(B, S_L, dtype=torch.long) * self.tokenizer.pad_token_id
        attention_masks = torch.zeros(B, S_L, dtype=torch.long)

        S_L_i = max(entry['input_i_length'] for entry in batch)
        input_ids_i = torch.ones(B, S_L_i, dtype=torch.long) * self.tokenizer.pad_token_id


        T_L = max(entry['target_length'] for entry in batch)
        target_ids = torch.ones(B, T_L, dtype=torch.long) * self.tokenizer.pad_token_id

        N_L = max(neg_length for entry in batch for neg_length in entry['neg_length'])
        neg_ids = torch.ones(B * self.neg_num, N_L, dtype=torch.long) * self.tokenizer.pad_token_id

        T_L_i = max(entry['target_i_length'] for entry in batch)
        target_ids_i = torch.ones(B, T_L_i, dtype=torch.long) * self.tokenizer.pad_token_id
        tgt_attention_masks = torch.zeros(B, T_L_i, dtype=torch.long)


        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']
            attention_masks[i, :entry['input_length']] = 1

            input_ids_i[i, :entry['input_i_length']] = entry['input_ids_i']
            target_ids_i[i, :entry['target_i_length']] = entry['target_ids_i']
            tgt_attention_masks[i, :entry['target_i_length']] = 1

            for j in range(self.neg_num):
                index = i  * self.neg_num + j
                neg_ids[index, :entry['neg_length'][j]] = entry['neg_ids'][j]

            sources.append(entry['source'])
            sent_ids.append(entry['sent_id'])
            targets.append(entry['target']) 
        
        batch_entry['input_ids'] = input_ids
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['attention_masks'] = attention_masks
        batch_entry['target_ids'] = target_ids

        batch_entry['neg_ids'] = neg_ids
        batch_entry['neg_num_total'] = self.neg_num

        batch_entry['target_ids_i'] = target_ids_i
        batch_entry['tgt_attention_masks'] = tgt_attention_masks
        word_mask = input_ids_i != self.tokenizer.pad_token_id
        input_ids_i[~word_mask] = -100
        batch_entry['input_ids_i'] = input_ids_i



        batch_entry['targets'] = targets
        batch_entry['sources'] = sources
        batch_entry['sent_ids'] = sent_ids

        return batch_entry


def get_loader(args, split='train', mode='train', tokenizer=None,
               batch_size=32, workers=4, topk=-1):

    sampler = None
    dataset = ChemDataset(
                split,
                dataset=args.dataset_dir,
                topk=topk,
                tokenizer=tokenizer,
                neg_num=args.neg_num
                )
                
    if mode == 'train':
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=(sampler is None),
            num_workers=workers, 
            pin_memory=True, 
            sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, 
            pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)


    loader.evaluator = Evaluator()
    return loader, sampler, dataset



class Evaluator:
    def __init__(self):
        self.scorers = [
            (Bleu(4),  ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L")
            ]

    

    def score(self, ref, hypo):
        final_scores = {}
        for scorer, method in self.scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score

        return final_scores
    def evaluate(self, quesid2ans):
        hypo = {}
        ref = {}
        i = 0
        for k in quesid2ans:
            ans, _, tgt = quesid2ans[k]
            hypo[i] = [ans]
            ref[i] = [tgt]
            i += 1

        score = self.score(ref, hypo)
        print(score)
        
        return {'score':2*score['ROUGE_L']*score['Bleu_4']/(score['Bleu_4']+ score['ROUGE_L']), 'bleu':score['Bleu_4'], 'rouge':score['ROUGE_L']}

    def dump_result(self, quesid2ans: dict, path):

        with open(path, 'w') as f:
            for k in quesid2ans:
                ans, src, tgt = quesid2ans[k]
                result = {'cid':k, 'src':src, 'pred':ans, 'ground': tgt}
                f.write(json.dumps(result) + '\n')