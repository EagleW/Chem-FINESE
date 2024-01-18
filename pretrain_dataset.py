from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import json
import torch
import random
from torch.utils.data.distributed import DistributedSampler
from bleu.bleu import Bleu
import random
import itertools
from bisect import bisect
from rouge.rouge import Rouge

class ChemDataset(Dataset):
    def __init__(self, split='train', dataset='chemner_filter_cleaned_data', topk=-1, tokenizer=None):
        self.topk = topk
        self.tokenizer = tokenizer
        fpath = '../' + dataset + '/'

        fname = fpath + '%s.json' % split

        self.data = self.loadData(fname)

    def __len__(self):
        return len(self.data)
    
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
                for entity in sorted(tmp_entities, key=lambda d: d['start']):
                    type_ = entity["type"].replace('_', ' ')
                    text = entity["text"]
                    start = entity["start"]
                    entities.append(text + ' <%s>' % type_)
                    new_entity.append((text,type_,start))
                output = ', '.join(entities)

                input_ids_i = self.tokenizer(input_, truncation=True, max_length=512).input_ids
                target_ids_i = self.tokenizer(output, truncation=True, max_length=512).input_ids

                out_dict = {
                    'source': {'input':input_, 'entities':new_entity, 'sent_tokens':sent_tokens},
                    'sent_id': sentid,
                    'target': input_,

                    'target_length': len(input_ids_i),
                    'target_ids': torch.LongTensor(input_ids_i),
                    'input_ids': torch.LongTensor(target_ids_i),
                    'input_length': len(target_ids_i),
                }


                data.append(out_dict)
                if len(data) > self.topk and self.topk != -1:
                    return data
        return data

    def __getitem__(self, idx):
        datum = self.data[idx]

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


        T_L = max(entry['target_length'] for entry in batch)
        target_ids = torch.ones(B, T_L, dtype=torch.long) * self.tokenizer.pad_token_id


        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']
            attention_masks[i, :entry['input_length']] = 1

            sources.append(entry['source'])
            sent_ids.append(entry['sent_id'])
            targets.append(entry['target']) 
        
        batch_entry['input_ids'] = input_ids
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['attention_masks'] = attention_masks
        batch_entry['target_ids'] = target_ids


        batch_entry['targets'] = targets
        batch_entry['sources'] = sources
        batch_entry['sent_ids'] = sent_ids
        return batch_entry

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
        

def get_loader(args, split='train', mode='train', tokenizer=None,
               batch_size=32, workers=4, topk=-1):

    sampler = None
    dataset = ChemDataset(
                split,
                dataset=args.dataset_dir,
                topk=topk,
                tokenizer=tokenizer)
                
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