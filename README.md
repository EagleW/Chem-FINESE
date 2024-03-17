# Chem-FINESE: Validating Fine-Grained Few-shot Entity Extraction through Text Reconstruction

[Chem-FINESE: Validating Fine-Grained Few-shot Entity Extraction through Text Reconstruction](https://arxiv.org/pdf/2401.10189.pdf)

Accepted by Findings of the Association for Computational Linguistics: EACL 2024

Table of Contents

=================

* [Overview](#overview)
  
* [Requirements](#requirements)
  
* [Quickstart](#quickstart)
  
* [Citation](#citation)

## Overview


<p align="center">
  <img src="https://eaglew.github.io/images/chemical_finese.png?raw=true" alt="Photo" style="width: 50%;"/>
</p>

## Requirements

### Environment:

* Python 3.10.12
 
* Ubuntu 22.04

### Setup Instructions:

To set up the environment for this repository, please follow the steps below:

Step 1: Create a Python environment (optional)
If you wish to use a specific Python environment, you can create one using the following:

```bash
conda create -n pyt1.12 python=3.10.12
```

Step 2: Install PyTorch with CUDA (optional)
If you want to use PyTorch with CUDA support, you can install it using the following:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Step 3: Install Python dependencies
To install the required Python dependencies, run the following command:

```bash
pip install -r requirements.txt
```

Step 4: Download NLTK data

```bash
python -m nltk.downloader punkt
```

### Data Description

This repositiory contains two chemical few-shot fine-grained entity extraction dataset based on ChemNER and CHEMET.
We choose the values $6, 9, 12, 15, 18$ as the potential maximum entity mentions for k-shot for both datasets. 
`annotation` folder contains annotation guidelines and fine-grained entity ontology.
`CHEMET` folder contains full CHEMET dataset and its few-shot subsets. Each folder contains four files: `train.json`, `valid.json`, `test.json`, and `types.json`.
`ChemNER+` folder contains full ChemNER+ dataset and its few-shot subsets. Each folder contains four files: `train.json`, `valid.json`, `test.json`, and `types.json`.
`train.json`, `valid.json`, `test.json` are used for training, validation, and testing respectively. Each file contains multiple lines. Each line represent an instance. The schema for each instance is listed below:

```python

{
    "coupling":        #   sentence id
    "sent_tokens":     #   tokens in the sentence
    "entities":        #   ground truth entities in the sentence, which is a list containing entity type, text, start position, end position
    "f1":              #   semantic similarity between entity list and input
    }
```

## Quickstart

Modify `file` path under `pretrain.sh` and `finetune_cl.sh`.

### Finetuning

You can fisrt pretrain your self-validation model by running `pretrain.sh` in this folder. 

```bash
bash pretrain.sh 
```

You can then finetune your model by running `finetune_cl.sh` in this folder. 

```bash
bash finetune_cl.sh 
```


### Test

You can then test your model by running `test_cl.sh` in this folder. 

```bash
bash test_cl.sh 
```


## Citation

```bib
@inproceedings{wang-etal-2024-chem,
    title = "Chem-{FINESE}: Validating Fine-Grained Few-shot Entity Extraction through Text Reconstruction",
    author = "Wang, Qingyun  and
      Zhang, Zixuan  and
      Li, Hongxiang  and
      Liu, Xuan  and
      Han, Jiawei  and
      Zhao, Huimin  and
      Ji, Heng",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2024",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-eacl.1",
    pages = "1--16",
    abstract = "Fine-grained few-shot entity extraction in the chemical domain faces two unique challenges. First, compared with entity extraction tasks in the general domain, sentences from chemical papers usually contain more entities. Moreover, entity extraction models usually have difficulty extracting entities of long-tailed types. In this paper, we propose Chem-FINESE, a novel sequence-to-sequence (seq2seq) based few-shot entity extraction approach, to address these two challenges. Our Chem-FINESE has two components: a seq2seq entity extractor to extract named entities from the input sentence and a seq2seq self-validation module to reconstruct the original input sentence from extracted entities. Inspired by the fact that a good entity extraction system needs to extract entities faithfully, our new self-validation module leverages entity extraction results to reconstruct the original input sentence. Besides, we design a new contrastive loss to reduce excessive copying during the extraction process. Finally, we release ChemNER+, a new fine-grained chemical entity extraction dataset that is annotated by domain experts with the ChemNER schema. Experiments in few-shot settings with both ChemNER+ and CHEMET datasets show that our newly proposed framework has contributed up to 8.26{\%} and 6.84{\%} absolute F1-score gains respectively.",
}
```
