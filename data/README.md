### Data

`annotation` folder contains annotation guidelines and fine-grained entity ontology.
`CHEMET` folder contains full CHEMET dataset and its few-shot subsets. Each folder contains four files: `train.json`, `valid.json`, `test.json`, and `types.json`.
`ChemNER+` folder contains full ChemNER+ dataset and its few-shot subsets. Each folder contains four files: `train.json`, `valid.json`, `test.json`, and `types.json`.
`train.json`, `valid.json`, `test.json` are used for training, validation, and testing respectively. Each file contains multiple lines. Each line represent an instance. The schema for each instance is listed below:
```

{
    "coupling":        #   sentence id
    "sent_tokens":     #   tokens in the sentence
    "entities":        #   ground truth entities in the sentence, which is a list containing entity type, text, start position, end position
    "f1":              #   semantic similarity between entity list and input
    }
```