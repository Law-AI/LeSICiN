# LeSICiN
Dataset and codes for the paper "LeSICiN: A Heterogeneous Graph-based Approach for Automatic Legal Statute Identification from Indian Legal Documents", accepted and to be published at AAAI 2022.

## About
The task of **Legal Statute Identification (LSI)** aims to identify the legal statutes that are relevant to a given description of facts or evidence of a legal case.
Existing methods only utilize the textual content of facts and legal articles to guide such a task. However, the citation network among case documents and legal statutes is a rich source of additional information, which is not considered by existing models. 
In this work, we take the first step towards utilising both the text and the legal citation network for the LSI task.
We curate a large novel dataset for this task, including facts of cases from several major Indian Courts of Law, and statutes from the Indian Penal Code (IPC). 
Modeling the statutes and training documents as a heterogeneous graph, our proposed model **LeSICiN** can learn rich textual and graphical features, and can also tune itself to correlate these features. 
Thereafter, the model can be used to inductively predict links between test documents (new nodes whose graphical features are not available to the model) and statutes (existing nodes). 
Extensive experiments on the dataset show that our model comfortably outperforms several state-of-the-art baselines, by exploiting the graphical structure along with textual features.

## Repo Organization
```
- model
  - basicmodules.py - Contains basic building blocks -- LSTM and Attention Networks
  - submodules.py - Contains Text and Graph Encoders and Matching Network 
  - model.py - Contains main module LeSICiN
- data_helper.py - Helper codes for constructing Dataset and batching logic
- helper.py - Helper codes for creating vocabularies, label weights, training loop, metrics, etc.
- run.py - Script for running training and evaluation, and eventual testing
```

## Configs
To make it easy to configure experiments on the go, we make use of two config files stored in configs/ folder.

*data_path.json* - Specifies the full file paths for loading data, models, etc.
```
train_src [string]: path to train source data file
dev_src [string]: path to dev source data file
test_src [string]: path to test source data file
sec_src [string]: path to sec source data file

train_cache [string]: path to train cached data file
dev_cache [string]: path to dev cached data file
test_cache [string]: path to test cached data file
sec_cache [string]: path to sec cached data file

s2v_path [string/null]: path to pretrained sent2vec file (null if you do not want to sent vectorize)

type_map [string]: path to file that maps id of each node to its type
label_tree [string]: path to file that stores edges of the label tree
citation_network [string]: path to file that stores edges of the Fact-Section citation net
schemas [string]: path to file that stores the schemas

model_load [string/null]: checkpointed model to load from (null to train from scratch)
metrics_load [string/null]: saved validation metrics to act as benchmark (null to train from scratch)

model_dump [string]: path to file where trained model will be saved
metrics_dump [string]: path to file where best validation metrics will be saved
```

*hyperparams.json* - Controls the model and experiment hyperparameters and few other settings, like the seed.

## Training
```
python run.py
```
### This repo will be thoroughly updated with richer comments and detailed usage instructions soon ###
