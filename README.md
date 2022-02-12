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

## Citation
If you use this dataset or the codes, please refer to the following paper:
```
  @inproceedings{paul2022lesicin,
   author = {Paul, Shounak and Goyal, Pawan and Ghosh, Saptarshi},
   title = {{LeSICiN: A Heterogeneous Graph-based Approach for Automatic Legal Statute Identification from Indian Legal Documents}},
   booktitle = {{Proceedings of the 36th AAAI Conference on Artificial Intelligence (AAAI)}},
   year = {2022}
  }
 ```

## Repo Organization
```
- model
  - basicmodules.py - Contains basic building blocks -- LSTM and Attention Networks
  - submodules.py - Contains Text and Graph Encoders and Matching Network 
  - model.py - Contains main module LeSICiN
- data_helper.py - Helper codes for constructing Dataset and batching logic
- helper.py - Helper codes for creating vocabularies, label weights, training loop, metrics, etc.
- run.py - Script for running training and evaluation, and/or testing, or inference
```
## Data 
Datasets are in the form of .jsonl files, with one instance (dict) per line, having the following keys
```
id [string]: id/name of the fact instance / section instance
text [list[string]]: text in the form of list of sentences (each sentence is a string)
labels [list[string] / null]: gold-standard labels (not needed for inference) 
```
Apart from this, we need a type_map, which will be a dict mapping node ids to their type (Act / Chapter / Topic / Section / Fact). We also need two files, storing the edges of the label tree and the Fact-Section citation network in the following format
```
(src node id, relationship name, trg node id)
```
Finally, we also need the metapath schemas for each node type. Each individual schema is a list of the edges that make up the metapath. An edge is described by the tuple
```
(src node type, relationship name, trg node type)
```
You can find all these files at: https://doi.org/10.5281/zenodo.6053791

## Configs
To make it easy to configure experiments on the go, we make use of two config files stored in configs/ folder.

*data_path.json* - Specifies the full file paths for loading data, models, etc.
```
sec_src [string]: path to sec source data file
train_src [string/null]: path to train source data file (null if not running train-dev)
dev_src [string/null]: path to dev source data file (null if not running train-dev)
test_src [string/null]: path to test source data file (null if not running test)
infer_src [string/null]: path to infer source data file (null if not running infer) 

sec_cache [string]: path to sec cached data file
train_cache [string/null]: path to train cached data file (null if not running train-dev)
dev_cache [string/null]: path to dev cached data file (null if not running train-dev)
test_cache [string/null]: path to test cached data file (null if not running test)
infer_cache [string/null]: path to infer cached data file (null if not running infer) 

s2v_path [string/null]: path to pretrained sent2vec file (null if you do not want to sent vectorize)

type_map [string]: path to file that maps id of each node to its type
label_tree [string]: path to file that stores edges of the label tree
citation_network [string]: path to file that stores edges of the Fact-Section citation net
schemas [string]: path to file that stores the schemas

model_load [string/null]: checkpointed model to load from (null to train from scratch)
metrics_load [string/null]: saved validation metrics to act as benchmark (null to train from scratch)

model_dump [string]: path to file where trained model will be saved
dev_metrics_dump [string]: path to file where best validation metrics will be saved
test_metrics_dump [string]: path to file where test metrics will be saved
infer_trg [string]: path to file where inference predictions will be saved
```

*hyperparams.json* - Controls the model and experiment hyperparameters and few other settings, like the seed.
```
seed [int]: universal seed for random, numpy and torch
do_train_dev [bool]: true if running train-dev
do_test [bool]: true if running test
do_infer [bool]: true if running infer
vocab_limit [int/null]: maximum vocabulary size [null if using sent vectorization]
vocab_thresh [int/null]: minimum frequency for a word to be considered in vocabulary [null if using sent vectorization]
weight_scheme {"tws", "vws"}: choose between Threshold-based weighting scheme and Vanilla Weighting Scheme as discussed in the paper
tws_thresh [float/null]: TWS threshold as discussed in the paper (null if using VWS)

train_bs [int]: batch size (no. of facts) for training
dev_bs [int]: batch size (no. of facts) for validation
test_bs [int]: batch size (no. of facts) for testing
infer_bs [int]: batch size (no. of facts) for inference
max_segments [int]: maximum no. of sentences per document (fact or section)
max_segment_size [int/null]: maximum no. of words per sentence (null if using sent vectorization)
num_mpath_samples [int]: no. of metapath instances to sample per metapath schema

hidden_size [int]: hidden dimension for all intermediate layers (if using sent vectorization, make sure this is equal to the dimension of the sent2vec embeddings)

opt_lr [float]: learning rate for optimizer
opt_wt_decay [float]: optimizer weight decay
sch_factor [float]: factor for the ReduceLROnPlateau scheduler
sch_patience [int]: patience for the ReduceLROnPlateau scheduler
num_epoch [int]: no. of training epochs

pthresh [float]: prediction threshold to be used by model
thetas [tuple[float]]: thetas in the order (attr, struct, align)
lambdas [tuple[float]]: lambdas in the order (attr, align)
dropout [float]: dropout factor for model layers
```

## Running the Script
All kinds of operations (train/dev/test/infer) can be performed by the "run.py" code, by appropriately configuring its settings. See the section above to understand the different settings. You can run the script using:
```
python run.py
```
## Outputs
In case of train / dev / test, a metrics object is saved in the path specified in dev / test metrics dump key in data path config, which contains the following scores:
```
- macro
  - precision
  - recall
  - f1
- micro
  - precision
  - recall
  - f1
- jaccard
```
If training is performed, the model state corresponding to the best dev loss is also saved in the path specified in model dump key in data path config.
During inference, instead of metrics, a jsonl file is saved in the path specified in infer trg key in data path config. Each line in the jsonl file is a dict with the following keys:
```
id [string]: id of the fact instance
predictions [list[string]]: model predictions for this instance
```
