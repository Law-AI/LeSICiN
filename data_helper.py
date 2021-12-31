import torch
import numpy as np
import string
import copy
import multiprocessing as mp
from tqdm import tqdm
import json
import pickle as pkl

class LSIDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_file=None, data_list=None):
        super().__init__()
        
        self.annotated = False
        self.sent_vectorized = False
        
        if data_list is not None:
            self.dataset = copy.deepcopy(data_list)
            for instance in tqdm(self.dataset, desc="Loading data from list"):
                instance['text'] = instance['text']
                if 'labels' in instance:
                    self.annotated = True
                    instance['labels'] = np.array(instance['labels'])
        
        elif jsonl_file is not None:
            self.dataset = []
            with open(jsonl_file) as fr:
                for line in tqdm(fr, desc="Loading data from file"):
                    doc = json.loads(line)
                    text = np.array([sent for sent in doc['text']])
                    newdoc = {'id': doc['id'], 'text': text}
                    if 'labels' in doc:
                        self.annotated = True
                        labels = np.array(doc['labels'])
                        newdoc['labels'] = labels
                    self.dataset.append(newdoc)
                    
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        return self.dataset[index]
    
    def save_data(self, data_file):
        with open(data_file, 'wb') as fw:
            pkl.dump(self, fw)
            
    def load_data(data_file):
        with open(data_file, 'rb') as fr:
            return pkl.load(fr)
        
    # remove puncutations and empty sentences
    def preprocess(self):
        for i, instance in enumerate(tqdm(self.dataset, desc="Preprocessing")):
            text = []
            for j, sent in enumerate(instance['text']):
                ppsent = sent.strip().lower().translate(str.maketrans('', '', string.punctuation))
                if len(ppsent.split()) > 1:
                    text.append(ppsent)
            instance['text'] = np.array(text)
    
    # break each sentence string into word tokens
    def tokenize(self):
        for i, instance in enumerate(tqdm(self.dataset, desc="Tokenizing")):
            text = []
            for j, sent in enumerate(instance['text']):
                toksent = np.array(sent.strip().split())
                text.append(toksent)
            instance['text'] = np.array(text, dtype=object)
    
    # generate a vector for each sentence using Sent2Vec
    def sent_vectorize(self, sent2vec_model):      
        for i, instance in enumerate(tqdm(self.dataset, desc="Embedding sentences")):
            esents = sent2vec_model.embed_sentences(instance['text'])
            instance['text'] = np.delete(esents, np.where(esents.sum(axis=1) == 0)[0], axis=0)
        self.sent_vectorized = True


# unified code for generating mini batches of data for both facts and sections during train / dev / test / inference
class MiniBatch:
    def __init__(self, examples, vocab=None, label_vocab=None, schemas=None, type_map=None, node_vocab=None, edge_vocab=None, adjacency=None, hidden_size=200, max_segments=4, max_segment_size=8, num_mpath_samples=2):
        # provide vocab if not sent vectorized, None otherwise
        self.sent_vectorized = True if vocab is None else False
        # provide label_vocab if annotated, None otherwise
        self.annotated = True if label_vocab is not None else False
        # provide graph data if struct encoder is to be used on these examples, None otherwise
        self.sample_metapaths = True if schemas is not None else False

        self.max_segments = max_segments
        
        if not self.sent_vectorized:
            self.vocab = vocab
            self.max_segment_size = max_segment_size
        else:
            self.sent_hidden_size = hidden_size
            
        if self.annotated:
            self.label_vocab = label_vocab
            
        if self.sample_metapaths:
            self.schemas = schemas
            self.type_map = type_map
            self.node_vocab = node_vocab
            self.edge_vocab = edge_vocab
            self.adjacency = adjacency
            self.num_mpath_samples = num_mpath_samples
        
        max_len = max([len(d['text']) for d in examples])
        max_segments = min(self.max_segments, max_len)
        
        # expected shape of text tensors
        if not self.sent_vectorized:
            max_segment_size = min(self.max_segment_size, max([len(s) for d in examples for s in d['text']]))
            self.tokens = torch.zeros(len(examples), max_segments, max_segment_size, dtype=torch.long) # [D, S, W]
        else:
            self.doc_inputs = torch.zeros(len(examples), max_segments, self.sent_hidden_size) # [D, S, H]
        
        self.example_ids = []
        
        if self.annotated:
            # expected shape of true label indicator tensors
            self.labels = torch.zeros(len(examples), len(self.label_vocab)) # [D, C]
        
        for i, instance in enumerate(examples):
            if not self.sent_vectorized:
                for j, sent in enumerate(instance['text']):
                    # fill up the j-th sentence of i-th example with word tokens
                    self.tokens[i, j, :len(sent)] = torch.from_numpy(np.array([self.vocab[w] for w in sent]))   
            else:
                # fill up the i-th example with sentence embeddings
                self.doc_inputs[i, :len(instance['text']), :] = torch.from_numpy(instance['text'])[:max_segments]
            
            self.example_ids.append(instance['id'])
            
            if self.annotated:
                label_list = torch.from_numpy(np.array([self.label_vocab[l] for l in instance['labels']]))
                self.labels[i].scatter_(0, label_list, 1.) 
                
        if not self.sent_vectorized:
            self.mask = (self.tokens != 0).float() # [D, S, W]
        else:
            self.mask = (self.doc_inputs != 0).any(dim=2).float() # [D, S]
                
        if self.sample_metapaths:
            trg_node_tokens = torch.tensor([self.node_vocab[self.type_map[x]][x] for x in self.example_ids])
            self.node_tokens, self.edge_tokens = self.generate_metapaths(trg_node_tokens, self.schemas, self.adjacency, self.edge_vocab, num_samples=self.num_mpath_samples) # N * [M, D, L+1], N * [M, D, L]
    
    # sample metapaths using adjacency matrices
    def generate_metapaths(self, indices, schemas, adjacency, edge_vocab, num_samples=2): # [D,]
        indices = indices.repeat(num_samples) # [M*D,]
        
        tokens, edge_tokens = [], []
        
        # repeat over all schemas
        for i in range(len(schemas)):
            ins_tokens, ins_edge_tokens = [indices], []
            for keys in schemas[i]:
                neighbours = adjacency[keys].sample(num_neighbors=1, subset=ins_tokens[-1]).squeeze(1) # [M*D,]
                relations = torch.full(neighbours.shape, edge_vocab[keys[1]], dtype=torch.long) # [M*D,]
                
                ins_tokens.append(neighbours)
                ins_edge_tokens.append(relations)
            
            ins_tokens = torch.stack(ins_tokens, dim=1)
            ins_tokens = ins_tokens.view(num_samples, -1, ins_tokens.size(1)) # [M, D, L+1]
            
            ins_edge_tokens = torch.stack(ins_edge_tokens, dim=1)
            ins_edge_tokens = ins_edge_tokens.view(num_samples, -1, ins_edge_tokens.size(1)) # [M, D, L]
            
            tokens.append(ins_tokens)
            edge_tokens.append(ins_edge_tokens)
        
        return tokens, edge_tokens                    
    
    # automatic memory pinning for faster cpu to cuda transfer
    def pin_memory(self):
        if not self.sent_vectorized:
            self.tokens.pin_memory()
        else:
            self.doc_inputs.pin_memory()
        self.mask.pin_memory()
        if self.annotated:
            self.labels.pin_memory()
        if self.sample_metapaths:
            for i in range(len(self.node_tokens)):
                self.node_tokens[i].pin_memory()
                self.edge_tokens[i].pin_memory()
        return self
    
    # transfer pinned cpu tensors to cuda
    def cuda(self, dev='cuda'):
        if not self.sent_vectorized:
            self.tokens = self.tokens.cuda(dev, non_blocking=True)
        else:
            self.doc_inputs = self.doc_inputs.cuda(dev, non_blocking=True)
        self.mask = self.mask.cuda(dev, non_blocking=True)
        if self.annotated:
            self.labels = self.labels.cuda(dev, non_blocking=True)
        if self.sample_metapaths:
            for i in range(len(self.node_tokens)):
                self.node_tokens[i] = self.node_tokens[i].cuda(dev, non_blocking=True)
                self.edge_tokens[i] = self.edge_tokens[i].cuda(dev, non_blocking=True)
        return self

def collate_func(examples, **kwargs):
    return MiniBatch(examples, **kwargs) 
