import torch
import torch_sparse
from tqdm import tqdm
from collections import defaultdict

from data_helper import MiniBatch

def generate_vocabs(train_data, label_data, limit=30000, thresh=1):
    if not train_data.sent_vectorized:
        freqs = defaultdict(int)
        for instance in tqdm(train_data.dataset + label_data.dataset, desc="Creating vocabulary"):
            for sent in instance['text']:
                for word in sent:
                    freqs[word] += 1
        vocab_set = set(w for w, f in freqs.items() if f >= thresh)
        vocab = {k: i for i, k in enumerate(vocab_set)}
    else:
        vocab = None
    
    label_vocab = {}
    for instance in tqdm(label_data.dataset, desc="Creating label vocabulary"):
        label_vocab[instance['id']] = len(label_vocab)

    return vocab, label_vocab

def generate_graph(label_vocab, type_map, label_tree_edges, cit_net_edges, label_name='section'):
    node_vocab = defaultdict(dict)
    node_vocab[label_name] = label_vocab

    edge_vocab = {}
    edge_indices = defaultdict(list)

    for (node_a, edge_type, node_b) in label_tree_edges + cit_net_edges:
        node_a_type, node_b_type = type_map[node_a], type_map[node_b]

        if edge_type not in edge_vocab:
            edge_vocab[edge_type] = len(edge_vocab)

        if node_a not in node_vocab[node_a_type]:
            node_vocab[node_a_type][node_a] = len(node_vocab[node_a_type])
        if node_b not in node_vocab[node_b_type]:
            node_vocab[node_b_type][node_b] = len(node_vocab[node_b_type])
            
        node_a_token = node_vocab[node_a_type][node_a]
        node_b_token = node_vocab[node_b_type][node_b]

        edge_indices[(node_a_type, edge_type, node_b_type)].append([node_a_token, node_b_token])

    num_nodes = {ntype: len(nodes) for ntype, nodes in node_vocab.items()}

    adjacency = {}
    for keys, edges in edge_indices.items():
        row, col = torch.tensor(edges).t()
        sizes = (num_nodes[keys[0]], num_nodes[keys[-1]])
        adj = torch_sparse.SparseTensor(row=row, col=col, sparse_sizes=sizes)
        adjacency[tuple(keys)] = adj

    return node_vocab, edge_vocab, edge_indices, adjacency

def generate_label_weights(train_data, label_vocab, dev='cuda:0', scheme="tws", thresh=10.):
    pos = torch.zeros(len(label_vocab), device=dev)
    for instance in tqdm(train_data, desc="Generating label weights"):
        for l in instance['labels']:
            pos[label_vocab[l]] += 1
    weights = torch.clamp(pos.max() / pos, max=thresh) if scheme == 'tws' else len(train_data) / pos
    return weights

def collate_func(examples, **kwargs):
    return MiniBatch(examples, **kwargs) 

def train_dev_pass(model, optimizer, fact_loader, sec_batch, metrics, pred_threshold=None, train=False):
    model.train() if train else model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    for i, fact_batch in enumerate(tqdm(fact_loader, desc="Flowing data through model")):
        if fact_batch.doc_inputs.size(1) != fact_batch.mask.size(1):
            print("Before Model, Fact and Sec:", fact_batch.doc_inputs.shape, fact_batch.mask.shape)
        if fact_batch.labels.size(1) != 100:
            print("Before model:", fact_batch.doc_inputs.shape, fact_batch.mask.shape, fact_batch.labels.shape)
            print(fact_batch.labels)
        
        loss, predictions = model(fact_batch.cuda(), sec_batch.cuda(), pthresh=pred_threshold)
        
        if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        batch_loss = loss.item()
        metrics(predictions, fact_batch.labels, loss=batch_loss)
    
    return metrics.calculate_metrics()

class MultiLabelMetrics(torch.nn.Module):
    def __init__(self, num_classes, dev='cuda', loss=True):
        super().__init__()
        
        self.match = torch.zeros(num_classes, device=dev)
        self.predictions = torch.zeros(num_classes, device=dev)
        self.labels = torch.zeros(num_classes, device=dev)
        self.counter = 0
        if loss:
            self.run_loss = 0
        
    def forward(self, predictions, labels, loss=None):
        match = predictions * labels
        
        self.match += match.sum(dim=0)
        self.predictions += predictions.sum(dim=0)
        self.labels += labels.sum(dim=0)
        self.counter += 1
        if loss is not None:
            self.run_loss += loss
        
    def refresh(self):
        self.match.fill_(0)
        self.predictions.fill_(0)
        self.labels.fill_(0)
        self.counter = 0
        if 'run_loss' in self.__dict__:
            self.run_loss = 0
        return self
            
    def calculate_metrics(self, refresh=True):
        prec = self.match / self.predictions
        rec = self.match / self.labels
        
        prec[prec.isnan()] = 0
        rec[rec.isnan()] = 0
        
        f1 = 2 * prec * rec / (prec + rec)
        f1[f1.isnan()] = 0
        
        self.macro_prec = prec.mean().item()
        self.macro_rec = rec.mean().item()
        self.macro_f1 = f1.mean().item()
        
        match_total = self.match.sum().item()
        preds_total = self.predictions.sum().item()
        labels_total = self.labels.sum().item()
        
        self.micro_prec = match_total / preds_total if preds_total > 0 else 0
        self.micro_rec = match_total / labels_total
        self.micro_f1 = 0 if self.micro_prec + self.micro_rec == 0 else 2 * self.micro_prec * self.micro_rec / (self.micro_prec + self.micro_rec) 
        self.jacc = match_total / (preds_total + labels_total - match_total)

        if 'run_loss' in self.__dict__:
            self.loss = self.run_loss / self.counter
        
        if refresh:
            self.refresh()
        
        return self
