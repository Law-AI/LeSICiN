import torch
import torch_sparse
from tqdm import tqdm
from collections import defaultdict

from data_helper import MiniBatch

# Create word vocab (if not sent vectorized) and label vocab from the specific order in the Section dataset
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

# Create the entire graph by combining the label tree and fact-sec citation network
def generate_graph(label_vocab, type_map, label_tree_edges, cit_net_edges, label_name='section'):
    node_vocab = defaultdict(dict) # each key is a node_type, and each value is a dict storing the vocab of all nodes under given node type
    node_vocab[label_name] = label_vocab # manually set this since we want the label vocab to be consistent with node vocab for labels

    edge_vocab = {}
    edge_indices = defaultdict(list) # each key is a tuple (src node type, relationship name, trg node type), and each value is a list storing the edges from src node type to trg node type

    for (node_a, edge_type, node_b) in label_tree_edges + cit_net_edges:
        # first get the node type
        node_a_type, node_b_type = type_map[node_a], type_map[node_b]

        # create new vocab entries for edges and nodes
        if edge_type not in edge_vocab:
            edge_vocab[edge_type] = len(edge_vocab)

        if node_a not in node_vocab[node_a_type]:
            node_vocab[node_a_type][node_a] = len(node_vocab[node_a_type])
        if node_b not in node_vocab[node_b_type]:
            node_vocab[node_b_type][node_b] = len(node_vocab[node_b_type])
            
        # get node indices    
        node_a_token = node_vocab[node_a_type][node_a]
        node_b_token = node_vocab[node_b_type][node_b]

        edge_indices[(node_a_type, edge_type, node_b_type)].append([node_a_token, node_b_token])

    num_nodes = {ntype: len(nodes) for ntype, nodes in node_vocab.items()}

    # same as edge_indices except that the edges under each key are now stored as sparse matrices
    adjacency = {}
    for keys, edges in edge_indices.items():
        row, col = torch.tensor(edges).t()
        sizes = (num_nodes[keys[0]], num_nodes[keys[-1]])
        adj = torch_sparse.SparseTensor(row=row, col=col, sparse_sizes=sizes)
        adjacency[tuple(keys)] = adj

    return node_vocab, edge_vocab, edge_indices, adjacency

# create label weights for BCE Loss since we have unbalanced class distribution
def generate_label_weights(train_data, label_vocab, dev='cuda:0', scheme="tws", thresh=10.):
    pos = torch.zeros(len(label_vocab), device=dev)
    for instance in tqdm(train_data, desc="Generating label weights"):
        for l in instance['labels']:
            pos[label_vocab[l]] += 1
    weights = torch.clamp(pos.max() / pos, max=thresh) if scheme == 'tws' else len(train_data) / pos
    return weights

# Unified code to deal with a single train / dev / test / inference pass over the dataset
def train_dev_pass(model, optimizer, fact_loader, sec_batch, metrics=None, pred_threshold=None, train=False, infer=False, label_vocab=False):
    model.train() if train else model.eval()
    
    if infer:
        outputs = []
        inv_label_vocab = {v: k for k, v in label_vocab.items()}

    for i, fact_batch in enumerate(tqdm(fact_loader, desc="Flowing data through model")):    
        torch.cuda.empty_cache()

        loss, predictions = model(fact_batch.cuda(), sec_batch.cuda(), pthresh=pred_threshold)
        
        if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        if not infer:
            batch_loss = loss.item()
            metrics(predictions, fact_batch.labels, loss=batch_loss)
        
        else:
            for i, instance_preds in enumerate(predictions):
                # gather true predictions
                pred_list_indices = torch.nonzero(instance_preds, as_tuple=False).squeeze(1)
                pred_list = [inv_label_vocab[idx] for idx in pred_list_indices]
                outputs.append({'id': fact_batch.example_ids[i], 'predictions': pred_list})

    return metrics.calculate_metrics() if not infer else outputs

class MultiLabelMetrics(torch.nn.Module):
    def __init__(self, num_classes, dev='cuda', loss=True):
        super().__init__()
        
        self.match = torch.zeros(num_classes, device=dev) # count no. of true positives for each label
        self.predictions = torch.zeros(num_classes, device=dev) # count no. of true positives + false positives for each label
        self.labels = torch.zeros(num_classes, device=dev) # count no. of true positives + false negatives for each label
        self.run_jacc = 0 # running sum of jaccard scores
        self.counter = 0 # count no. of batches
        if loss:
            self.run_loss = 0 # running sum of losses
    
    # to be called with a batch of predictions and true labels    
    def forward(self, predictions, labels, loss=None):
        match = predictions * labels # true positives for this batch
        
        # increment counts
        self.match += match.sum(dim=0)
        self.predictions += predictions.sum(dim=0)
        self.labels += labels.sum(dim=0)
        self.run_jacc += torch.sum(torch.logical_and(predictions, labels).sum(dim=1) / torch.logical_or(predictions, labels).sum(dim=1)).item()
        self.counter += 1

        if loss is not None:
            self.run_loss += loss
        
    # reset counters    
    def refresh(self):
        self.match.fill_(0)
        self.predictions.fill_(0)
        self.labels.fill_(0)
        self.run_jacc = 0
        self.counter = 0
        if 'run_loss' in self.__dict__:
            self.run_loss = 0
        return self
            
    # calculate the metrics and return self
    def calculate_metrics(self, refresh=True):
        prec = self.match / self.predictions # P = TP / (TP + FP)
        rec = self.match / self.labels # R = TP (TP + FN)
        
        prec[prec.isnan()] = 0
        rec[rec.isnan()] = 0
        
        f1 = 2 * prec * rec / (prec + rec) # F1 = 2 * P * R / (P + R)
        f1[f1.isnan()] = 0
        
        # macro --> average across each label
        self.macro_prec = prec.mean().item()
        self.macro_rec = rec.mean().item()
        self.macro_f1 = f1.mean().item()
        
        match_total = self.match.sum().item()
        preds_total = self.predictions.sum().item()
        labels_total = self.labels.sum().item()
        
        # micro --> take total counts
        self.micro_prec = match_total / preds_total if preds_total > 0 else 0
        self.micro_rec = match_total / labels_total
        self.micro_f1 = 0 if self.micro_prec + self.micro_rec == 0 else 2 * self.micro_prec * self.micro_rec / (self.micro_prec + self.micro_rec) 
        
        self.jacc = self.run_jacc / self.counter
        if 'run_loss' in self.__dict__:
            self.loss = self.run_loss / self.counter
        
        if refresh:
            self.refresh()
        
        return self
