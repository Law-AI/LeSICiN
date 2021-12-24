import torch
import random
from tqdm import tqdm
from functools import partial

from model.model import *
from data_helper import *
from helper import *

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

sent2vec_model = sent2vec.Sent2vecModel()
sent2vec_model.load_model("ils2v.bin")

train_dataset = LSIDataset(jsonl_file="data/train.jsonl")
train_dataset.preprocess()
train_dataset.sent_vectorize(sent2vec_model)
train_dataset.save_data("cache/train.pkl")

# train_dataset = LSIDataset.load_data("cache/train.pkl")

sec_dataset = LSIDataset(jsonl_file="data/secs.jsonl")
sec_dataset.preprocess()
sec_dataset.sent_vectorize(sent2vec_model)
sec_dataset.save_data("cache/secs.pkl")

# sec_dataset = LSIDataset.load_data("cache/secs.pkl")

dev_dataset = LSIDataset(jsonl_file="data/dev.jsonl")
dev_dataset.preprocess()
dev_dataset.sent_vectorize(sent2vec_model)
dev_dataset.save_data("cache/dev.pkl")

# dev_dataset = LSIDataset.load_data("cache/dev.pkl")

test_dataset = LSIDataset(jsonl_file="data/test.jsonl")
test_dataset.preprocess()
test_dataset.sent_vectorize(sent2vec_model)
test_dataset.save_data("cache/test.pkl")

# test_dataset = LSIDataset.load_data("cache/test.pkl")

_, label_vocab = generate_vocabs(train_dataset, sec_dataset)
with open("data/type_map.json") as fr:
    type_map = json.load(fr)
with open("data/label_tree.json") as fr:
    label_tree = json.load(fr)
with open("data/citation_network.json") as fr:
    citation_net = json.load(fr)
with open("data/schemas.json") as fr:
    schemas = json.load(fr)
for sch in schemas.values():
    for path in sch:
        for i, edge in enumerate(path):
            path[i] = tuple(path[i])

node_vocab, edge_vocab, edge_indices, adjacency = generate_graph(label_vocab, type_map, label_tree, citation_net)
sec_weights = generate_label_weights(train_dataset, label_vocab)

L = len(label_vocab)
N = {k: len(v) for k,v in node_vocab.items()}
E = len(edge_vocab)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=64, 
    collate_fn=partial(
        collate_func, 
        label_vocab=label_vocab, 
        schemas=schemas['fact'], 
        type_map=type_map, 
        node_vocab=node_vocab, 
        edge_vocab=edge_vocab, 
        adjacency=adjacency, 
        max_segments=128, 
        num_mpath_samples=8), 
    pin_memory=True, 
    num_workers=4
)

sec_loader = torch.utils.data.DataLoader(
    sec_dataset, 
    batch_size=len(label_vocab), 
    collate_fn=partial(
        collate_func, 
        schemas=schemas['section'], 
        type_map=type_map, 
        node_vocab=node_vocab, 
        edge_vocab=edge_vocab, 
        adjacency=adjacency, 
        max_segments=128, 
        num_mpath_samples=4), 
    pin_memory=True, 
    num_workers=4
)

dev_loader = torch.utils.data.DataLoader(
    dev_dataset, 
    batch_size=512, 
    collate_fn=partial(
        collate_func, 
        label_vocab=label_vocab,  
        max_segments=128
    ), 
    pin_memory=True, 
    num_workers=4
)

for sec_batch in sec_loader:
    break

lsc_model = LeSICiN(200, L, N, E, label_weights=sec_weights, lambdas=(0.25,0.75), thetas=(1,2,3)).cuda()

lsc_model.load_state_dict(torch.load("saved/best_model2.pt", map_location='cuda'))

with open("saved/best_metrics.pkl", 'rb') as fr:
   best_metrics = pkl.load(fr)
   best_loss = best_metrics.loss
best_model = lsc_model.state_dict()
# best_loss = 0.1601

optimizer = torch.optim.AdamW(lsc_model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.2)
train_mlmetrics = MultiLabelMetrics(L)
dev_mlmetrics = MultiLabelMetrics(L)


for epoch in range(20):
    train_mlmetrics = train_dev_pass(lsc_model, optimizer, train_loader, sec_batch, train_mlmetrics, train=True, pred_threshold=0.51)
    dev_mlmetrics = train_dev_pass(lsc_model, optimizer, dev_loader, sec_batch, dev_mlmetrics, pred_threshold=0.51)
    
    train_loss, dev_loss = train_mlmetrics.loss, dev_mlmetrics.loss

    if dev_loss < best_loss:
        best_loss = dev_loss
        best_metrics = dev_mlmetrics
        best_model = lsc_model.state_dict()
        
    scheduler.step(dev_loss)
        
    print("%5d || %.4f | %.4f || %.4f | %.4f %.4f %.4f" % (epoch, train_loss, train_mlmetrics.macro_f1, dev_loss, dev_mlmetrics.macro_prec, dev_mlmetrics.macro_rec, dev_mlmetrics.macro_f1))

torch.save(best_model, "saved/best_model.pt")
with open("saved/best_metrics.pkl", 'wb') as fw:
	pkl.dump(best_metrics, fw)
