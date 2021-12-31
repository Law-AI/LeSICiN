print("\nPreparing PyTorch environment")
print("==================================================")

import torch
import random
from tqdm import tqdm
from functools import partial
import sent2vec

from model.model import *
from data_helper import *
from helper import *


torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

with open("configs/data_path.json") as fr:
    dc = json.load(fr)
with open("configs/hyperparams.json") as fr:
    hc = json.load(fr)

SEED = hc['seed']

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

if dc['s2v_path'] is not None:
    sent2vec_model = sent2vec.Sent2vecModel()
    sent2vec_model.load_model(dc['s2v_path'])

print("\nPreparing Datasets")
print("==================================================")
sec_dataset = LSIDataset(jsonl_file=dc['sec_src'])
sec_dataset.preprocess()
sec_dataset.sent_vectorize(sent2vec_model)
sec_dataset.save_data(dc['sec_cache'])

# sec_dataset = LSIDataset.load_data(dc['dev_cache'])

if hc['do_train_dev']:
    train_dataset = LSIDataset(jsonl_file=dc['train_src'])
    train_dataset.preprocess()
    train_dataset.sent_vectorize(sent2vec_model)
    train_dataset.save_data(dc['train_cache'])

    # train_dataset = LSIDataset.load_data(dc['train_cache'])

    dev_dataset = LSIDataset(jsonl_file=dc['dev_src'])
    dev_dataset.preprocess()
    dev_dataset.sent_vectorize(sent2vec_model)
    dev_dataset.save_data(dc['dev_cache'])

    # dev_dataset = LSIDataset.load_data(dc['dev_cache'])

if hc['do_test']:
    test_dataset = LSIDataset(jsonl_file=dc['test_src'])
    test_dataset.preprocess()
    test_dataset.sent_vectorize(sent2vec_model)
    test_dataset.save_data(dc['test_cache'])

    # test_dataset = LSIDataset.load_data(dc['test_cache'])

if hc['do_infer']:
    infer_dataset = LSIDataset(jsonl_file=dc['infer_src'])
    infer_dataset.preprocess()
    infer_dataset.sent_vectorize(sent2vec_model)
    infer_dataset.save_data(dc['infer_cache'])

    # infer_dataset = LSIDataset.load_data(dc['infer_cache'])

print("\nGathering other data")
print("==================================================")
vocab, label_vocab = generate_vocabs(train_dataset, sec_dataset, limit=hc['vocab_limit'], thresh=hc['vocab_thresh'])
with open(dc['type_map']) as fr:
    type_map = json.load(fr)
with open(dc['label_tree']) as fr:
    label_tree = json.load(fr)
with open(dc['citation_network']) as fr:
    citation_net = json.load(fr)
with open(dc['schemas']) as fr:
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
        max_segments=hc['max_segments'],
        max_segment_size=hc['max_segment_size'],
        num_mpath_samples=hc['num_mpath_samples']
        ), 
    pin_memory=True, 
    num_workers=4
)

if hc['do_train_dev']:
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=hc['train_bs'], 
        collate_fn=partial(
            collate_func, 
            label_vocab=label_vocab, 
            schemas=schemas['fact'], 
            type_map=type_map, 
            node_vocab=node_vocab, 
            edge_vocab=edge_vocab, 
            adjacency=adjacency, 
            max_segments=hc['max_segments'],
            max_segment_size=hc['max_segment_size'], 
            num_mpath_samples=hc['num_mpath_samples']
            ), 
        pin_memory=True, 
        num_workers=4
    )

    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, 
        batch_size=hc['dev_bs'], 
        collate_fn=partial(
            collate_func, 
            label_vocab=label_vocab,  
            max_segments=hc['max_segments'],
            max_segment_size=hc['max_segment_size']
            ), 
        pin_memory=True, 
        num_workers=4
    )

if hc['do_test']:
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=hc['test_bs'], 
        collate_fn=partial(
            collate_func, 
            label_vocab=label_vocab,  
            max_segments=hc['max_segments'],
            max_segment_size=hc['max_segment_size']
            ), 
        pin_memory=True, 
        num_workers=4
    )

if hc['do_infer']:
    infer_loader = torch.utils.data.DataLoader(
        infer_dataset, 
        batch_size=hc['infer_bs'], 
        collate_fn=partial(
            collate_func, 
            label_vocab=label_vocab,  
            max_segments=hc['max_segments'],
            max_segment_size=hc['max_segment_size']
            ), 
        pin_memory=True, 
        num_workers=4
    )

for sec_batch in sec_loader:
    break

print("\nPreparing Model")
print("==================================================")
lsc_model = LeSICiN(
    hc['hidden_size'], 
    L, 
    N, 
    E, 
    label_weights=sec_weights, 
    lambdas=hc['lambdas'], 
    thetas=hc['thetas'], 
    pthresh=hc['pthresh'], 
    drop=hc['dropout']
    ).cuda()

if dc['model_load'] is not None:
    lsc_model.load_state_dict(torch.load(dc['model_load'], map_location='cuda'))


if hc['do_train_dev']:
    if dc['metrics_load'] is not None:
        with open(dc['metrics_dump'], 'rb') as fr:
           best_metrics = pkl.load(fr)
           best_loss = best_metrics.loss
    else:
        best_loss = float('inf')

    best_model = lsc_model.state_dict()


    optimizer = torch.optim.AdamW(lsc_model.parameters(), lr=hc['opt_lr'], weight_decay=hc['opt_wt_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=hc['sch_patience'], factor=hc['sch_factor'])
    train_mlmetrics = MultiLabelMetrics(L)
    dev_mlmetrics = MultiLabelMetrics(L)

    print("\nRunning Train/Dev")
    print("==================================================")
    for epoch in range(hc['num_epochs']):
        train_mlmetrics = train_dev_pass(lsc_model, optimizer, train_loader, sec_batch, metrics=train_mlmetrics, train=True, pred_threshold=hc['pthresh'])
        dev_mlmetrics = train_dev_pass(lsc_model, optimizer, dev_loader, sec_batch, metrics=dev_mlmetrics, pred_threshold=hc['pthresh'])
        
        train_loss, dev_loss = train_mlmetrics.loss, dev_mlmetrics.loss

        if dev_loss < best_loss:
            best_loss = dev_loss
            best_metrics = dev_mlmetrics
            best_model = lsc_model.state_dict()
            
        scheduler.step(dev_loss)
            
        print("%5d || %.4f | %.4f || %.4f | %.4f %.4f %.4f" % (epoch, train_loss, train_mlmetrics.macro_f1, dev_loss, dev_mlmetrics.macro_prec, dev_mlmetrics.macro_rec, dev_mlmetrics.macro_f1))

    print("\nCollecting outputs")
    print("==================================================")
    torch.save(best_model, dc['model_dump'])
    with open(dc['dev_metrics_dump'], 'wb') as fw:
    	pkl.dump(best_metrics, fw)

    if hc['do_test']:
        lsc_model.load_state_dict(best_model)

    print("VALIDATION Results || %.4f | %.4f %.4f %.4f" % (best_loss, best_metrics.macro_prec, best_metrics.macro_rec, best_metrics.macro_f1))

if hc['do_test']:
    test_mlmetrics = MultiLabelMetrics(L)
    print("\nRunning Test")
    print("==================================================")
    test_mlmetrics = train_dev_pass(lsc_model, optimizer, test_loader, sec_batch, metrics=test_mlmetrics, pred_threshold=hc['pthresh'])
    with open(dc['test_metrics_dump'], 'wb') as fw:
        pkl.dump(test_mlmetrics, fw)
    print("TEST Results || %.4f | %.4f %.4f %.4f" % (test_mlmetrics.loss, test_mlmetrics.macro_prec, test_mlmetrics.macro_rec, test_mlmetrics.macro_f1))

if hc['do_infer']:
    print("\nRunning Test")
    print("==================================================")
    infer_outputs = train_dev_pass(lsc_model, optimizer, infer_loader, sec_batch, infer=True, pred_threshold=hc['pthresh'], label_vocab=label_vocab)
    with open(dc['infer_trg'], 'w') as fw:
        fw.write('\n'.join([json.dumps(doc) for doc in infer_outputs]))


