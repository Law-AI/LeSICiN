import torch
from model.submodules import HierAttnNet, MetapathAggrNet, MatchNet

class LeSICiN(torch.nn.Module):
    def __init__(self, hidden_size, num_labels, node_vocab_size, edge_vocab_size, vocab_size=None, label_weights=None, pthresh=0.65, lambdas=(0.25, 0.75), thetas=(1, 2, 3), drop=0.5):
        super().__init__()
        
        self.text_encoder = HierAttnNet(hidden_size, vocab_size=vocab_size)
        self.graph_encoder = MetapathAggrNet(node_vocab_size, edge_vocab_size, hidden_size)
        self.match_network = MatchNet(hidden_size, num_labels)
        
        self.match_context_transform = torch.nn.Linear(hidden_size, hidden_size)
        self.intra_context_transform = torch.nn.Linear(hidden_size, 2 * hidden_size)
        self.inter_context_transform = torch.nn.Linear(hidden_size, 2 * hidden_size)
        
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=label_weights)
        
        self.pred_threshold = pthresh
        self.lambdas = lambdas
        self.thetas = thetas
        self.dropout = torch.nn.Dropout(drop)
        
    def calculate_losses(self, logits_list, labels):
        loss = 0
        for i, logits in enumerate(logits_list):
            if logits is not None:
                loss += self.thetas[i] * self.criterion(logits, labels)
        return loss
        
    def forward(self, fact_batch, sec_batch, pthresh=None):
        if pthresh is not None:
            self.pred_threshold = pthresh        
        
        # Encode facts using HAN
        if not fact_batch.sent_vectorized:
            fact_attr_hidden = self.text_encoder(tokens=fact_batch.tokens, mask=fact_batch.mask) # [D, H] 
        else:
            fact_attr_hidden = self.text_encoder(doc_inputs=fact_batch.doc_inputs, mask=fact_batch.mask) # [D, H]
        
        # Encode secs using HAN
        if not sec_batch.sent_vectorized:
            sec_attr_hidden = self.text_encoder(tokens=sec_batch.tokens, mask=sec_batch.mask) # [C, H]
        else:
            sec_attr_hidden = self.text_encoder(doc_inputs=sec_batch.doc_inputs, mask=sec_batch.mask) # [C, H]
        
        attr_match_context = self.dropout(self.match_context_transform(fact_attr_hidden))
        
        # print(fact_attr_hidden.shape, sec_attr_hidden.shape, attr_match_context.shape)
        attr_logits, attr_scores = self.match_network(fact_attr_hidden, sec_attr_hidden, context=attr_match_context)
        
        sec_intra_context = self.dropout(self.intra_context_transform(sec_attr_hidden)).repeat(sec_batch.num_mpath_samples, 1) # [M*C, H]
        sec_inter_context = self.dropout(self.inter_context_transform(sec_attr_hidden)) # [C, H]
        
        sec_struct_hidden = self.graph_encoder(sec_batch.node_tokens, sec_batch.edge_tokens, sec_batch.schemas, intra_context=sec_intra_context, inter_context=sec_inter_context) # [C, H] 
        
        align_logits, align_scores = self.match_network(fact_attr_hidden, sec_struct_hidden, context=attr_match_context)
        
        if fact_batch.sample_metapaths:
            # Generate contexts
            fact_intra_context = self.dropout(self.intra_context_transform(fact_attr_hidden)).repeat(fact_batch.num_mpath_samples, 1) # [M*D, H]
            fact_inter_context = self.dropout(self.inter_context_transform(fact_attr_hidden)) # [D, H]
            
            fact_struct_hidden = self.graph_encoder(fact_batch.node_tokens, fact_batch.edge_tokens, fact_batch.schemas, intra_context=fact_intra_context, inter_context=fact_inter_context) # [D, H]
            struct_match_context = self.dropout(self.match_context_transform(fact_struct_hidden)) # [D, H]
            struct_logits, struct_scores = self.match_network(fact_struct_hidden, sec_struct_hidden, context=struct_match_context)
            
        else:
            struct_logits = None
            
        scores = (self.lambdas[0] * attr_scores + self.lambdas[-1] * align_scores)
        predictions = (scores > self.pred_threshold).float()
        
        if fact_batch.annotated:
            if attr_logits.shape != fact_batch.labels.shape:
                print(attr_logits.shape, fact_batch.labels.shape)
            loss = self.calculate_losses([attr_logits, struct_logits, align_logits], fact_batch.labels)
        else:
            loss = None
            
        return loss, predictions
