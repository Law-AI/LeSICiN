import torch
from model.basicmodules import LstmNet, AttnNet

class HierAttnNet(torch.nn.Module):
    def __init__(self, hidden_size, vocab_size=None, drop=0.1):
        super().__init__()
        
        if vocab_size is not None:
            self.word_embedding = torch.nn.Embedding(vocab_size, hidden_size)
            self.sent_lstm = LstmNet(hidden_size)
            self.sent_attn = AttnNet(hidden_size, drop=drop)
        
        self.doc_lstm = LstmNet(hidden_size)
        self.doc_attn = AttnNet(hidden_size, drop=drop)
        
    def forward(self, tokens=None, doc_inputs=None, mask=None, sent_dyn_context=None, doc_dyn_context=None): # [B, S, W], [B, S, H], [B, S, W] / [B, S], [B, S, H], [B, H]
        if tokens is not None:
            sent_inputs = self.word_embedding(tokens)
            
            # flatten to 3-D
            sent_inputs = sent_inputs.view(-1, sent_inputs.size(2), sent_inputs.size(3))
            sent_mask = mask.view(-1, mask.size(2))
            
            if sent_dyn_context is not None:
                sent_dyn_context = sent_dyn_context.view(-1, sent_dyn_context.size(2))
                
            sent_hidden_all = self.sent_lstm(sent_inputs, sent_mask)
            sent_hidden = self.sent_attn(sent_hidden_all, sent_mask, dyn_context=sent_dyn_context)
            
            doc_inputs = sent_hidden.view(tokens.size(0), tokens.size(1), -1)
            doc_mask = (mask.sum(dim=2) > 0).float()
        else:
            doc_mask = mask
            
        doc_hidden_all = self.doc_lstm(doc_inputs, doc_mask)
        doc_hidden = self.doc_attn(doc_hidden_all, doc_mask, dyn_context=doc_dyn_context)
        return doc_hidden

class MetapathAggrNet(torch.nn.Module):
    def __init__(self, node_vocab_size, edge_vocab_size, hidden_size, drop=0.1, gdel=14.):
        super().__init__()
        self.emb_range = gdel / hidden_size
        
        self.node_embedding = torch.nn.ModuleDict({ntype: torch.nn.Embedding(num_nodes, hidden_size) for ntype, num_nodes in node_vocab_size.items()})
        for ntype, ntype_weights in self.node_embedding.items():
            ntype_weights.weight.data.uniform_(- self.emb_range, self.emb_range)
        
        self.scale_fc = torch.nn.ModuleDict({ntype: torch.nn.Linear(hidden_size, hidden_size) for ntype in node_vocab_size})
        
        self.edge_embedding = torch.nn.Embedding(edge_vocab_size, hidden_size // 2)
        self.edge_embedding.weight.data.uniform_(- self.emb_range, self.emb_range)
        
        self.intra_attention = AttnNet(2 * hidden_size, drop=drop)
        
        self.inter_fc = torch.nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.inter_context = torch.nn.Parameter(torch.rand(2 * hidden_size))
        
        self.output_fc = torch.nn.Linear(2 * hidden_size, hidden_size)
        
        self.dropout = torch.nn.Dropout(drop)
    
    # Embed each node index using the node embedding matrix and then scale to generate same sized embeddings for each node type     
    def embed_and_scale(self, tokens, edge_tokens, schema): # [B, L+1], [B, L]
        inputs, edge_inputs = [], []
        
        node_type = schema[0][0]  
        node_input = self.dropout(self.node_embedding[node_type](tokens[:, 0])) # [B, H]
        inputs.append(self.dropout(self.scale_fc[node_type](node_input)))
        
        for i in range(edge_tokens.size(1)):
            node_type = schema[i][2]
            node_input = self.dropout(self.node_embedding[node_type](tokens[:, i+1])) # [B, H]
            inputs.append(self.dropout(self.scale_fc[node_type](node_input)))
                          
            edge_inputs.append(self.dropout(self.edge_embedding(edge_tokens[:, i])))
        inputs = torch.stack(inputs, dim=1) # [B, L+1, H]
        edge_inputs = torch.stack(edge_inputs, dim=1) # [B, L, H]
        return inputs, edge_inputs
    
    # We are following the official implementation of the RotatE algorithm --- https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding
    def rotational_encoding(self, inputs, edge_inputs): # [B, L+1, H], [B, L, H/2]
        PI = 3.14159265358979323846
        hidden = inputs.clone()
        for i in reversed(range(edge_inputs.size(1))):
            hid_real, hid_imag = torch.chunk(hidden.clone()[:, i+1:, :], 2, dim=2) # [B, L-i, H/2], [B, L-i, H/2]
            inp_real, inp_imag = torch.chunk(inputs[:, i, :], 2, dim=1) # [B, H/2], [B, H/2]
            
            edge_complex = edge_inputs[:, i, :] / (self.emb_range / PI)
            edge_real, edge_imag = torch.cos(edge_inputs[:, i, :]), torch.sin(edge_inputs[:, i, :]) # [B, H/2], [B, H/2]
           
            out_real = inp_real.unsqueeze(1) + edge_real.unsqueeze(1) * hid_real - edge_imag.unsqueeze(1) * hid_imag # [B, L-i, H/2]
            out_imag = inp_imag.unsqueeze(1) + edge_imag.unsqueeze(1) * hid_real + edge_real.unsqueeze(1) * hid_imag # [B, L-i, H/2]
            
            hidden[:, i+1:, :] = torch.cat([out_real, out_imag], dim=2)
        path_lens = 1 + torch.arange(hidden.size(1), device=hidden.device) # [L+1]
        return hidden / path_lens.unsqueeze(0).unsqueeze(2)
                               
    def forward(self, tokens, edge_tokens, schemas, intra_context=None, inter_context=None): 
        hidden = []

        # serially perform intra-metapath aggregation across the different schemas
        for i in range(len(tokens)):
            # flatten out the multiple samples of the same schema                       
            mpath_tokens = tokens[i].view(-1, tokens[i].size(2)) # [M*D, L+1]
            mpath_edge_tokens = edge_tokens[i].view(-1, edge_tokens[i].size(2)) # [M*D, L]
                               
            mpath_inputs, mpath_edge_inputs = self.embed_and_scale(mpath_tokens, mpath_edge_tokens, schemas[i])
                               
            mpath_hidden_all = self.rotational_encoding(mpath_inputs, mpath_edge_inputs) # [M*D, L+1, H]

            # the first element in the sequence is the target node, the rest are transformed embeddings for other nodes in the metapath
            mpath_hidden_all = torch.cat([mpath_hidden_all[:, 0, :].unsqueeze(1).repeat(1, mpath_hidden_all.size(1) - 1, 1), mpath_hidden_all[:, 1:, :]], dim=2) # [M*D, L, 2H]                   
            mpath_hidden = torch.relu(self.intra_attention(mpath_hidden_all, dyn_context=intra_context)) # [M*D, 2H]

            # aggregate transformed embeddings from multiple samples of the same schema 
            mpath_hidden = torch.sum(mpath_hidden.view(tokens[i].size(0), tokens[i].size(1), -1), dim=0) # [D, 2H]
            hidden.append(mpath_hidden)
        hidden = torch.stack(hidden, dim=1) # [D, N, 2H]
        
        # perform inter-metapath aggregation across transformed embeddings for each schema
        hidden_act = torch.mean(torch.tanh(self.dropout(self.inter_fc(hidden))), dim=0).expand_as(hidden) # [D, N, 2H]
        context = self.inter_context.unsqueeze(0).repeat(hidden_act.size(0), 1).unsqueeze(2) if inter_context is None else inter_context.unsqueeze(2)
        scores = torch.bmm(hidden_act, context) # [D, N, 1]
                               
        outputs = torch.sum(hidden * scores, dim=1) # [D, 2H]
        outputs = self.dropout(self.output_fc(outputs)) # [D, H]
        
        return outputs

class MatchNet(torch.nn.Module):
    def __init__(self, hidden_size, num_labels, drop=0.1):
        super().__init__()
        
        self.match_lstm = LstmNet(hidden_size)
        self.match_attn = AttnNet(hidden_size, drop=drop)
        self.match_fc = torch.nn.Linear(2 * hidden_size, num_labels)
        
        self.dropout = torch.nn.Dropout(drop)
        
    def forward(self, fact_inputs, sec_inputs, context=None): # [D, H], [C, H]
        sec_inputs = sec_inputs.expand(fact_inputs.size(0), sec_inputs.size(0), sec_inputs.size(1)) # [D, C, H]
        
        sec_hidden_all = self.match_lstm(sec_inputs) # [D, C, H]
        sec_hidden = self.match_attn(sec_hidden_all, dyn_context=context) # [D, H]
        
        logits = self.dropout(self.match_fc(torch.cat([fact_inputs, sec_hidden], dim=1))) # [D, C]
        scores = torch.sigmoid(logits).detach() # [D, C]
        return logits, scores
