import torch

class LstmNet(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)
        
    def forward(self, inputs, mask=None): # [B, S, H], [B, S]
        mask = mask if mask is not None else torch.ones(inputs.size(0), inputs.size(1), device=inputs.device)
        lengths = mask.sum(dim=-1) # [B,]
        
        # need to pack inputs before passing to RNN and unpack obtained outputs
        pck_inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        pck_hidden_all = self.lstm(pck_inputs)[0]
        hidden_all = torch.nn.utils.rnn.pad_packed_sequence(pck_hidden_all, batch_first=True)[0] # [B, S, H]
        
        return hidden_all

class AttnNet(torch.nn.Module):
    def __init__(self, hidden_size, drop=0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.attn_fc = torch.nn.Linear(hidden_size, hidden_size)
        self.context = torch.nn.Parameter(torch.rand(hidden_size))
        
        self.dropout = torch.nn.Dropout(drop)
        
    def forward(self, inputs, mask=None, dyn_context=None): # [B, S, H], [B, S], [B, H]
        
        mask = mask if mask is not None else torch.ones(inputs.size(0), inputs.size(1), device=inputs.device)
        # use static (learned) context vector if dynamic context is unavailable
        context = dyn_context if dyn_context is not None else self.context.expand(inputs.size(0), self.hidden_size) # [B, H]
        
        act_inputs = torch.tanh(self.dropout(self.attn_fc(inputs)))
        
        scores = torch.bmm(act_inputs, context.unsqueeze(2)).squeeze(2) # [B, S]
        msk_scores = scores.masked_fill((1 - mask).bool(), -1e-32)
        msk_scores = torch.nn.functional.softmax(msk_scores, dim=1)
        
        hidden = torch.sum(inputs * msk_scores.unsqueeze(2), dim=1) # [B, H]
        return hidden
