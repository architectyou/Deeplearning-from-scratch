# Trnasformer Decoder
# -> Autoregressive sequence generation : text generation & completion

from torch import nn
from torch.nn import functional as F
from embedding import InputEmbeddings, PositionalEncoding
from encoder_layer import FeedForwardSubLayer
from multi_head_self_attention import MultiHeadAttention

'''
1. Maked multi-head self-attention
- hide later tokens in sequence

2. Decoder-Only transformer head
- linear + softmax over vocab
- predict most likely next tokens
'''

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff_sublayer = FeedForwardSubLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # encoder 와 다른 점은 padding mask 대신 tgt_mask (causal mask) 사용
    def forward(self, x, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff_sublayer(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
    
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length):
        super(TransformerDecoder, self).__init__()
        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        
        
    def forward(self, x, tgt_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, tgt_mask)
        
        x = self.fc(x)
        return F.log_softmax(x, dim = -1)