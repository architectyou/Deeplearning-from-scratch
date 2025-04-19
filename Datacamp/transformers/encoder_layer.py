import torch.nn as nn
import torch.nn.functional as F
from multi_head_self_attention import MultiHeadAttention
from embedding import InputEmbeddings, PositionalEncoding

class FeedForwardSubLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        # d_ff = dimension between linear layers
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # forward : processes attention outputs to capture complex, non-linear patterns
        return self.fc2(self.relu(self.fc1(x)))
    
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff_sublayer = FeedForwardSubLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, src_mask):
        # mask : prevents processing padding tokens
        attn_output = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff_sublayer(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
# Transformer Encoder : 여러 개의 EncoderLayer를 쌓아서 구성
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length):
        super().__init__()
        self.embedding = InputEmbeddings(vocab_size, d_model )
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        
    def forward(self, x, src_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
    
    
# Encoder transformer head
class ClassifierHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)
        # fc : fully connected layer (transformers encoder hidden states into num_classes)
        
    def forward(self, x):
        logits = self.fc(x)
        return F.log_softmax(logits, dim = -1)
    

class RegressionHead(nn.Module):
    def __init__(self, d_model, output_dim):
        super().__init__()
        self.fc = nn.Linear(d_model, output_dim)
        # output dim is 1 when predicting a single numericla value
        
    def forward(self, x):
        return self.fc(x)