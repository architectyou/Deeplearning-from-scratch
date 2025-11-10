# attention mechanism
import torch.nn as nn
import torch.nn.functional as F
import torch

class MultiHeadAttention(nn.Module):
    def __self__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads" # asserts 정의 갑이 맞지 않으면 AssertError 반환
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        # attention head에 대한 3개의 linear layer 정의
        # bias false = no impact on performance while reducing complexity (only for inputs)
        self.query_linear = nn.Linear(d_model, d_model, bias = False)
        self.key_linear = nn.Linear(d_model, d_model, bias = False)
        self.value_linear = nn.Linear(d_model, d_model, bias = False)
        
        self.output_linear = nn.Linear(d_model, d_model)
        
    # mechanism의 다양한 프로세스에 대한 메서드 정의

    def split_heads(self, x, batch_size):
        # head간의 query, key, value tensor 분리
        seq_length = x.size(1)
        x = x.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3) # 0번째 차원은 배치 차원, 2번째 차원은 헤드 차원, 1번째 차원은 시퀀스 차원, 3번째 차원은 헤드 차원
        
        
    def compute_attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None :
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim = -1)
        return torch.matmul(attention_weights, value)
        
    def combine_heads(self, x, batch_size):
        # attention head를 원래의 임베딩 모양으로 다시 변환
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch_size, -1, self.d_model)
    
    def forward(self, query, key, value, mask = None):
        batch_size = query.size(0)
        
        query = self.split_heads(self.query_linear(query), batch_size)
        key = self.split_heads(self.key_linear(key), batch_size)
        value = self.split_heads(self.value_linear(value), batch_size)
        
        attention_weights = self.compute_attention(query, key, value, mask)
        
        output = self.combine_heads(attention_weights, batch_size)
        return self.output_linear(output)