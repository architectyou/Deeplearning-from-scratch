import torch
import torch.nn as nn
import torch.nn.functional as F

class SlidingWindowAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, window_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.window_size = window_size
        
        # 투영 레이어
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        # 1. Q, K, V 계산
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 2. 슬라이딩 윈도우 마스크 생성
        seq_len = x.size(1)
        mask = self._create_window_mask(seq_len)
        
        # 3. 어텐션 계산
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.hidden_size ** 0.5)
        attn_scores = attn_scores + mask  # 마스크 적용
        
        # 4. 소프트맥스 및 출력 계산
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        return self.out_proj(output)
    
    def _create_window_mask(self, seq_len):
        # 윈도우 마스크 생성 함수
        mask = torch.ones(seq_len, seq_len) * float('-inf')
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 0.0
        return mask