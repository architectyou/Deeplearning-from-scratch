import torch
import math
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size : int, d_model : int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model) # positional embedding pe를 0으로 초기화
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1) # unsqeeze 이용 positional encoding 이용
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term) # 짝수 벡터 값에 적용
        pe[:, 1::2] = torch.cos(position * div_term) # 홀수 벡터 값에 적용

        # register_buffer : 훈련 중에 학습 가능한 매개 변수로 만들지 않고도 pe 저장
        # 모델 내부적으로 가지고 있으면서, 학습은 필요 없는 tensor의 경우 사용
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # 원본 값에 positional encoding 값 더하기
        return x + self.pe[:, :x.size(1)]