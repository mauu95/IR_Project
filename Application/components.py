import torch
import torch.nn as nn
from utility import *
from dictionary import *
import torch.nn.functional as F
from torch import Tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def position_encoding(seq_len: int, dim_model: int = 512, device: torch.device = torch.device("cpu")):
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** torch.div(dim, dim_model, rounding_mode="floor"))

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase)).reshape(seq_len,1,dim_model)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )

class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))

def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )

def scaled_dot_product_attention(query, key, value):
    attention_matrix = torch.bmm(query, key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = F.softmax(attention_matrix / scale, dim=-1)
    result = softmax.bmm(value)
    return result

class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, query, key, value):
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))

def Sentence2Tensor(dictionary, sentence):
    indexes = dictionary.indexes_from_sentence(sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def Indexes2Tensor(word_index):
    return torch.tensor(word_index, dtype=torch.long, device=device).view(-1, 1)