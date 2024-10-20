import torch.nn as nn
from .single import Attention

# ciallo: h（注意力头的数量） d_model（模型的维度） dropout（dropout概率，默认为0.1）作为参数
class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h     # ciallo: 计算每个注意力头的维度 d_k
        self.h = h

        # ciallo: create three Linears. Correspond to Q K V and used to dim convert
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)]) 
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # ciallo:对 query、key 和 value 分别应用之前定义的三个线性层，
        # 并将输出形状重塑为 (batch_size, h, seq_len, d_k)，其中 h 是头的数量，seq_len 是序列长度
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
