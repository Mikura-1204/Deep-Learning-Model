import torch.nn as nn

# ciallo: 将输入序列中的词汇或标记映射到对应的嵌入向量，这是模型处理文本输入的基础
class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)
