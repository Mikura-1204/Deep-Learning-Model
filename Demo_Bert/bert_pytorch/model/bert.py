import torch.nn as nn

from .transformer import TransformerBlock
from .embedding import BERTEmbedding

# ciallo: in fact, real model is here to construct
#         Main steps is three Mask-Embed-TransformEncode
# 
#   Args: vocab_size: 词汇表的大小，即模型可以处理的不同词的数量。
#         hidden: 隐藏层的维度大小（默认值为 768），表示每个词嵌入向量的维度。
#         n_layers: Transformer Block 的层数，BERT Base 模型有 12 层。
#         attn_heads: 每个 Transformer Block 中的注意力头数，BERT Base 模型有 12 个注意力头。
#         dropout: Dropout 概率，用于防止过拟合
class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        # ciallo: 每个 TransformerBlock 是一个完整的 Transformer 编码器块
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
