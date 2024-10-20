import torch.nn as nn

# ciallo: 将段落标签（如句子A、句子B）映射到对应的嵌入向量，通常在 BERT 模型中用于表示哪个句子属于哪一部分
#         Like the location of Sequence
class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)
