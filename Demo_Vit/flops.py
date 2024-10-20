import torch
from fvcore.nn import FlopCountAnalysis

from vit_model import Attention

# ciallo: 这段代码的主要目的是比较自注意力层（单头）和多头注意力层（8头）的计算复杂度。通过生成相同输入形状的随机张量，
# 并计算其在这两种注意力层上的浮点运算量（FLOPs），我们可以分析两者在计算上的差异
def main():
    # Self-Attention
    a1 = Attention(dim=512, num_heads=1)
    a1.proj = torch.nn.Identity()  # remove Wo

    # Multi-Head Attention
    a2 = Attention(dim=512, num_heads=8)

    # [batch_size, num_tokens, total_embed_dim]
    t = (torch.rand(32, 1024, 512),)

    flops1 = FlopCountAnalysis(a1, t)
    print("Self-Attention FLOPs:", flops1.total())

    flops2 = FlopCountAnalysis(a2, t)
    print("Multi-Head Attention FLOPs:", flops2.total())


if __name__ == '__main__':
    main()

