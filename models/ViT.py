import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange


class GlobalMSA(nn.Module):
    "Global Multi-head Self-Attention, without layernorm"
    def __init__(self, dim, heads=8, dim_head=64, dropout=0) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        project_out = not (heads == 1 and dim_head == dim)

        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x) # b, hw, dim_in -> b, hw, head*dim
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v)) # b, head, hw, dim

        attn = torch.matmul(q, k.transpose(-1, -2).contiguous()) * self.scale # b, head, hw, hw
        attn = attn.softmax(dim=-1) 

        attn = self.dropout(attn)

        out = torch.matmul(attn, v) # b, head, hw, dim
        out = rearrange(out, 'b h n d -> b n (h d)') # b, hw, head*dim
        return self.to_out(out) # b, hw, dim_in
    

class linear_FFN(nn.Module):
    "Linear Feed Forward Network"
    def __init__(self, dim, hidden_dim, dropout = 0) -> None:
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)
            )

    def forward(self, x):
        return self.net(x)

class dwconv_FFN(nn.Module):
    "Depthwise conv based Feed Forward Network"
    def __init__(self, ch, k) -> None:
        super().__init__()
        self.gelu = nn.GELU()
        self.conv1 = nn.Conv2d(in_channels=ch, out_channels=ch * k, kernel_size=1)
        self.dwconv = nn.Conv2d(in_channels=ch * k, out_channels=ch * k, kernel_size=3, stride=1, padding=1, groups = ch * k)
        self.conv2 = nn.Conv2d(in_channels=ch * k, out_channels=ch, kernel_size=1)

    def forward(self, x):
        x = self.conv1
        x = self.gelu(x)
        x = self.dwconv(x)
        x = self.gelu(x)
        x = self.conv2(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, msa, ffn, dim, heads=8, dim_head=64, dropout=0) -> None:
        super().__init__()
        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)
        self.msa = msa(dim, heads, dim_head, dropout)
        self.ffn = ffn(dim, dim*2, dropout)

    def forward(self, x):
        x1 = self.layernorm1(x)
        x1 = self.msa(x1)
        x2 = x + x1
        x3 = self.layernorm2(x2)
        x3 = self.ffn(x2)
        out = x2 + x3

        return out


class ViT(nn.Module):
    def __init__(self, msa, ffn, pos_emb=None, mapping=nn.Linear, blocknum=1, down=[], nC=28, step=2, heads=8, dim_head=64, dropout=0):
        super().__init__()
        self.nC = nC
        dim = nC * (4 ** 2)
        self.step = step
        self.blocknum = blocknum
        
        self.emb = pos_emb(dim) if pos_emb is not None else nn.Identity()
        self.mapping = mapping(dim, dim)

        self.fusion = nn.Conv2d(in_channels=nC, out_channels=nC, kernel_size=3, stride=1, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(blocknum):
            if i in down:
                conv = nn.Conv2d(in_channels=dim, out_channels=dim*2, kernel_size=3, stride=2, padding=1, bias=False)
            elif blocknum - 1 - i in down:
                conv = nn.ConvTranspose2d(in_channels=dim, out_channels=dim//2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
            else:
                conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False)

            self.blocks.append(
                nn.ModuleList([TransformerBlock(msa, ffn, dim, heads, dim_head, dropout), conv])
            )

            if i in down:
                dim = dim * 2
            elif blocknum - 1 - i in down:
                dim = dim // 2 

    def initial_x(self, y, nC=28, step=2):
        """
        :param y: [b,1,256,310]
        :return: z: [b,nC,256,256]
        """

        bs, row, col = y.shape
        x = torch.zeros(bs, nC, row, row).to(y.device)
        for i in range(nC):
            x[:, i, :, :] = y[:, :, step * i:step * i + col - (nC - 1) * step]
        x = self.fusion(x)
        return x

    def forward(self, x, mask=None):
        """
        x : [B, H, W+d*(C-1)]
        out : [B, C, H, W]
        """
        x = self.initial_x(x) # B, C, H, W

        x = F.pixel_unshuffle(x, 4)
        b, c, h, w = x.shape

        x = rearrange(x, 'b c h w -> b (h w) c') # B, HW, C
        x = self.emb(x) # B, HW, C

        for i in range(self.blocknum):
            former, conv = self.blocks[i]
            if i == 0:
                x1 = former(x) # B, HW, C
            else:
                x1 = former(x1)

            x1 = rearrange(x1, 'b (h w) c -> b c h w', h=h)
            x1 = conv(x1) # B, C, H, W
            x1 = rearrange(x1, 'b c h w -> b (h w) c') # B, HW, C

        x1 = self.mapping(x1)

        out = x + x1 # B, HW, C
        out = rearrange(out, 'b (h w) c -> b c h w', h=h) # B, C, H, W

        out = F.pixel_shuffle(out, 4)

        return out
