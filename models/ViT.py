import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


class ToPatchEmb(nn.Module):
    def __init__(self, img_size, patch_size, ch, dim) -> None:
        super().__init__()
        img_height, img_width = img_size
        patch_height, patch_width = patch_size
        patch_dim = ch * patch_height * patch_width
        patch_num = img_height * img_width // patch_height // patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
            PositionEmb(patch_num, dim)
        )

    def forward(self, x):
        # B, C, H, W --> B, (H/patch_h)*(W/patch_w), dim 
        x = self.to_patch_embedding(x)
        return x
    

class ToWindowEmb(nn.Module):
    def __init__(self, img_size, window_size, ch, dim):
        super().__init__()
        window_height, window_width = window_size
        self.to_window_embedding = nn.Sequential(
            Rearrange('b c (h w1) (w w2) -> (b h w) (w1 w2) c', w1=window_height, w2=window_width),
            nn.LayerNorm(ch),
            nn.Linear(ch, dim),
            nn.LayerNorm(dim),
            PositionEmb(window_height*window_width, dim)
        )

    def forward(self, x):
        x = self.to_window_embedding(x)
        return x

    
class PositionEmb(nn.Module):
    def __init__(self, patch_num, dim) -> None:
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, patch_num, dim))

    def forward(self, x):
        b, _, _ = x.shape
        return x + self.pos_emb

    
class LinearMapping(nn.Module):
    def __init__(self, patch_size, ch, dim) -> None:
        super().__init__()
        self.patch_height, self.patch_width = patch_size
        patch_dim = ch * self.patch_height * self.patch_width
        self.mapping = nn.Linear(dim, patch_dim)

    def forward(self, x, size):
        H, W =  size
        # B, (H/patch_h)*(W/patch_w), dim --> B, (H/patch_h)*(W/patch_w), C*patch_h*patch_w
        x = self.mapping(x) 
        
        # B, (H/patch_h)*(W/patch_w), C*patch_h*patch_w --> B, C, H, W
        x = rearrange(x, 'b (h w) (c p1 p2) -> b (h p1) (w p2) c', h=int(H/self.patch_height), w=int(W/self.patch_height), p1=self.patch_height, p2=self.patch_height)

        x = x.permute(0, 3, 1, 2)
        
        return x

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

        # attn = torch.matmul(q, k.transpose(-1, -2).contiguous()) * self.scale # b, head, hw, hw
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        attn = attn.softmax(dim=-1) 

        attn = self.dropout(attn)

        # out = torch.matmul(attn, v) # b, head, hw, dim
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)') # b, hw, head*dim
        return self.to_out(out) # b, hw, dim_in
    

class WindowAttention(nn.Module):
    def __init__(self, shift_size, dim, heads, dim_head, dropout):
        super().__init__()

        self.shift_size = shift_size
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

    def window_partition(self, x):
        return x
    
    def window_reverse(self, x):
        return x

    def forward(self, x):
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            x = x

        x = self.window_partition(x)
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x) # b, hw, dim_in -> b, hw, head*dim
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v)) # b, head, hw, dim

        # attn = torch.matmul(q, k.transpose(-1, -2).contiguous()) * self.scale # b, head, hw, hw
        attn = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        attn = attn.softmax(dim=-1) 

        attn = self.dropout(attn)

        # out = torch.matmul(attn, v) # b, head, hw, dim
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)') # b, hw, head*dim

        out = self.window_reverse(out)
        if self.shift_size > 0:
            out = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            out = out

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
        x3 = self.ffn(x3)
        out = x2 + x3

        return out


class ViT(nn.Module):
    def __init__(self, msa, ffn, embedding, mapping, blocknum=1, nC=28, step=2, dim=512, heads=8, dim_head=64, dropout=0):
        super().__init__()
        self.nC = nC
        self.step = step
        self.blocknum = blocknum
        
        self.embedding = embedding
        self.mapping = mapping

        self.blocks = nn.ModuleList([])

        for i in range(blocknum):
            self.blocks.append(
                TransformerBlock(msa, ffn, dim, heads, dim_head, dropout)
            )

    def initial_x(self, y, mask=None, nC=28, step=2):
        """
        :param y: [b,1,256,310]
        :return: z: [b,nC,256,256]
        """

        bs, row, col = y.shape
        x = torch.zeros(bs, self.nC, row, row).to(y.device)
        for i in range(self.nC):
            x[:, i, :, :] = y[:, :, self.step * i:self.step * i + col - (self.nC - 1) * self.step]
        
        return x

    def forward(self, x, mask=None):
        """
        x : [B, H, W+d*(C-1)]
        out : [B, C, H, W]
        """
        x = self.initial_x(x, mask) # B, C, H, W
        B, C, H, W = x.shape
        x0 = self.embedding(x) # B, (H/patch_h)*(W/patch_w), dim

        for i, transformerblock in enumerate(self.blocks):
            if i == 0:
                x1 = transformerblock(x0) # B, (H/patch_h)*(W/patch_w), dim
            else:
                x1 = transformerblock(x1)

        x1 = self.mapping(x1, (H, W)) # B, C, H, W

        out = x + x1 # B, C, H, W

        return out
