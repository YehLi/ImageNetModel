import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint
import math
from mmcv.cnn import build_norm_layer

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x

class MergeFFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

        self.fc_proxy = nn.Sequential(
            nn.Linear(in_features, 2*in_features),
            nn.GELU(),
            nn.Linear(2*in_features, in_features),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x, semantics = torch.split(x, [H*W, x.shape[1] - H*W], dim=1)
        semantics = self.fc_proxy(semantics)
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        x = torch.cat([x, semantics], dim=1)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class DualAttention(nn.Module):
    def __init__(self, dim, num_heads, drop_path=0.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.q_proxy = nn.Linear(dim, dim)
        self.kv_proxy = nn.Linear(dim, dim * 2)
        self.q_proxy_ln = nn.LayerNorm(dim)

        self.p_ln = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path*1.0) if drop_path > 0. else nn.Identity()

        self.mlp_proxy = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * dim, dim),
        )
        self.proxy_ln = nn.LayerNorm(dim)

        self.qkv_proxy = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim*3)
        )

        layer_scale_init_value = 1e-6
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma3 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def selfatt(self, semantics):
        B, N, C = semantics.shape
        qkv = self.qkv_proxy(semantics).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        semantics = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return semantics

    def forward(self, x, H, W, semantics):
        semantics = semantics + self.drop_path(self.gamma1 * self.selfatt(semantics))

        B, N, C = x.shape
        B_p, N_p, C_p = semantics.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q_semantics = self.q_proxy(self.q_proxy_ln(semantics)).reshape(B_p, N_p, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv_semantics = self.kv_proxy(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kp, vp = kv_semantics[0], kv_semantics[1]
        attn = (q_semantics @ kp.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        _semantics = (attn @ vp).transpose(1, 2).reshape(B, N_p, C) * self.gamma2
        semantics = semantics + self.drop_path(_semantics)
        semantics = semantics + self.drop_path(self.gamma3 * self.mlp_proxy(self.p_ln(semantics)))

        kv = self.kv(self.proxy_ln(semantics)).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x, semantics

class MergeBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, drop_path=0., norm_layer=nn.LayerNorm, is_last=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention(dim, num_heads)

        if is_last:
            self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        else:
            self.mlp = MergeFFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.is_last = is_last
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        layer_scale_init_value = 1e-6
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x), H, W))

        if self.is_last:
            x, _ = torch.split(x, [H*W, x.shape[1] - H*W], dim=1)
            x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x), H, W))
        else:
            x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x), H, W))
        return x

class DualBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.attn = DualAttention(dim, num_heads, drop_path=drop_path)
        self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        layer_scale_init_value = 1e-6
        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
            requires_grad=True) if layer_scale_init_value > 0 else None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, semantics):
        _x, semantics = self.attn(self.norm1(x), H, W, semantics)
        x = x + self.drop_path(self.gamma1 * _x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x), H, W))
        return x, semantics   

class DownSamples(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class Stem(nn.Module):
    def __init__(self, in_channels, stem_hidden_dim, out_channels):
        super().__init__()
        hidden_dim = stem_hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2,
                      padding=3, bias=False),  # 112x112
            build_norm_layer(dict(type='BN', requires_grad=False), hidden_dim)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            build_norm_layer(dict(type='BN', requires_grad=False), hidden_dim)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            build_norm_layer(dict(type='BN', requires_grad=False), hidden_dim)[1],
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(hidden_dim,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.norm = nn.LayerNorm(out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class SemanticEmbed(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj_proxy = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, semantics):
        semantics = self.proj_proxy(semantics)
        return semantics

class DualVit(nn.Module):
    def __init__(self, 
        stem_width=32, 
        in_chans=3, 
        embed_dims=[64, 128, 320, 448],
        num_heads=[2, 4, 10, 14], 
        mlp_ratios=[8, 8, 4, 3], 
        drop_path_rate=0.15, 
        norm_layer=nn.LayerNorm, 
        depths=[3, 4, 6, 3],
        num_stages=4,
        pretrained=None
    ):
        super().__init__()
        self.depths = depths
        self.num_stages = num_stages

        self.sep_stage = 2
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        for i in range(num_stages):
            if i == 0:
                patch_embed = Stem(in_chans, stem_width, embed_dims[i])
            else:
                patch_embed = DownSamples(embed_dims[i - 1], embed_dims[i])

            if i == 0:
                self.q = nn.Parameter(torch.empty((64, embed_dims[0])), requires_grad=True)
                self.q_embed = nn.Sequential(
                    nn.LayerNorm(embed_dims[0]),
                    nn.Linear(embed_dims[0], embed_dims[0])
                )
                self.pool = nn.AvgPool2d((7,7), stride=7)
                self.kv = nn.Linear(embed_dims[0], 2*embed_dims[0])
                self.scale = embed_dims[0] ** -0.5
                self.proxy_ln = nn.LayerNorm(embed_dims[0])
                self.se = nn.Sequential(
                    nn.Linear(embed_dims[0], embed_dims[0]),
                    nn.ReLU(inplace=True),
                    nn.Linear(embed_dims[0], 2*embed_dims[0])
                )
                trunc_normal_(self.q, std=.02)
            else:
                semantic_embed = SemanticEmbed(
                    embed_dims[i - 1], embed_dims[i]
                )
                setattr(self, f"proxy_embed{i + 1}", semantic_embed)

            if i >= self.sep_stage:
                block = nn.ModuleList([
                    MergeBlock(
                        dim=embed_dims[i], 
                        num_heads=num_heads[i], 
                        mlp_ratio=mlp_ratios[i]-1 if (j%2!=0 and i==2) else mlp_ratios[i],
                        drop_path=dpr[cur + j], 
                        norm_layer=norm_layer,
                        is_last=((i==3) and (j == depths[i]-1)))
                for j in range(depths[i])])
            else:
                block = nn.ModuleList([
                    DualBlock(
                        dim=embed_dims[i], 
                        num_heads=num_heads[i], 
                        mlp_ratio=mlp_ratios[i],
                        drop_path=dpr[cur + j], 
                        norm_layer=norm_layer)
                for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])
            norm_proxy = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

            if i != num_stages - 1:
                setattr(self, f"norm_proxy{i + 1}", norm_proxy)

        self.apply(self._init_weights)
        self.init_weights(pretrained)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def forward_sep(self, x):
        B = x.shape[0]
        outs = []
        for i in range(self.sep_stage):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")

            x, H, W = patch_embed(x)
            C = x.shape[-1]
            if i == 0:
                x_down = self.pool(x.reshape(B, H, W, C).permute(0, 3, 1, 2))
                x_down_H, x_down_W = x_down.shape[2:]
                x_down = x_down.view(B, C, -1).permute(0, 2, 1)
                kv = self.kv(x_down).view(B, -1,  2, C).permute(2, 0, 1, 3)
                k, v = kv[0], kv[1]  # B, N, C

                self_q = self.q.reshape(8, 8, -1).permute(2, 0, 1)
                self_q = F.interpolate(self_q.unsqueeze(0), size=(x_down_H, x_down_W), mode='bicubic').squeeze(0).permute(1, 2, 0)
                self_q = self_q.reshape(-1, self_q.shape[-1])

                attn = (self.q_embed(self_q) @ k.transpose(-1, -2)) * self.scale   # q: 1, M, C,   k: B, N, C -> B, M, N
                attn = attn.softmax(-1)  # B, M, N
                semantics = attn @ v   # B, M, C
                semantics = semantics.view(B, -1, C)

                semantics = torch.cat([semantics.unsqueeze(2), x_down.unsqueeze(2)], dim=2)
                se = self.se(semantics.sum(2).mean(1))
                se = se.view(B, 2, C).softmax(1)
                semantics = (semantics * se.unsqueeze(1)).sum(2)
                semantics = self.proxy_ln(semantics)
            else:
                semantics_embed = getattr(self, f"proxy_embed{i + 1}")
                semantics = semantics_embed(semantics)

            for blk in block:
                x, semantics = blk(x, H, W, semantics)

            norm = getattr(self, f"norm{i + 1}")
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

            norm_semantics = getattr(self, f"norm_proxy{i + 1}")
            semantics = norm_semantics(semantics)
        return x, semantics, outs


    def forward_merge(self, x, semantics):
        B = x.shape[0]
        outs = []
        for i in range(self.sep_stage, self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, H, W = patch_embed(x)

            semantics_embed = getattr(self, f"proxy_embed{i + 1}")
            semantics = semantics_embed(semantics)

            x = torch.cat([x, semantics], dim=1)
            for blk in block:
                x = blk(x, H, W)

            if i != self.num_stages - 1:
                semantics = x[:, H*W:]
                x = x[:, 0:H*W]
                norm_semantics = getattr(self, f"norm_proxy{i + 1}")
                semantics = norm_semantics(semantics)

            norm = getattr(self, f"norm{i + 1}")
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs

    def forward(self, x):
        x, semantics, out1 = self.forward_sep(x)
        out2 = self.forward_merge(x, semantics)
        outs = out1 + out2
        return outs

@BACKBONES.register_module()
class dualvit_s(DualVit):
    def __init__(self, **kwargs):
        super(dualvit_s, self).__init__(
            stem_width=32, 
            embed_dims=[64, 128, 320, 448], 
            num_heads=[2, 4, 10, 14], 
            mlp_ratios=[8, 8, 4, 3],
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[3, 4, 6, 3],
            drop_path_rate=0.15, 
            pretrained=kwargs['pretrained']
        )

@BACKBONES.register_module()
class dualvit_b(DualVit):
    def __init__(self, **kwargs):
        super(dualvit_b, self).__init__(
            stem_width=64, 
            embed_dims=[64, 128, 320, 512], 
            num_heads=[2, 4, 10, 16], 
            mlp_ratios=[8, 8, 4, 3],
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            depths=[3, 4, 15, 3],
            drop_path_rate=0.15, 
            pretrained=kwargs['pretrained']
        )
