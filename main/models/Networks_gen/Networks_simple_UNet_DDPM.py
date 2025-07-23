import math
import torch
from torch import nn, einsum
from functools import partial
import torch.nn.functional as F
from einops import rearrange, reduce

from models.Networks_gen.Networks_UNet_DDPM import FCN, exists, default, Upsample, WeightStandardizedConv2d, Residual, LayerNorm, PreNorm, SinusoidalPosEmb, LearnedSinusoidalPosEmb


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 3, 2, 1)




# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, norm=nn.InstanceNorm2d, act=nn.ReLU):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = norm(dim_out)
        self.act = act()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm=nn.InstanceNorm2d, act=nn.ReLU):
        super().__init__()
        self.mlp = nn.Sequential(
            act(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, norm=nn.InstanceNorm2d, act=nn.ReLU)
        self.block2 = Block(dim_out, dim_out, norm=nn.InstanceNorm2d, act=nn.ReLU)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


def l2norm(t):
    return F.normalize(t, dim=-1)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, scale=10):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q, k = map(l2norm, (q, k))

        sim = einsum('b h d i, b h d j -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class simple_Unet_for_Improved_DDPM_class(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            condition_channels=3,
            condition=False,
            class_cond=None,
            learned_variance=False,
            learned_sinusoidal_cond=False,
            learned_sinusoidal_dim=16,
            norm='in',
            act=nn.ReLU,
            activate='none',
            seg_head=False,
            # seg_model=False
    ):
        super().__init__()

        self.class_cond = class_cond
        # determine dimensions
        self.channels = channels
        self.condition = condition
        self.seg_head = seg_head
        # self.seg_model = seg_model
        input_channels = channels + condition_channels if condition else channels
        self.activate = activate

        if norm == 'in':
            norm = partial(nn.InstanceNorm2d, affine=True)
        elif norm == 'bn':
            norm = nn.BatchNorm2d

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, norm=norm, act=act)

        # class embeddings
        time_dim = dim * 4
        if self.class_cond is not None:
            self.label_emb = nn.Embedding(class_cond, time_dim)

        self.learned_sinusoidal_cond = learned_sinusoidal_cond
        # time embeddings
        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            act(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

        if self.seg_head:
            self.seg_head_conv = FCN(in_channel=dim)

    def forward(self, x, time, x_self_cond=None, seg_cond=None, label=None):
        if self.condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            if seg_cond is not None:
                x = torch.cat((x_self_cond, seg_cond, x), dim=1)
            else:
                x = torch.cat((x_self_cond, x), dim=1)
        x = self.init_conv(x)
        r = x.clone()
        emb = self.time_mlp(time)
        if self.class_cond is not None:
            emb = emb + self.label_emb(label)

        h = []
        for block1, downsample in self.downs:
            x = block1(x, emb)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, emb)
        # self.feature = x
        x = self.mid_block2(x, emb)

        for block1, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, emb)
            x = upsample(x)
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, emb)
        if self.seg_head:
            return self.final_conv(x), self.seg_head_conv(x)
        elif self.seg_model:
            x = self.final_conv(x)
            return x, self.seg_model_conv(x)
        else:
            return self.final_conv(x)



class simple_Unet_without_Tembedding(nn.Module):
    def __init__(
            self,
            dim,
            in_channels=1,
            out_channels=1,
            dim_mults=(1, 2, 4, 8),
            norm='in',
            act=nn.ReLU
    ):
        super().__init__()
        if norm == 'in':
            norm = partial(nn.InstanceNorm2d, affine=True)
        elif norm == 'bn':
            norm = nn.BatchNorm2d

        self.init_conv = nn.Conv2d(in_channels, dim, 7, padding=3)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, norm=norm, act=act)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=None),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=None)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=None)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=None),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=None)
        self.final_conv = nn.Conv2d(dim, out_channels, 1)

    def forward(self, x):
        x = self.init_conv(x)
        r = x.clone()
        h = []
        for block1, downsample in self.downs:
            x = block1(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x)
        x = self.mid_block2(x)

        for block1, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x)
        return self.final_conv(x)


if __name__ == '__main__':
    net = simple_Unet_for_Improved_DDPM(dim=32, init_dim=None, out_dim=None, dim_mults=(1, 2, 4, 8), channels=1,
                                          condition_channels=2, condition=True, class_cond=2, norm='in',
                                          act=nn.ReLU, activate=None)
    # net = simple_Unet(dim=32,
    #         init_dim=None,
    #         out_dim=None,
    #         dim_mults=(1, 2, 4, 8),
    #         channels=1,
    #         condition_channels=5,
    #         condition=True)
    x = torch.rand(1, 1, 424, 424)
    cond = torch.rand(1, 2, 424, 424)
    label = torch.ones(1).long()
    t = torch.rand(1)
    out = net(x=x, time=t, x_self_cond=cond, label=label)
    print(out.size())