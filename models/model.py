import math
import torch
import torch.nn as nn
import copy

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1, requires_grad=False).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a



def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=16, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)



        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class UnetBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        out_ch, ch_mult = config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps
        
        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))
       
        ch = config.model.ch_num
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        assert x.shape[2] == x.shape[3] == self.resolution

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1))
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Model(nn.Module):
    def __init__(self, config, betas, seq):
        super().__init__()
        num_block = config.diffusion.num_block
        self.pred_mean = config.training.train_type == "layer_v2"
        self.learn_alpha = config.diffusion.learn_alpha
        # 为每一个block计算config
        # block属性如果是[{value: xxx, num: yy}, {value: yyy, num: zz}, ...]
        list_value = {}
        for key, value in vars(config.model).items():
            if isinstance(value, list) and isinstance(value[0], dict):
                new_value = []
                for data in value:
                    new_value += [data['value']] * data['num']
                assert len(new_value) == num_block, f'num_block: {num_block} != {len(new_value)}'
                list_value[key] = new_value
        configs = []
        for i in range(num_block):
            new_config = copy.deepcopy(config)
            for key, value in list_value.items():
                setattr(new_config.model, key, value[i])
            configs.append(new_config)

        block_type = globals()[config.model.block_type]  
        self.models = nn.ModuleDict({
            str(key): block_type(config) 
            for key, config in zip(seq, configs)
        })
        self.seq = seq
        self.betas = betas

        if self.learn_alpha:
            self.embd_a = nn.Embedding(config.diffusion.num_diffusion_timesteps, 1)
            self.embd_b = nn.Embedding(config.diffusion.num_diffusion_timesteps, 1)
        pass
    
    def forward(self, x, t, last_t=None):
        et = self.models[str(t)](x)
        if self.pred_mean:
            # layer t 是输入t, 输出t-1
            et = self.get_x_next(et, x, t, last_t)
        return et

    def __getitem__(self, i):
        return self.models[str(i)]
    
    def get_x_next(self, et, x, i, j):
        # 直接预测均值
        n = x.size(0)
        t = (torch.ones(n, requires_grad=False) * i).to(x.device)
        next_t = (torch.ones(n, requires_grad=False) * j).to(x.device)    
        at = compute_alpha(self.betas, t.long())
        at_next = compute_alpha(self.betas, next_t.long())
        a = at_next.sqrt() / at.sqrt()
        b = (1 - at_next).sqrt() - (1 - at).sqrt() / at.sqrt() * at_next.sqrt()
        t = torch.ones(x.shape[0], device=x.device) * t
        if self.learn_alpha:
            embd_a =  self.embd_a(t.long())
            embd_b = self.embd_b(t.long())
            embd_a = embd_a.reshape((-1, 1, 1, 1))
            embd_b = embd_b.reshape((-1, 1, 1, 1))
            a, b = embd_a + a, embd_b + b
        x = a * x + b * et
        return x
    
    def sample(self, x):
        
        for i, j in zip(reversed(self.seq[1:]), reversed(self.seq[:-1])):        
            et = self.forward(x, i, j)
            if not self.pred_mean:
                x = self.get_x_next(et, x, i, j)
            else:
                x = et
        return x

"""
7000
Model(EMA): IS: 7.532(0.085), FID: 29.244
frechet_inception_distance: 33.23762 (-1)
frechet_inception_distance: 29.36446
8000
Model(EMA): IS: 7.531(0.082), FID: 29.382 
3000
Model(EMA): IS: 8.251(0.101), FID: 21.084
4000
Model(EMA): IS: 8.182(0.094), FID: 22.527
3000? (20)
Model(EMA): IS: 8.678(0.076), FID: 10.453
4000(50)
Model(EMA): IS: 8.188(0.126), FID: 13.217
4800(50)
Model(EMA): IS: 8.404(0.107), FID: 10.614
6000(50)
Model(EMA): IS: 8.558(0.108), FID:  9.456
7500(50)
Model(EMA): IS: 8.644(0.094), FID:  9.093
8700(50)
Model(EMA): IS: 8.711(0.096), FID:  8.929

1000(mean)
Model(EMA): IS: 4.772(0.032), FID: 96.514
"""