import math
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F

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


class _UnetBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        out_ch, ch_mult = config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.model.input_size
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


class UnetBlock(_UnetBlock):
    def __init__(self, config, betas) -> None:
        super().__init__(config)
        learn_alpha = config.diffusion.learn_alpha
        self.pred_mean = config.training.train_type == "layer_v2" 
        self.sample_block = SampleBlock(betas, learn_alpha)
        self.output_size = config.model.output_size
        if config.model.upsamp_with_conv:
            self.up_sample = Upsample(3, True)
        else:
            self.up_sample = lambda x: F.interpolate(x, size=(self.output_size, self.output_size), mode="bicubic", align_corners=False)
        
    def _resize(self, x, size):
        if size != x.shape[2] or size != x.shape[3]:
            x = F.interpolate(x, size=(size, size), mode='bicubic', align_corners=False)
        return x
    
    def resize_input(self, x):
        # 训练辅助函数，缩放输入
        return self._resize(x, self.resolution)
    
    def resize_output(self, x):
        # 训练辅助函数，缩放输出
        return self._resize(x, self.output_size)
        
    def upsample_output(self, x):
        if x.shape[2] != self.output_size:
            x = self.up_sample(x)
        return x
    
    def forward(self, x, t, last_t=None):
        et = super().forward(x)
        if self.pred_mean:
            # layer t 是输入t, 输出t-1
            et = self.sample_block(et, x, t, last_t)
        et = self.upsample_output(et)
        return et
    
    def sample(self, x, i, j):
        x = self.resize_input(x)
        et = super().forward(x)
        x = self.sample_block(et, x, i, j)
        x = self.upsample_output(x)
        return x

class SampleBlock(nn.Module):
    def __init__(self, betas, learn_alpha) -> None:
        super().__init__()
        self.betas = betas
        self.learn_alpha = learn_alpha
        if learn_alpha:
            self.embd_a = nn.Parameter(torch.zeros(1, requires_grad=True)) 
            self.embd_b = nn.Parameter(torch.zeros(1, requires_grad=True)) 
        
    def forward(self, et, x, i, j):
         # 直接预测均值
        n = x.size(0)
        t = (torch.ones(n, requires_grad=False) * i).to(x.device)
        next_t = (torch.ones(n, requires_grad=False) * j).to(x.device)    
        at = compute_alpha(self.betas, t.long())
        at_next = compute_alpha(self.betas, next_t.long())
        a = at_next.sqrt() / at.sqrt()
        b = (1 - at_next).sqrt() - (1 - at).sqrt() / at.sqrt() * at_next.sqrt()
        if self.learn_alpha:
            embd_a =  self.embd_a.reshape((-1, 1, 1, 1))
            embd_b = self.embd_b.reshape((-1, 1, 1, 1))
            a, b = embd_a + a, embd_b + b
        x = a * x + b * et
        return x
        

class Model(nn.Module):
    def __init__(self, config, betas, seq):
        super().__init__()
        num_block = config.diffusion.num_block
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
            str(key): block_type(config, betas) 
            for key, config in zip(seq, configs)
        })
        self.seq = seq
        self.betas = betas
    
    
    def forward(self, x, t, last_t=None):
        et = self.models[str(t)](x, t, last_t)
        return et

    def __getitem__(self, i):
        return self.models[str(i)]
    
    def sample(self, x):
        x = self.models[str(self.seq[-1])].resize_input(x)
        for i, j in zip(reversed(self.seq[1:]), reversed(self.seq[:-1])):        
            x = self.models[str(i)].sample(x, i, j)                
        return x

"""
10:
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
20:
    3000? 
    Model(EMA): IS: 8.678(0.076), FID: 10.453
50:    
    4000
    Model(EMA): IS: 8.188(0.126), FID: 13.217
    4800
    Model(EMA): IS: 8.404(0.107), FID: 10.614
    6000
    Model(EMA): IS: 8.558(0.108), FID:  9.456
    7500
    Model(EMA): IS: 8.644(0.094), FID:  9.093
    8700
    Model(EMA): IS: 8.711(0.096), FID:  8.929
    10000
    Model(EMA): IS: 8.755(0.100), FID:  8.814
20:
loss v2:
    1000
    Model(EMA): IS: 4.077(0.040), FID:122.615
loss v2, scale:
    1000
    Model: IS: 2.693(0.020), FID:186.341
    Model(EMA): IS: 2.980(0.018), FID:164.365
loss v3:
    890
    Model(EMA): IS: 4.432(0.023), FID:108.831
    Model(EMA): IS: 4.772(0.032), FID: 96.514(不算最后一步)
    2000
    Model(EMA): IS: 8.487(0.083), FID: 10.401
    3000
    Model(EMA): IS: 8.705(0.098), FID: 10.461
loss v3, learn_alpha:
    300
    Model: IS: 6.881(0.097), FID: 33.209 
    700
    Model: IS: 7.367(0.062), FID: 25.280
    1000
    Model(EMA): IS: 7.156(0.062), FID: 32.943
    Model: IS: 7.676(0.101), FID: 24.219
    1600
    Model(EMA): IS: 8.174(0.084), FID: 13.286
    Model: IS: 7.921(0.124), FID: 19.502
    2100
    Model(EMA): IS: 8.384(0.127), FID: 10.051
    Model: IS: 7.912(0.130), FID: 19.594
loss v2, learn_alpha:
    Model(EMA): IS: 1.832(0.012), FID:440.912
"""