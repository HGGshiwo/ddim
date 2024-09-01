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
    if in_channels % 16 == 0:
        num_groups = 16
    elif in_channels % 8 == 0:
        num_groups = 8
    else:
        num_groups = in_channels
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


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

    def forward(self, x, kv=None):
        h_ = x
        h_ = self.norm(h_)
        if kv is not None:
            h_, true_x = h_.chunk(2)
        q = self.q(h_)
        if kv is None:
            k = self.k(h_)
            v = self.v(h_)
        else:
            k = self.k(true_x)
            v = self.v(true_x)

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

        if kv is not None:
            h_ = torch.cat([h_, true_x], dim=0)
        
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

    def forward(self, x, true_x=None):
        assert x.shape[2] == x.shape[3] == self.resolution
        if true_x is not None:
            x = torch.cat([x, true_x], dim=0)
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h, true_x)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h, true_x)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1))
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, true_x)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if true_x is not None:
            return h[:h.size(0)//2]
        return h


class UnetBlock(_UnetBlock):
    def __init__(self, config, betas) -> None:    
        super().__init__(config)
        learn_alpha = config.diffusion.learn_alpha
        self.pred_mean = config.training.train_type == "layer_v2" 
        self.sample_block = SampleBlock(betas, learn_alpha)
        self.output_size = config.model.output_size
        self.in_ch = config.model.in_channels
        self.out_ch = config.model.out_ch
        self.detach = getattr(config.training, 'detach', False)
        
        if self.in_ch != 3:
            self.pixel_unshuffle = nn.PixelUnshuffle(2)
        if  self.out_ch != 3:
            self.pixel_shuffle = nn.PixelShuffle(2)
        
    def forward_with_shuffle(self, x, true_x=None):
        if self.in_ch != 3:
            x = self.pixel_unshuffle(x)
        et = super().forward(x, true_x)
        if self.out_ch != 3:
            et = self.pixel_shuffle(et)
        return et

    def forward(self, x, t, last_t=None, true_x=None):
        et = self.forward_with_shuffle(x, true_x)
        if self.pred_mean and last_t is not None:
            # layer t 是输入t, 输出t-1
            et = self.sample_block(et, x, t, last_t)
        return et
    
    def sample(self, x, i, j, true_x=None):
        # et = self.forward_with_shuffle(x, true_x)
        et = self.forward_with_shuffle(x) # 这里小心修改了，不再使用cross_attention
        if true_x is None:
            if self.detach:
                x = x.detach()
            x = self.sample_block(et, x, i, j)
        else:
            x = self.sample_block(et, true_x, i, j)
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
    
    
    def forward(self, x, t=None, last_t=None):
        et = self.models[str(t)](x, t, last_t)
        return et

    def __getitem__(self, i):
        return self.models[str(i)]
    
    def sample(self, x):
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
    Model: IS: 8.146(0.114), FID: 17.749
    
    scale:
    1600 
    Model(EMA): IS: 8.248(0.106), FID: 18.247
    2700
    Model(EMA): IS: 8.533(0.068), FID: 16.048
    3900
    Model(EMA): IS: 8.597(0.105), FID: 16.380
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
    Model: IS: 8.332(0.067), FID: 13.132 
    Model(EMA): IS: 8.755(0.100), FID:  8.814
100(use_time_embed):
    1600
    Model(EMA): IS: 8.858(0.055), FID:  9.743
    2100
    Model(EMA): IS: 9.208(0.113), FID:  6.522
    4100
    Model(EMA): IS: 9.316(0.094), FID:  5.644
    5200
    Model(EMA): IS: 9.185(0.078), FID:  5.404
    6600
    Model(EMA): IS: 9.237(0.095), FID:  5.328 
    
ddim pre_train
    frechet_inception_distance: 5.605276
    frechet_inception_distance: 3.981065 (不算最后一步)

250(use_time_embed)
    1300
    Model(EMA): IS: 8.572(0.097), FID:  8.769
    1900
    Model(EMA): IS: 8.728(0.103), FID:  7.363 
    3000 
    Model(EMA): IS: 8.909(0.089), FID:  6.321 
    4100
    Model(EMA): IS: 8.966(0.112), FID:  5.896
    4800

20:
loss v2:
    1000
    Model(EMA): IS: 4.077(0.040), FID:122.615
loss v2, scale:
    1000
    Model: IS: 2.693(0.020), FID:186.341
    Model(EMA): IS: 2.980(0.018), FID:164.365
    800
    Model(EMA): IS: 5.115(0.051), FID:102.984 
    Model: IS: 5.118(0.089), FID: 95.512
    2000
    Model: IS: 5.432(0.063), FID: 88.983    
    Model(EMA): IS: 5.838(0.064), FID: 77.194
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
    
best:
    Model(EMA): IS: 8.762(0.120), FID: 11.739 (428394012)
    Model(EMA): IS: 8.580(0.084), FID: 13.226 (252986612)
    Model(EMA): IS: 8.768(0.107), FID: 10.879 (336919884)
    x3:
    Model(EMA): IS: 8.627(0.083), FID: 12.033 (364173740)

first_layer32:
    1100?
    Model(EMA): IS: 8.563(0.091), FID: 12.578 
    2000
    Model(EMA): IS: 8.566(0.085), FID: 12.447
first_layer48:
    1100?
    Model(EMA): IS: 8.698(0.100), FID: 11.599
    1900
    Model(EMA): IS: 8.695(0.092), FID: 11.578
first_layer64:
    4000
    Model(EMA): IS: 8.774(0.105), FID: 11.410
first_layer72:
    3700
    Model(EMA): IS: 8.735(0.122), FID: 11.321 
    
layer_100
    6800
    Model(EMA): IS: 6.923(0.060), FID: 33.512
    10000
    Model(EMA): IS: 7.631(0.105), FID: 21.299 
    
    10692(950-990)
    Model(EMA): IS: 7.631(0.105), FID: 21.302
    11700(950-990)
    Model(EMA): IS: 7.622(0.102), FID: 21.277

    10700(2-6)
    Model(EMA): IS: 7.713(0.101), FID: 20.109
    11900(2-6)
    Model(EMA): IS: 7.729(0.111), FID: 20.011

    10200(7-11)
    Model(EMA): IS: 7.686(0.104), FID: 20.672
    10900(7-11)
    Model(EMA): IS: 7.690(0.103), FID: 20.647
    11500(7-11)
    Model(EMA): IS: 7.692(0.105), FID: 20.674

    10600(12-16)
    Model(EMA): IS: 7.667(0.103), FID: 20.923
    11300(12-16)
    Model(EMA): IS: 7.660(0.098), FID: 20.981
    12300(12-16)
    Model(EMA): IS: 7.651(0.098), FID: 21.016

    10660(17-21)
    Model(EMA): IS: 7.661(0.102), FID: 21.054
    11750(17-21)
    Model(EMA): IS: 7.653(0.099), FID: 21.136

    best
    Model(EMA): IS: 7.737(0.096), FID: 19.686
    Model(EMA): IS: 7.802(0.102), FID: 19.195
    Model(EMA): IS: 7.801(0.104), FID: 19.233
    Model(EMA): IS: 7.830(0.106), FID: 18.949
    Model(EMA): IS: 7.836(0.106), FID: 18.988
    Model(EMA): IS: 7.835(0.104), FID: 19.047
    Model(EMA): IS: 7.827(0.103), FID: 19.085
    Model(EMA): IS: 7.859(0.099), FID: 18.789
    Model(EMA): IS: 7.862(0.095), FID: 18.837
    """