import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-5, affine=True)

    def forward(self, x):
        return self.gn(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv3d(channels, channels, 3, 2, 0)
        self.norm = GroupNorm(channels)
        self.act = Swish()

    def forward(self, x):
        pad = (0, 1, 0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.act(self.norm(self.conv(x)))


class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels

        self.gn = GroupNorm(channels)
        self.q = nn.Conv3d(channels, channels, 1, 1, 0)
        self.k = nn.Conv3d(channels, channels, 1, 1, 0)
        self.v = nn.Conv3d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv3d(channels, channels, 1, 1, 0)
        self.norm = GroupNorm(channels)
        self.act = Swish()

        print("out layer, Norm")

    def forward(self, x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w, d = q.shape

        q = q.reshape(b, c, h*w*d)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w*d)
        v = v.reshape(b, c, h*w*d)

        attn = torch.bmm(q, k)
        attn = attn * (int(c)**(-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w, d)
        A = self.proj_out(A)
        A = A + x
        return self.act(self.norm(A))


from monai.networks.blocks.dynunet_block import UnetResBlock, UnetUpBlock
from collections import OrderedDict
norm_params = ("GROUP", {"num_groups": 32, "affine": True})
act_params = ("swish", {"alpha": 1})

class BasicBlocks(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels):
        #https://docs.monai.io/en/latest/_modules/monai/networks/blocks/dynunet_block.html
        super(BasicBlocks, self).__init__()
        lis = []
        for i in range(num_blocks):
            lis.append(("{}_block".format(i), UnetResBlock(spatial_dims=3, in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, norm_name=norm_params, act_name=act_params)))
            in_channels = out_channels
        self.blocks = nn.Sequential(OrderedDict(lis))
    def forward(self, x):
        return self.blocks(x)
'''
basic = BasicBlocks(2, 32, 64)
a = torch.randn(2,32,24,24)
y = basic(a)
print(basic)
print(y.size())
'''

class UpBlocks(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels):
        #https://docs.monai.io/en/latest/_modules/monai/networks/blocks/dynunet_block.html
        super(UpBlocks, self).__init__()
        self.up = UnetUpBlock(spatial_dims=3, in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, upsample_kernel_size=2, norm_name=norm_params, act_name=act_params)
        lis = []
        for i in range(num_blocks-1):
            lis.append(("{}_block".format(i), UnetResBlock(spatial_dims=3, in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, norm_name=norm_params, act_name=act_params)))
        self.blocks = nn.Sequential(OrderedDict(lis))
    def forward(self, inp, skip):
        out = self.up(inp, skip)
        return self.blocks(out)
    
'''
up = UpBlocks(1, 64, 32)
inp = torch.randn(2,64,24,24)
skip = torch.randn(2,32,48,48)
y = up(inp, skip)
print(up)
print(y.size())
'''

class CNA(nn.Module):
    def __init__(self, in_channels, out_channels, k=3):
        super(CNA, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, k, 1, k//2)
        self.norm = GroupNorm(out_channels)
        self.act = Swish()
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out
    

class UpSampleBlock(nn.Module):
    def __init__(self, channels, k=3):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv3d(channels, channels, k, 1, k//2)
        self.norm = GroupNorm(channels)
        self.act = Swish()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        return self.act(self.norm(self.conv(x)))