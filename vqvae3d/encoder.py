import torch.nn as nn
import torch
from .helper import BasicBlocks, DownSampleBlock, UpBlocks, NonLocalBlock, CNA
import os
import argparse
from collections import OrderedDict

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        #channels = [64, 128, 128, 256, 256, 512]
        channels = [32, 64, 128, 256, 512]
        num_res_blocks = 1
        resolution = args.imgsize
        attn_resolutions = [resolution // 2**(len(channels)-2)]
        self.patch_embed = CNA(1, channels[0], 3)

        self.neck = BasicBlocks(1, channels[0], channels[0])

        block_lis = []
        self.num_blocks = len(channels) - 2
        for i in range(self.num_blocks):
            in_channels = channels[i]
            out_channels = channels[i+1]
            lis = [BasicBlocks(num_res_blocks, in_channels, out_channels), DownSampleBlock(out_channels)]
            resolution //= 2
            #if resolution in attn_resolutions:
            #    lis.append(NonLocalBlock(out_channels))
            block_lis.append(("block_{}".format(i), nn.Sequential(*lis)))
        self.blocks = nn.Sequential(OrderedDict(block_lis))

        self.mid_blocks = nn.Sequential(
            BasicBlocks(1, channels[-2], channels[-1]),
            NonLocalBlock(channels[-1]),
            BasicBlocks(1, channels[-1], channels[-1]),
            CNA(channels[-1], args.latent_dim),
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.neck(x)
        x_lis = [x]
        for blk in self.blocks[:-1]:
            x = blk(x)
            x_lis.append(x)
        x = self.blocks[-1](x)
        x = self.mid_blocks(x)
        x_lis.append(x)
        return x_lis

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--imgsize', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--gpu', type=str, default="0", help='Which device the training is on')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    encoder = Encoder(args).cuda()
    H = 256
    bs = 1
    a = torch.randn(bs,1,H,H,H//4).cuda()
    y_lis = encoder(a)
    for y in y_lis:
        print(y.size())
    import time
    time.sleep(30)
    print(encoder)