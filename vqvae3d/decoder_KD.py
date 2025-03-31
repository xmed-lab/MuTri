import torch.nn as nn
import torch
from .helper import BasicBlocks, DownSampleBlock, UpBlocks, NonLocalBlock, CNA, UpSampleBlock
import os
import argparse
from collections import OrderedDict

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        print("using standard decoder")
        channels = [256, 128, 64, 32]
        #num_res_blocks = [1,1,1] #
        num_res_blocks = [3,3,3] #


        in_channels = channels[0]
        self.mid_blocks = nn.Sequential(
            CNA(in_channels, channels[0]),
            BasicBlocks(1, channels[0], channels[0]),
            #NonLocalBlock(channels[0]),
            BasicBlocks(1, channels[0], channels[0]),
        )

        block_lis = []
        self.num_blocks = len(channels) - 1
        for i in range(self.num_blocks):
            in_channels = channels[i]
            out_channels = channels[i+1]
            block_lis.append(("block_{}".format(i), UpBlocks(num_res_blocks[i], in_channels, out_channels)))

        self.blocks = nn.Sequential(OrderedDict(block_lis))

        self.out_blocks = nn.Sequential(
            BasicBlocks(1, channels[-1], channels[-1]),
            nn.Conv3d(channels[-1], 1, 3, 1, 1)
            )
        
    
    def forward(self, x_lis):
        x = x_lis[-1]
        x = self.mid_blocks(x)
        for bi, blk in enumerate(self.blocks):
            skip = x_lis[-2-bi]
            x = blk(x, skip)
        x = self.out_blocks(x)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--imgsize', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--gpu', type=str, default="1", help='Which device the training is on')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    decoder = Decoder(args).cuda()
    H = 128
    D = 48
    bs = 1
    x4 = torch.randn(bs, 512,H//16,H//16,D//16).cuda()
    x3 = torch.randn(bs, 256,H//8,H//8,D//8).cuda()
    x2 = torch.randn(bs, 128,H//4,H//4,D//4).cuda()
    x1 = torch.randn(bs, 64,H//2,H//2,D//2).cuda()
    x0 = torch.randn(bs, 32,H,H,D).cuda() 
    y = decoder([x0,x1,x2,x3,x4])
    print(y.size())
    import time
    time.sleep(30)
    print(args)