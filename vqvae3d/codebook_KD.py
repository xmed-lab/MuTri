import torch
import torch.nn as nn
import os
import argparse
import torch.nn.functional as F
import loralib as lora

class TriPlane(nn.Module):
    def __init__(self, base_r=448, base_spacing=0.8, dim=32):
        super(TriPlane, self).__init__()
        self.xy = nn.Parameter(torch.zeros(1, dim, base_r*2, base_r*2, 1))
        self.xz = nn.Parameter(torch.zeros(1, dim, base_r*2, 1, base_r*2))
        self.yz = nn.Parameter(torch.zeros(1, dim, 1, base_r*2, base_r*2))
        self.base_r = base_r
        self.base_spacing = base_spacing
        print("trip position with {} resolution and {} dimensions and multip".format(base_r, dim))

    def forward(self, shape, info):
        spacing, bx, by, bz = info[0]
        bx = int(bx * self.base_r + self.base_r); by = int(by * self.base_r + self.base_r); bz = int(bz * self.base_r + self.base_r)
        H, W, D = shape
        Hn = int(spacing/self.base_spacing*H); Wn = int(spacing/self.base_spacing*W); Dn = int(spacing/self.base_spacing*D)

        #print(bx, Hn, by, Wn, bz, Dn)
        data_xy = self.xy[..., bx:bx+Hn, by:by+Wn, :]
        data_xy = F.interpolate(data_xy, size=(H, W, 1), mode='trilinear')
        data_xz = self.xz[..., bx:bx+Hn, :, bz:bz+Dn]
        data_xz = F.interpolate(data_xz, size=(H, 1, D), mode='trilinear')
        data_yz = self.yz[..., :, by:by+Wn, bz:bz+Dn]
        data_yz = F.interpolate(data_yz, size=(1, W, D), mode='trilinear')

        return (data_xy + data_xz + data_yz) / 3


class Codebook(nn.Module):
    def __init__(self,  latent_dim, num_codebook_vectors, resolution=448):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = num_codebook_vectors
        print("codebooks: {}".format(self.num_codebook_vectors))
        self.latent_dim = latent_dim
        self.beta = 0.25 #args.beta

        # if self.hasposition:
        # #    self.positions = nn.Parameter(torch.randn(1, latent_dim, resolution, resolution) * .01)
        # #    print("position embedding", self.positions.size())
        #     self.trip = TriPlane(base_r=resolution, dim=latent_dim)

        self.embedding_key = nn.Embedding(self.num_codebook_vectors, latent_dim)
        self.embedding_key.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

        self.embedding_value = nn.Embedding(self.num_codebook_vectors, latent_dim)
        self.embedding_value.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

        self.drop = nn.Dropout(p=0.4)

    def forward(self, z, info=None):

        z = z.permute(0, 2, 3, 4, 1).contiguous()        
        z_flattened = z.view(-1, self.latent_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding_key.weight**2, dim=1) - \
            2*(torch.matmul(z_flattened, self.embedding_key.weight.t()))


        min_encoding_indices = torch.argmin(d, dim=1)

        z_q_key = self.embedding_key(min_encoding_indices).view(z.shape)
        z_q_value = self.embedding_value(min_encoding_indices).view(z.shape)


        loss = torch.mean((z_q_key.detach() - z)**2) + self.beta * torch.mean((z_q_key - z.detach())**2)  

        #z_q = z + (z_q - z).detach()
        #z_q = z + (z_q - z).detach() 
        z_q_value = z_q_value.permute(0, 4, 1, 2, 3)   
        return z_q_value, min_encoding_indices, loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--imgsize', type=int, default=256, help='Latent dimension n_z (default: 256)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors (default: 256)')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar (default: 0.25)')
    parser.add_argument('--gpu', type=str, default="0", help='Which device the training is on')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    decoder = Codebook(args).cuda()
    H = 16
    bs = 18
    a = torch.randn(bs,args.latent_dim,H,H).cuda()
    y, _, __ = decoder(a)
    print(y.size())
    import time
    time.sleep(30)
    print(args)