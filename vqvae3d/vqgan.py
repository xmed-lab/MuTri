import torch
import torch.nn as nn
from .encoder import Encoder
from .mednext import Encoder_MedNext
from .decoder import Decoder
from .codebook import Codebook

class Adapter(nn.Module):
    def __init__(self, args):
        super(Adapter, self).__init__()



class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        #channels = [args.latent_dim//4, args.latent_dim//2, args.latent_dim]
        #books = [args.num_codebook_vectors * 1, args.num_codebook_vectors * 1, args.num_codebook_vectors]
        #resolutions = [args.imgsize//2, args.imgsize//4, args.imgsize//8]

        base_r = 256
        channels = [latent_dim//8, latent_dim//4, args.latent_dim//2, latent_dim]
        books = [num_codebook_vectors * 1, num_codebook_vectors * 1, args.num_codebook_vectors * 2, args.num_codebook_vectors * 2]
        resolutions = [base_r, base_r//2, base_r//4, base_r//8]
        self.encoder = Encoder(args).cuda() if args.encoder == "unet" else Encoder_MedNext(kernel_size=5, mode=args.mode).cuda()
        self.decoder = Decoder(args).cuda() 

        codebooks = [Codebook(args, channels[i], books[i], resolutions[i]) for i in range(len(channels))]
        self.codebooks = nn.Sequential(
                        *codebooks
                        ).cuda()

        quant_lis = [
            nn.Conv3d(channel, channel, 1) for channel in channels
        ]
        self.quant_convs = nn.Sequential(
                        *quant_lis
                        ).cuda()

        post_quant_lis = [
            nn.Conv3d(channel, channel, 1) for channel in channels
        ]
        self.post_quant_convs = nn.Sequential(
                        *post_quant_lis
                        ).cuda()

    def forward(self, imgs, info=None):
        encoded_images = self.encoder(imgs) 
        x_lis = [] 
        loss = [] 

        for i, quant_conv in enumerate(self.quant_convs):
            quant_conv_encoded_images = quant_conv(encoded_images[i])
            codebook_mapping, codebook_indices, q_loss = self.codebooks[i](quant_conv_encoded_images, info)
            post_quant_conv_mapping = self.post_quant_convs[i](codebook_mapping)
            x_lis.append(post_quant_conv_mapping)
            loss.append(q_loss)

        # if self.adapter:
        # x_lis = self.adapters(x_lis) + x_lis

        decoded_images = self.decoder(x_lis)
        q_loss = loss[0] + loss[1] + loss[2] + loss[3]
        
        return decoded_images, codebook_indices, q_loss

    def encode(self, imgs):
        encoded_images = self.encoder(imgs)
        quant_conv_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_images)
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        pt = torch.load(path)
        #print(pt)
        self.load_state_dict(pt)








