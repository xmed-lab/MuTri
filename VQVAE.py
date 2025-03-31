import torch
import torch.nn as nn

from vqvae3d.helper import BasicBlocks, DownSampleBlock, UpBlocks
from vqvae3d.mednext import Encoder_MedNext
from vqvae3d.decoder import Decoder
from vqvae3d.codebook import Codebook
import os

from torchsummary import summary

class Basic_Adapter(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.layers = nn.Sequential(
                        BasicBlocks(1, channel, channel*2),
                        BasicBlocks(1, channel*2, channel),
                        nn.Conv3d(channel, channel, 1)
                    )
    def forward(self, x):
        return self.layers(x)
    
class Adapter_New(nn.Module):
    
    def __init__(self, r=3, channel=32):
        super().__init__()
        
        self.layers = nn.Sequential(*[Basic_Adapter(channel) for i in range(r)])
        
        
    def forward(self, x):
        
        residual = x
        
        for layer in self.layers:
            x = residual + layer(x)
        
        return x

class MultiAdapter_New(nn.Module):   # Adapter
    
    def __init__(self,):
        super().__init__()
        print("nnew MultiAdapter_New")
        channel = 32
        r = 3
        self.enc_block_0 = Adapter_New(r, channel*1)
        self.enc_block_1 = Adapter_New(r, channel*2)
        self.enc_block_2 = Adapter_New(r, channel*4)
        self.enc_block_3 = Adapter_New(r, channel*8)
        
    def forward(self, x_lis):
        x_0, x_1, x_2, x_3 = x_lis
        
        x_0_denoise = self.enc_block_0(x_0) 
        
        x_1_denoise = self.enc_block_1(x_1) 
        
        x_2_denoise = self.enc_block_2(x_2) 
        
        x_3_denoise = self.enc_block_3(x_3)
        
        return [x_0_denoise, x_1_denoise, x_2_denoise, x_3_denoise]




class Adapter(nn.Module):
    
    def __init__(self, channel):
        super().__init__()
        self.down = BasicBlocks(1, channel, channel*2)
        self.up = BasicBlocks(1, channel*2, channel)
        self.out = nn.Conv3d(channel, channel, 1)
        
    def forward(self, x):
        
        residual = x
        x_down = self.down(x)
        x_up = self.up(x_down)
        x_up = self.out(x_up)
        
        return x_up + residual

class MultiAdapter(nn.Module):
    def __init__(self,):
        super().__init__()
        print("new MultiAdapter")
        channel = 32
        r = 6 #3
        self.enc_block_0 = nn.Sequential(
            Adapter(channel*1),
            Adapter(channel*1),
            Adapter(channel*1)
        )
        self.enc_block_1 = nn.Sequential(
            Adapter(channel*2),
            Adapter(channel*2),
            Adapter(channel*2)
        )
        self.enc_block_2 = nn.Sequential(
            Adapter(channel*4),
            Adapter(channel*4),
            Adapter(channel*4)
        )
        self.enc_block_3 = nn.Sequential(
            Adapter(channel*8),
            Adapter(channel*8),
            Adapter(channel*8)
        )
        
    def forward(self, x_lis):
        x_0, x_1, x_2, x_3 = x_lis
        
        x_0_denoise = self.enc_block_0(x_0) #+ x_0
        
        x_1_denoise = self.enc_block_1(x_1) #+ x_1
        
        x_2_denoise = self.enc_block_2(x_2) #+ x_2
        
        x_3_denoise = self.enc_block_3(x_3) #+ x_3
        
        return [x_0_denoise, x_1_denoise, x_2_denoise, x_3_denoise]


class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()
        #channels = [args.latent_dim//4, args.latent_dim//2, args.latent_dim]
        #books = [args.num_codebook_vectors * 1, args.num_codebook_vectors * 1, args.num_codebook_vectors]
        #resolutions = [args.imgsize//2, args.imgsize//4, args.imgsize//8]

        base_r = 256
        latent_dim = 256
        num_codebook_vectors = 128
        mode = "L"       

        channels = [latent_dim//8, latent_dim//4, latent_dim//2, latent_dim]
        books = [num_codebook_vectors * 1, num_codebook_vectors * 1, num_codebook_vectors * 2, num_codebook_vectors * 2]
        resolutions = [base_r, base_r//2, base_r//4, base_r//8]
        self.encoder = Encoder_MedNext(kernel_size=5, mode=mode).cuda()
        self.decoder = Decoder().cuda() 

        codebooks = [Codebook(channels[i], books[i], resolutions[i]) for i in range(len(channels))]
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

        #self.Adapter = MultiAdapter_New() 

    def forward(self, imgs, info=None, Pre =False):
        encoded_images = self.encoder(imgs)
        x_lis = []
        loss = []

        for i, quant_conv in enumerate(self.quant_convs):
            quant_conv_encoded_images = quant_conv(encoded_images[i])
            codebook_mapping, codebook_indices, q_loss = self.codebooks[i](quant_conv_encoded_images, info)
            post_quant_conv_mapping = self.post_quant_convs[i](codebook_mapping)
            x_lis.append(post_quant_conv_mapping)
            loss.append(q_loss)  
        
        #x_lis = self.Adapter(x_lis)
        
        decoded_images = self.decoder(x_lis)
        q_loss = loss[0] + loss[1] + loss[2] + loss[3]
        
        if Pre == False:
            return decoded_images, codebook_indices, q_loss
        else:
            return decoded_images, codebook_indices, q_loss,encoded_images, x_lis

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

    def save_network(self, network, network_label, epoch_label,opt,device):
        save_dir = os.path.join(opt.checkpoints_dir, opt.name) 
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_filename_encoder = '%s_encoder_%s.pth' % (epoch_label, network_label)
        save_filename_decoder = '%s_decoder_%s.pth' % (epoch_label, network_label)
        save_filename_codebook = '%s_codebook_%s.pth' % (epoch_label, network_label)

        save_path = os.path.join(save_dir, save_filename)
        save_path_encoder  = os.path.join(save_dir, save_filename_encoder)
        save_path_deconder = os.path.join(save_dir, save_filename_decoder)
        save_path_codebook = os.path.join(save_dir, save_filename_codebook)

        torch.save(network.encoder.cpu().state_dict(), save_path_encoder)
        torch.save(network.decoder.cpu().state_dict(), save_path_deconder)
        torch.save(network.codebooks.cpu().state_dict(), save_path_codebook)
        torch.save(network.cpu().state_dict(), save_path)
        network.to(device)  


if __name__ == '__main__':
    model = VQGAN().cuda()
    x0 = torch.randn(1, 1,128,128,48).cuda() 
    #decoded_images, codebook_indices, q_loss =  model(x0)
    # summary(model.encoder, (1,128,128,48))
    # print(decoded_images.shape)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"The model has {num_params:,} parameters.")
