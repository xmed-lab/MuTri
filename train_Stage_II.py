import time
from options.train_options import TrainOptions
from util.visualizer3d import Visualizer
from data import create_dataset
from models import create_model
import torch
import numpy as np
import VQVAE
from argparse import ArgumentParser, Namespace
from scheduler import CycleScheduler,LRFinder
from torch import nn, optim
import util.util3d as util
from util.image_pool import ImagePool
from collections import OrderedDict
import random
import torch.nn.functional as F
import loralib as lora
import os
import itertools  
from einops import rearrange, reduce, repeat
import math



class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.proj_head = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_out)
        )
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.proj_head(x)
        x = self.l2norm(x)
        return x

class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class MultiProjector(nn.Module):
    def __init__(self,):
        super().__init__()
        print("Multi Projector")
        self.enc_block_0 = Embed(dim_in=32, dim_out=1024)
        self.enc_block_1 = Embed(dim_in=64, dim_out=1024)
        self.enc_block_2 = Embed(dim_in=128, dim_out=1024)
        self.enc_block_3 = Embed(dim_in=256, dim_out=1024)
        

    def forward(self, x_lis):
        x_0, x_1, x_2, x_3 = x_lis
        
        x_0_denoise = self.enc_block_0(x_0) #+ x_0
        
        x_1_denoise = self.enc_block_1(x_1) #+ x_1
        
        x_2_denoise = self.enc_block_2(x_2) #+ x_2
        
        x_3_denoise = self.enc_block_3(x_3) #+ x_3
        
        return [x_0_denoise, x_1_denoise, x_2_denoise, x_3_denoise]


def MAE_(fake,real):
    mae = 0.0
    mae = np.mean(np.abs(fake-real))
    return mae

def Norm(a):
    max_ = torch.max(a)
    min_ = torch.min(a)
    a_0_1 = (a-min_)/(max_-min_)
    return (a_0_1-0.5)*2


def crop(data, h=192, l=40, train=True):
    idx = 0 #np.random.randint(0,3, 1)[0]
    

    if idx == 0:
        sh, sw, sd = h, h, l
    elif idx == 1:
        sh, sw, sd = h, l, h
    else:
        sh, sw, sd = l, h, h
    B = 512
    data = data[:,:,:2*B]
    H, W, D = data.shape
    max_spacing = min(H/sh, W/sw, D/sd, 2.5)   # 2.0 / 0.8 = 2.5
    current_spacing = np.random.rand(1)[0] * (max_spacing - 1) + 1
    spacing = current_spacing * 0.8
    max_h, max_w, max_d = int(current_spacing*sh), int(current_spacing*sw), int(current_spacing*sd)

    bh = np.random.randint(0, H-max_h+1, 1)[0]
    bw = np.random.randint(0, W-max_w+1, 1)[0]
    bd = np.random.randint(0, D-max_d+1, 1)[0]
    #print(H,W,D,(bh, max_h), (bw, max_w), (bd, max_d))

    crop_data = data[bh:bh+max_h, bw:bw+max_w, bd:bd+max_d].unsqueeze(0).unsqueeze(0)

    crop_data = F.interpolate(crop_data, size=(sh, sw, sd), mode='trilinear')[0]

    absolute_bh = (bh - H//2) / B; absolute_bw = (bw - W//2) / B; absolute_bd = (bd - D//2) / B

    return crop_data, [spacing, absolute_bh, absolute_bw, absolute_bd]

def crop_2(data, data1, h=192, l=40, train=True):
    idx = 0 #np.random.randint(0,3, 1)[0]
    
    if idx == 0:
        sh, sw, sd = h, h, l
    elif idx == 1:
        sh, sw, sd = h, l, h
    else:
        sh, sw, sd = l, h, h
    B = 512
    data = data[:,:,:2*B]
    data1 = data1[:,:,:2*B]

    H, W, D = data.shape
    max_spacing = min(H/sh, W/sw, D/sd, 2.5)   # 2.0 / 0.8 = 2.5
    current_spacing = np.random.rand(1)[0] * (max_spacing - 1) + 1
    spacing = current_spacing * 0.8
    max_h, max_w, max_d = int(current_spacing*sh), int(current_spacing*sw), int(current_spacing*sd)

    bh = np.random.randint(0, H-max_h+1, 1)[0]
    bw = np.random.randint(0, W-max_w+1, 1)[0]
    bd = np.random.randint(0, D-max_d+1, 1)[0]
    #print(H,W,D,(bh, max_h), (bw, max_w), (bd, max_d))

    crop_data = data[bh:bh+max_h, bw:bw+max_w, bd:bd+max_d].unsqueeze(0).unsqueeze(0)
    crop_data1 = data1[bh:bh+max_h, bw:bw+max_w, bd:bd+max_d].unsqueeze(0).unsqueeze(0)

    # crop_data = data[bh:bh+max_h, bw:bw+max_w,   :].unsqueeze(0).unsqueeze(0)
    # crop_data1 = data1[bh:bh+max_h, bw:bw+max_w, :].unsqueeze(0).unsqueeze(0)

    crop_data = F.interpolate(crop_data, size=(sh, sw, sd), mode='trilinear')[0]
    crop_data1 = F.interpolate(crop_data1, size=(sh, sw, sd), mode='trilinear')[0]

    absolute_bh = (bh - H//2) / B; absolute_bw = (bw - W//2) / B; absolute_bd = (bd - D//2) / B

    return crop_data, crop_data1, [spacing, absolute_bh, absolute_bw, absolute_bd]



def test_single_case(model, image, stride_xy=64, stride_z=24, patch_size=(64, 64, 256)):

    itr = 0
    _, __, ww, hh, dd = image.size() 

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((ww,hh,dd))
    cnt = np.zeros((ww,hh,dd))

    for x in range(sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(sy):
            ys = min(stride_xy*y, hh-patch_size[1])
            for z in range(sz):
                itr += 1 
                zs = min(stride_z*z, dd-patch_size[2])
                test_patch = image[:, :, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]

            #print(test_patch.size(), (xs,xs+patch_size[0], ys,ys+patch_size[1], zs,zs+patch_size[2]))
                out, _, __ = model(test_patch)  #model(test_patch)
                #out = test_patch
                out = out[0, 0, ...].detach().cpu().numpy()
                score_map[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += out
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1

    score_map = score_map / cnt # [Z, Y, X]
    score_map = np.transpose(score_map, (0,2,1))
    score_map = torch.from_numpy(score_map).unsqueeze(0).unsqueeze(0)
    score_map = score_map.to(device)
    print(itr)
    #print(score_map.shape)
    #while(1):True 
    return score_map


opt = TrainOptions().parse()
dataset = create_dataset(opt, phase="train")  # create a dataset given opt.dataset_mode and other options
dataset_size = len(dataset)
print('#training images = %d' % dataset_size)


val_dataset = create_dataset(opt, phase="val") 
val_dataset_size = len(val_dataset)
print('#validation images = %d' % val_dataset_size)

#define VQ-VAE
#model = create_model(opt)
#model.setup(opt)       
device = "cuda"
model = VQVAE.VQVAE().to(device) 

model_OCT = VQVAE.VQVAE().to(device)
model_OCTA = VQVAE.VQVAE().to(device)
lr = 3e-4 #3e-4
OCT_Type = 'B' # 'A' 'B' 
Patch_list= [16,8,4,2]   
Patch_Num = 16 # 128/32 * 128/32
Weight= 0.5 
tau = 0.1

Expid = "Our"  
print("Weight:", Weight) 
print("tau:", tau)
print("Expid:",Expid) 


OCT_proj = MultiProjector().to(device)
OCT_Pre_proj = MultiProjector().to(device)
OCTA_Q_proj = MultiProjector().to(device)
OCTA_Pre_Q_proj = MultiProjector().to(device)

params = itertools.chain(model.parameters(), OCT_proj.parameters(), OCT_Pre_proj.parameters(), OCTA_Q_proj.parameters(), OCTA_Pre_Q_proj.parameters()) 
optimizer = optim.Adam(params, lr=lr) 

OCT_path = ''
model_OCT.load_state_dict(torch.load(OCT_path)) 

OCTA_path = ''
model_OCTA.load_state_dict(torch.load(OCTA_path)) 


scheduler_type = "cycle"
#scheduler_type = "LRFinder"
if scheduler_type == "cycle":
    scheduler = CycleScheduler(
        optimizer,
        lr,
        n_iter=len(dataset) * (opt.n_epochs + opt.n_epochs_decay + 1),
        momentum=None,
        warmup_proportion=0.05,
    )
elif scheduler_type == "LRFinder":
    scheduler = LRFinder(
        optimizer, 
        lr_min = lr*0.001, 
        lr_max = lr, 
        step_size =50, 
        linear=True
    )

#visualizer = Visualizer(opt)
total_steps = 0
val_total_iters = 0 

global_mae = 100000000000000
criterion = nn.MSELoss()
latent_loss_weight = 0.25
criterionL1= torch.nn.L1Loss()
criterion_mse = nn.MSELoss()


for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
    epoch_start_time = time.time()
    iter_start_time = time.time()
    epoch_iter = 0 
    #if "adap" in opt.name:
    #    model.update_weight_alpha()
    Train_MAE = 0
    Train_num = 0

    for i, data in enumerate(dataset):
        input_data_A = data['A'].float()
        input_data_B = data['B'].float()

        input_data_A = np.transpose(input_data_A[0][0], (0, 2, 1)) # 1 2 0
        input_data_B = np.transpose(input_data_B[0][0], (0, 2, 1))

        train_data_A,train_data_B,info = crop_2(input_data_A,input_data_B,  h=64, l=256, train=True)

        #print(train_data.shape)
        train_data_A = train_data_A.unsqueeze(0)
        train_data_A =train_data_A.to(device)

        train_data_B = train_data_B.unsqueeze(0)
        train_data_B =train_data_B.to(device)

        total_steps += opt.batch_size
        epoch_iter += opt.batch_size
        model.zero_grad()
        out,_, latent_loss,OCT_Fea,OCTA_QuanFea = model(train_data_A,Pre =True)
        recon_loss = criterion(out, train_data_B)
        latent_loss = latent_loss.mean()
        mutual_contrastive_loss_OCTA = 0.0 
        sim_consis_loss_OCTA = 0.0
        mutual_contrastive_loss_OCT  = 0.0   
        sim_consis_loss_OCT =0.0


        with torch.no_grad():
            _,_,_,Pre_OCT_Fea,_ = model_OCT(train_data_A,Pre =True)    
            Output_OCTA_Pre,_,_,_,Pre_OCTA_QuanFea = model_OCTA(train_data_B,Pre =True)   
            Pre_proj = torch.mean(Output_OCTA_Pre,4)
            #Pre_proj = Norm(Pre_proj) ## Norm 

        Proj_OCT_Fea = []
        Proj_Pre_OCT_Fea = []
        Proj_OCTA_QuanFea = []
        Proj_Pre_OCTA_QuanFea = []
        for ind in range(4): 
            Cur_OCT_Fea = rearrange(OCT_Fea[ind], 'b c (h p1) (w p2) z -> (b h w) c p1 p2 z', p1= Patch_list[ind] , p2= Patch_list[ind])
            Cur_Pre_OCT_Fea = rearrange(Pre_OCT_Fea[ind], 'b c (h p1) (w p2) z -> (b h w) c p1 p2 z', p1= Patch_list[ind] , p2= Patch_list[ind])
            Cur_OCTA_QuanFea = rearrange(OCTA_QuanFea[ind], 'b c (h p1) (w p2) z -> (b h w) c p1 p2 z', p1= Patch_list[ind] , p2= Patch_list[ind])
            Cur_Pre_OCTA_QuanFea = rearrange(Pre_OCTA_QuanFea[ind], 'b c (h p1) (w p2) z -> (b h w) c p1 p2 z', p1= Patch_list[ind] , p2= Patch_list[ind])

            Cur_OCT_Fea = F.adaptive_avg_pool3d(Cur_OCT_Fea,(1, 1, 1)).view(Patch_Num, -1)
            Cur_Pre_OCT_Fea = F.adaptive_avg_pool3d(Cur_Pre_OCT_Fea,(1, 1, 1)).view(Patch_Num, -1)
            Cur_OCTA_QuanFea = F.adaptive_avg_pool3d(Cur_OCTA_QuanFea,(1, 1, 1)).view(Patch_Num, -1)
            Cur_Pre_OCTA_QuanFea = F.adaptive_avg_pool3d(Cur_Pre_OCTA_QuanFea,(1, 1, 1)).view(Patch_Num, -1) 

            Proj_OCT_Fea.append(Cur_OCT_Fea)
            Proj_Pre_OCT_Fea.append(Cur_Pre_OCT_Fea)
            Proj_OCTA_QuanFea.append(Cur_OCTA_QuanFea)
            Proj_Pre_OCTA_QuanFea.append(Cur_Pre_OCTA_QuanFea)

        Proj_OCT_Fea = OCT_proj(Proj_OCT_Fea)
        Proj_Pre_OCT_Fea = OCT_Pre_proj(Proj_Pre_OCT_Fea)
        Proj_OCTA_QuanFea = OCTA_Q_proj(Proj_OCTA_QuanFea)
        Proj_Pre_OCTA_QuanFea = OCTA_Pre_Q_proj(Proj_Pre_OCTA_QuanFea)    

        loss_sim_proj = 0

        _proj = torch.mean(out,4)
        #_proj = Norm(_proj)  ## Norm  
        patch_proj_Pre = rearrange(Pre_proj, 'b c (h p1) (w p2)  -> (b h w) c p1 p2 ', p1= Patch_list[0] , p2= Patch_list[0])
        patch_proj  = rearrange(_proj, 'b c (h p1) (w p2) -> (b h w) c p1 p2 ', p1= Patch_list[0] , p2= Patch_list[0])  

        patch_proj_Pre = patch_proj_Pre.view(Patch_Num, -1)
        patch_proj = patch_proj.view(Patch_Num, -1)

        sim_consis_loss_proj= 0.0
        bases = patch_proj
        base_Pre =patch_proj_Pre
        k, c = bases.size()
        loss_sim_proj = 0
        num = 0
        for i in range(k - 1):
            for j in range(i + 1, k):
                num += 1
                simi = F.cosine_similarity(bases[i].unsqueeze(0), bases[j].unsqueeze(0).detach(), dim=1)
                simi = F.relu(simi)

                simi_Pre = F.cosine_similarity(base_Pre[i].unsqueeze(0), base_Pre[j].unsqueeze(0).detach(), dim=1)
                simi_Pre= F.relu(simi_Pre)

                loss_sim_proj += criterionL1(simi,simi_Pre)       

        sim_consis_loss_proj =  (loss_sim_proj/num)

        
        #****Coming Soon****# 
        #****This loss is still in progress for the journal version****# 
        mutual_contrastive_loss_OCTA = 0.0 
        mutual_contrastive_loss_OCT  = 0.0   


        loss = recon_loss + latent_loss_weight * latent_loss +(mutual_contrastive_loss_OCT + mutual_contrastive_loss_OCTA   + sim_consis_loss_proj )*Weight
        loss.backward()
        cur_mae = MAE_(out.detach().cpu().numpy(),train_data_B.cpu().numpy())
        Train_MAE += cur_mae
        Train_num += 1

        if  0:
            real_ = input_data
            fake_ = out

            fake_proj = torch.mean(fake_,3)
            real_proj = torch.mean(real_,3)

            fake_ = util.tensor2im3d(fake_.data)
            real_ = util.tensor2im3d(real_.data)

            fake_proj = util.tensor2im(fake_proj.data)
            real_proj = util.tensor2im(real_proj.data)
            For_Vis_data = OrderedDict([('fake_', fake_), ('real_', real_), ('fake_proj', fake_proj), ('real_proj', real_proj)])
            
            visualizer.display_current_results(For_Vis_data, epoch)

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

    
    print("Train MAE:",Train_MAE/Train_num)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save_network(model,OCT_Type, 'latest',opt,device)
    
    if epoch % opt.val_epoch_freq == 0: 
        device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')  
        with torch.no_grad():
            MAE = 0 
            num = 0 
            for i, data in enumerate(val_dataset):

                input_data_A = data['A'].float()
                input_data_B = data['B'].float()


                input_data_A = np.transpose(input_data_A[0][0], (0, 2, 1)) 
                train_data_A = input_data_A.unsqueeze(0).unsqueeze(0)
                train_data_A = train_data_A.to(device)
                train_data_B = input_data_B.to(device)
                fake_= test_single_case(model,train_data_A) 

                mae = MAE_(fake_.detach().cpu().numpy(),train_data_B.cpu().numpy())                
                MAE += mae
                num += 1 

            print ('Val MAE:',MAE/num)
            if MAE/num < global_mae:
                global_mae = MAE/num
                print('saving the current best model at the end of epoch %d, iters %d' % (epoch, total_steps))
                model.save_network(model, OCT_Type, 'best',opt,device)
                print("saving best...")

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time)) 
