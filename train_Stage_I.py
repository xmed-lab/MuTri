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
import math 

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
lr = 3e-4 #3e-4
OCT_Type = 'A'  # 'A' 'B'  
optimizer = optim.Adam(model.parameters(), lr=lr)
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


#for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
for epoch in range(opt.epoch_count, 2000 + 1):    
    epoch_start_time = time.time()
    iter_start_time = time.time()
    epoch_iter = 0
    #if "adap" in opt.name:
    #    model.update_weight_alpha()
    Train_MAE = 0
    Train_num = 0

    for i, data in enumerate(dataset):
        input_data = data[OCT_Type].float()
        input_data = np.transpose(input_data[0][0], (0, 2, 1))

        train_data, info = crop(input_data, h=64, l=256, train=True) 

        #print(train_data.shape)
        train_data = train_data.unsqueeze(0)
        train_data =train_data.to(device)
        info = torch.Tensor([info])
        info  = info.to(device)

        total_steps += opt.batch_size
        epoch_iter += opt.batch_size
        model.zero_grad()  
        out,_, latent_loss = model(train_data,info)

        recon_loss = criterion(out, train_data)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        cur_mae = MAE_(out.detach().cpu().numpy(),train_data.cpu().numpy())
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
        #print(loss)
    
    print("Train MAE:",Train_MAE/Train_num)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        #model.save('latest')
        model.save_network(model,OCT_Type, 'latest',opt,device)
    

    if epoch % opt.val_epoch_freq == 0: 
        device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        with torch.no_grad():
            MAE = 0 
            num = 0 
            for i, data in enumerate(val_dataset):
                gt_data = data[OCT_Type].float()
                input_data = np.transpose(gt_data[0][0], (0, 2, 1))
                input_data = input_data.unsqueeze(0).unsqueeze(0)
                input_data = input_data.to(device)

                fake_= test_single_case(model,input_data) # model(input_data) 


                mae = MAE_(fake_.detach().cpu().numpy(),gt_data.cpu().numpy())
                MAE += mae 
                num += 1 

            print ('Val MAE:',MAE/num)
            if MAE/num < global_mae:
                global_mae = MAE/num
                # Save best models checkpoints
                print('saving the current best model at the end of epoch %d, iters %d' % (epoch, total_steps))
                #model.save('best')
                #model.save(epoch)
                model.save_network(model, OCT_Type, 'best',opt,device)  
                print("saving best...")

            #visualizer.print_current_metrics(epoch, MAE/num)
          
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    # if epoch > opt.n_epochs:
    #    model.update_learning_rate()

