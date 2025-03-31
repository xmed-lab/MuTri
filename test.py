import time
import os
from options.test_options import TestOptions
#from data.data_loader import CreateDataLoader
from data import create_dataset
from models import create_model
import util.util3d as util
#from util.visualizer3d import Visualizer
from util.visualizer3d import save_images

from pdb import set_trace as st
from util import html
import VQVAE
import torch
from collections import OrderedDict
import numpy as np
import math

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip


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
                out, _, __ = model(test_patch)
                #out = test_patch
                out = out[0, 0, ...].detach().cpu().numpy()
                score_map[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += out
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1

    score_map = score_map / cnt # [Z, Y, X]
    #score_map = np.transpose(score_map, (2,0,1))
    score_map = np.transpose(score_map, (0,2,1))
    score_map = torch.from_numpy(score_map).unsqueeze(0).unsqueeze(0)
    score_map = score_map.to(device)
    print(itr)
    #print(score_map.shape)
    #while(1):True 
    return score_map


#data_loader = CreateDataLoader(opt)
#dataset = data_loader.load_data()

dataset = create_dataset(opt, phase='test')


device = "cuda"
model = VQVAE.VQVAE().to(device) 

OCT_Type = 'B' # 'A' 'B' 

Lora_Flag = False  

if Lora_Flag == True: 
    mode_save_path = "/home/eezzchen/TransPro/checkpoints/transpro_OCT2OCTA_LoRa/best_net_B.pth"
    mode_save_lora_path = "/home/eezzchen/TransPro/checkpoints/transpro_OCT2OCTA_LoRa_PreB/best_LoRa_B.pth"
    #model.load_state_dict(torch.load(mode_save_path)) 
    # Load the pretrained checkpoint first
    model.load_state_dict(torch.load(mode_save_path), strict=False)
    # Then load the LoRA checkpoint
    model.load_state_dict(torch.load(mode_save_lora_path), strict=False)
else:
    mode_save_path = "/home/eezzchen/TransPro/checkpoints/transpro_B_OCT2OCTA_KD_MVP_Exp_2/best_net_B.pth"        
    model.load_state_dict(torch.load(mode_save_path)) 

#model = create_model(opt)
#model.setup(opt) 

#visualizer = Visualizer(opt) transpro_B_OCT2OCTA_Info_Pretrain
# create website

web_dir = os.path.join(opt.results_dir, opt.test_name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test

for i, data in enumerate(dataset):

    if i >= opt.num_test:
        break  
    input_data = data['A'].float()

    #input_data_A = data['A'].float()
    input_data_B = data['B'].float()

    input_data_model = np.transpose(input_data[0][0], (0, 2, 1))
    input_data_model = input_data_model.unsqueeze(0).unsqueeze(0)
    input_data_model = input_data_model.to(device)


    with torch.no_grad():
        fake_= test_single_case(model,input_data_model) #model(input_data) 
        
        real_A = input_data.to(device)
        real_B = input_data_B.to(device)

        #fake_proj = torch.mean(fake_,3)
        #real_proj = torch.mean(real_,3)

        fake_= util.tensor2im3d(fake_.data)
        real_A = util.tensor2im3d(real_A.data)
        real_B = util.tensor2im3d(real_B.data)

        #fake_proj = util.tensor2im(fake_proj.data)
        #real_proj = util.tensor2im(real__proj.data)

    #print("66666")
    if OCT_Type == 'A':
        visuals = OrderedDict([('fake_A', fake_), ('real_A', real_)])
        img_path = data['A_paths']
    else:
        visuals = OrderedDict([('real_A', real_A),('fake_B', fake_), ('real_B', real_B)])
        img_path = data['B_paths']
    #img_path = model.get_image_paths()
    #print("AAAA:",data['A_paths'])
    #print("BBBB:",data['B_paths'])
    print('process image... %s' % img_path)
    save_images(webpage, visuals, img_path)

print("Finish!!!!")
webpage.save()


#  CUDA_VISIBLE_DEVICES=0 python test.py --dataroot  /home/eezzchen/OCT2OCTA3M_3D  --name MuTri_3M_Stage_I --test_name MuTri_3M --model MuTri --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch --input_nc 1 --output_nc 1 --gpu_ids 0 --num_test 15200 --which_epoch 194 --load_iter 194 




# CUDA_VISIBLE_DEVICES=0 python test.py --dataroot  /home/eezzchen/OCT2OCTA6M_3D  --name MuTri_3M_Stage_I --test_name MuTri_6M_2 --model MuTri --netG unet_256 --direction AtoB --lambda_A 10 --lambda_C 5 --dataset_mode alignedoct2octa3d --norm batch --input_nc 1 --output_nc 1 --gpu_ids 0 --num_test 15200 --which_epoch 194 --load_iter 194