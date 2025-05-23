from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import torch.nn.functional as F

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().detach().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def tensor2im3d(image_tensor):
    image_numpy = image_tensor[0][0].cpu().float().numpy()
    return image_numpy

def mask2im(image_tensor, imtype=np.uint8):
    image_numpy = F.one_hot(image_tensor.cpu().float().detach().argmax(dim=0), 2).permute(2, 0, 1).numpy()
    image_numpy = np.expand_dims(np.argmax(image_numpy, axis=0)*255,axis=2)
    return image_numpy.astype(imtype)

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image3d_fake(image_numpy, image_path, thesold = 252):  
    for i in range (image_numpy.shape[1]):
        img_arr = image_numpy[i,:,:]
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = np.tile(img_arr, (3, 1, 1))
        
        img_arr = (np.transpose(img_arr, (1, 2, 0)) + 1) / 2.0 * 255.0
        img_arr = img_arr.astype(np.uint8) 
        x,y,z = np.where(img_arr > thesold)
        img_arr[x,y,z] = 0  

        image_pil = Image.fromarray(img_arr)
        fn = image_path+str(i)+".png"
        image_pil.save(fn)

    #print("FAKE")
        
def save_image3d(image_numpy, image_path):
    for i in range (image_numpy.shape[1]):
        img_arr = image_numpy[i,:,:]
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = np.tile(img_arr, (3, 1, 1))
        
        img_arr = (np.transpose(img_arr, (1, 2, 0)) + 1) / 2.0 * 255.0
        img_arr = img_arr.astype(np.uint8)

        image_pil = Image.fromarray(img_arr)
        fn = image_path+str(i)+".png"
        image_pil.save(fn) 

    #print("Real")

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
