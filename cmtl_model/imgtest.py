import os
import torch
import numpy as np
from cmtl_model.src.crowd_count import CrowdCounter
from cmtl_model.src import network
import cv2

def preprocess_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32, copy=False)
    ht = img.shape[0]
    wd = img.shape[1]
    ht_1 = (ht//4)*4
    wd_1 = (wd//4)*4
    img = cv2.resize(img,(wd_1,ht_1))
    return img.reshape((1,1,img.shape[0],img.shape[1]))
    
models_path = {'A' : './cmtl_model/final_models/cmtl_shtechA_204.h5', 'B' : './cmtl_model/final_models/cmtl_shtechB_768.h5'}

 
def getPrediction(img, part = 'A'):
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = False

  net = CrowdCounter()
          
  trained_model = os.path.join(models_path[part])
  network.load_net(trained_model, net)
  net.cuda()
  net.eval()
                     
  img = preprocess_img(img)
  density_map = net(img)
  
  density_map = density_map.data.cpu().numpy()
  et_count = np.sum(density_map)
  
  density_map = 255*density_map/np.max(density_map)
  density_map= density_map[0][0]
  #plt.imshow(density_map)

  return density_map, et_count
