import os
import torch
import numpy as np
from mcnn_model.src.crowd_count import CrowdCounter
from mcnn_model.src import network
import cv2

def preprocess_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32, copy=False)
    ht = img.shape[0]
    wd = img.shape[1]
    ht_1 = (ht//4)*4
    wd_1 = (wd//4)*4
    img = cv2.resize(img,(wd_1,ht_1))
    img = img.reshape((1,1,img.shape[0],img.shape[1]))
    return img

def load_img(img_path):
    img = cv2.imread(img_path,0)
    return preprocess_img(img)

models_path = {'A' : './mcnn_model/final_models/mcnn_shtechA_660.h5', 'B' : './mcnn_model/final_models/mcnn_shtechB_110.h5'}


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
  
  return density_map,et_count

#getPrediction('pito')