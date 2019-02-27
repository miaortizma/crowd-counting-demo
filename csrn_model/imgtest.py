import cv2
import h5py
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as c
from keras.models import model_from_json

def load_model():
    # Function to load and return neural network model 
    json_file = open('./csrn_model/models/Model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("./csrn_model/weights/model_B_weights.h5")
    return loaded_model

def create_img(path):
    #Function to load,normalize and return image 
    print(path)
    im = Image.open(path).convert('RGB')
    
    im = np.array(im)
    
    im = im/255.0
    
    im[:,:,0]=(im[:,:,0]-0.485)/0.229
    im[:,:,1]=(im[:,:,1]-0.456)/0.224
    im[:,:,2]=(im[:,:,2]-0.406)/0.225


    im = np.expand_dims(im,axis  = 0)
    return im
  
def predict(path):
    #Function to load image,predict heat map, generate count and return (count , image , heat map)
    model = load_model()
    image = create_img(path)
    ans = model.predict(image)
    count = np.sum(ans)
    return count,image,ans
  
  
def preprocess_img(img):  
  img = img/255.0
    
  img[:,:,0]=(img[:,:,0]-0.485)/0.229
  img[:,:,1]=(img[:,:,1]-0.456)/0.224
  img[:,:,2]=(img[:,:,2]-0.406)/0.225

  img = np.expand_dims(img,axis  = 0)
  return img

def getPrediction(img, part = 'A'):
  json_file = open('./csrn_model/models/Model.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = model_from_json(loaded_model_json)
  model.load_weights('./csrn_model/weights/model_'+part+'_weights.h5')
  
  img = preprocess_img(img)
  ans = model.predict(img)
  count = np.sum(ans)
  ans = ans.reshape(ans.shape[1],ans.shape[2])
  return ans, count  


'''
ans, img, hmap = predict('data/part_B_final/test_data/images/IMG_1.jpg')


print(ans)
#Print count, image, heat map
plt.imshow(img.reshape(img.shape[1],img.shape[2],img.shape[3]))
plt.show()
hmap = hmap.reshape(hmap.shape[1],hmap.shape[2])
plt.imshow(hmap)
#plt.imshow(hmap.reshape(hmap.shape[1],hmap.shape[2]) , cmap = c.jet )
plt.show()
'''  
