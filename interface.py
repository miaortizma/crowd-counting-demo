# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 00:56:45 2019

@author: edwma
"""
import numpy as np
import tkinter as tk
import tkinter.filedialog
import cv2
from PIL import Image, ImageTk
import mcnn_model
import cmtl_model

class App:
    def __init__(self, window, cameraID = 1):
        self.window = window
        self.cameraID = cameraID
        tk.Label(window, text = 'Camera').grid(row = 1, column = 0)
        self.peopleCount = tk.StringVar()
        self.peopleCount.set('Predict')
        self.predLabel = tk.Label(window, textvariable = self.peopleCount).grid(row = 1, column = 1)
        
        window.title('Source> Camera')
        
        #Select between modes - dropdown
        optionList = ['Camera', 'File']
        self.currentSource = optionList[0]
        
        self.source = tk.StringVar()
        self.source.set(optionList[0])
        self.dropSource = tk.OptionMenu(window, self.source, *optionList, command = self.sourceSelection)
        self.dropSource.grid(row = 0,column = 0)
        
        
        #Select between models
        optionListModels = ['MCNN', 'CMTL']
        self.currentModel = optionListModels[0]
        self.model = tk.StringVar()
        self.model.set(optionListModels[0])
        self.dropModel = tk.OptionMenu(window, self.model, *optionListModels, command = self.modelSelection)
        self.dropModel.grid(row = 0, column = 1)
        
        #Select between shtech parts
        optionListPart = ['A', 'B']
        self.currentPart = optionListPart[0]
        self.part = tk.StringVar()
        self.part.set(optionListPart[0])
        self.dropPart = tk.OptionMenu(window, self.part, *optionListPart, command = self.partSelection)
        self.dropPart.grid(row = 0, column = 3)
        
        #To get initial width, height
        self.height,self.width,g = 360,360,3
        cap = cv2.VideoCapture(cameraID)
        ret,frame = cap.read()
        if(ret):
            self.height,self.width,g = frame.shape
        
        self.canvasCamera = tk.Canvas(window, width = self.width, height = self.height)
        self.canvasCamera.grid(row = 2, column = 0)
        self.canvasPredict= tk.Canvas(window, width = self.width, height = self.height)
        self.canvasPredict.grid(row = 2, column = 1)
        self.btnPredict = tk.Button(window, text = "Predict!", command = self.predict).grid(row = 3)
        
        self.window.mainloop()
    
    def modelSelection(self, opt):
      self.currentModel = opt
      
    def partSelection(self, opt):
      self.currentPart = opt
    
    def sourceSelection(self,opt):
        self.currentSource = opt
        
        if(opt == "File"):
            self.path = tkinter.filedialog.askopenfilename(filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
            if(not self.path):
                self.source.set("Camera")
                self.window.title("Source> Camera")
            else:
                self.window.title ("Source> "+self.path)
        else:
            self.window.title ("Source> "+self.path)
                
    def predict(self):
        cameraIMG = self.getCameraIMG()
        self.reloadCAM(cameraIMG)
        self.reloadPrediction(cameraIMG)
    
    def reloadPrediction(self,img):
      
        models = {'MCNN': mcnn_model.imgtest, 'CMTL' : cmtl_model.imgtest} 
        img, count= models[self.currentModel].getPrediction(img, self.currentPart)
        img = cv2.resize(img, (self.width,self.height), interpolation = cv2.INTER_CUBIC)
        
        self.photoPred = ImageTk.PhotoImage(image = Image.fromarray(img))
        self.canvasPredict.config(height = self.height, width = self.width)
        self.canvasPredict.create_image(0, 0, image=self.photoPred, anchor=tk.NW)
        self.peopleCount.set("Predict: "+str(count))
        
        
    def reloadCAM(self,img):
        self.height,self.width,g = img.shape
        self.photo = ImageTk.PhotoImage(image = Image.fromarray(img))
        self.canvasCamera.config(height = self.height, width = self.width)
        self.canvasCamera.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
    def getCameraIMG(self):
        if(self.currentSource == "Camera"):
            cap = cv2.VideoCapture(self.cameraID)
            ret,frame = cap.read()
            if(ret):
                cap.release()
                return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            else:
                return np.random.randint(255, size=(360,360,3),dtype=np.uint8)#default for NO_CAMERA
        
        return cv2.cvtColor(cv2.imread(self.path), cv2.COLOR_BGR2RGB)

App(tk.Tk(),cameraID = 0)
'''
class App:
    def __init__(self, window, window_title, image_path="descarga.jpg"):
        self.window = window
        self.window.title(window_title)
        # Load an image using OpenCV
        self.cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        
        # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
        self.height, self.width, no_channels = self.cv_img.shape
        # Create a canvas that can fit the above image
        self.canvas = tkinter.Canvas(window, width = self.width, height = self.height)
        self.canvas.pack()
        
        # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
        
        # Add a PhotoImage to the Canvas
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        
        # Button that lets the user blur the image
        self.btn_blur=tkinter.Button(window, text="Blur", width=50, command=self.blur_image)
        self.btn_blur.pack(anchor=tkinter.CENTER, expand=True)
        
        self.window.mainloop()
    
    # Callback for the "Blur" button
    def blur_image(self):
        self.cv_img = cv2.blur(self.cv_img, (3, 3))
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

App(tkinter.Tk(), "Tkinter and OpenCV")
'''
'''
import numpy as np
import matplotlib.pyplot as plt
import cv2

cap = cv2.VideoCapture(1) # video capture source camera (Here webcam of laptop) 
ret,frame = cap.read() # return a single frame in variable `frame`

if(ret):
    #cv2.imshow('img1',frame) #display the captured image
    cv2.imwrite('A.png',frame)

cap.release()
'''