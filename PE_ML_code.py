#!/usr/bin/env python
# coding: utf-8

# In[4]:


import gdcm


# In[5]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import cv2
import os
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader,Dataset
import albumentations as albu
import functools
import torch
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from joblib import parallel_backend


# In[6]:


import torchvision


# In[7]:


patient_csv_path = 'patient.csv'


# In[8]:


df = pd.read_csv(test_csv_path)


# In[9]:


import pydicom
import cv2
import os, os.path as osp

from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

def pixel_array(d):
    return d.pixel_array
def load_dicom_array(f):
    dicom_files = glob.glob(osp.join(f, '*.dcm'))
    dicoms = [pydicom.dcmread(d) for d in dicom_files]
    M = float(dicoms[0].RescaleSlope)
    B = float(dicoms[0].RescaleIntercept)
    z_pos = [float(d.ImagePositionPatient[-1]) for d in dicoms]
    dicoms = np.asarray([d.pixel_array for d in dicoms])
    dicoms = dicoms[np.argsort(z_pos)]
    dicoms = dicoms * M
    dicoms = dicoms + B
    return dicoms, np.asarray(dicom_files)[np.argsort(z_pos)]


def window(img, WL=50, WW=350):
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    return X


def save_array(X, save_dir, file_names):
    for ind, img in enumerate(X):
        savefile = osp.join(save_dir, file_names[ind])
        if not osp.exists(osp.dirname(savefile)): 
            os.makedirs(osp.dirname(savefile))
        _ = cv2.imwrite(osp.join(save_dir, file_names[ind]), img)


def edit_filenames(files):
    dicoms = [f"{ind:04d}_{f.split('/')[-1].replace('dcm','jpg')}" for ind,f in enumerate(files)]
    series = ['/'.join(f.split('/')[-3:-1]) for f in files]
    return [osp.join(s,d) for s,d in zip(series, dicoms)]


# In[10]:


def get_training_augmentation(y=256,x=256):
    train_transform = [albu.RandomBrightnessContrast(p=0.3),
                           albu.VerticalFlip(p=0.5),
                           albu.HorizontalFlip(p=0.5),
                           albu.Downscale(p=1.0,scale_min=0.35,scale_max=0.75,),
                           albu.Resize(y, x)]
    return albu.Compose(train_transform)


formatted_settings = {
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],}
def preprocess_input(
    x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs
):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    return x

def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_validation_augmentation(y=256,x=256):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [albu.Resize(y, x)]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype('float32')

def norm(img):
    img-=img.min()
    return img/img.max()


# In[11]:


class config:
    model_name="resnet"
    batch_size = 1
    WORKERS = 0
    classes =14
    resume = False
    epochs = 1
    MODEL_PATH = 'log/cpt'
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)


# In[12]:


import time
class Lungs:
    def __init__(self, dicom_folders):
        self.dicom_folders = dicom_folders
        self.transforms = get_validation_augmentation()
        self.preprocessing = get_preprocessing(functools.partial(preprocess_input, **formatted_settings))
    def __len__(self): 
        return len(self.dicom_folders)
    def get(self, i):
        s = time.time()
        data = load_dicom_array(self.dicom_folders[i])
        image, files = data
        image_lung = np.expand_dims(window(image, WL=-600, WW=1500), axis=3)
        image_mediastinal = np.expand_dims(window(image, WL=40, WW=400), axis=3)
        image_pe_specific = np.expand_dims(window(image, WL=100, WW=700), axis=3)
        image = np.concatenate([image_mediastinal, image_pe_specific, image_lung], axis=3)
        rat = MAX_LENGTH / np.max(image.shape[1:])
#         image = zoom(image, [1.,rat,rat,1.], prefilter=False, order=1)
        names = [row.split(".dcm")[0].split("/")[-3:] for row in files]
        images = []
        for img in image:
            if self.transforms:
                img = self.transforms(image=img)['image']
            if self.preprocessing:
                img = self.preprocessing(image=img)['image']
            images.append(img)
        return torch.from_numpy(np.array(images)),names
    
    def __getitem__(self, i):
        try:
            return self.get(i)
        except Exception as e:
            print(e)
            return None,None

MAX_LENGTH = 256.

dicom_folders = list(('../input/rsna-str-pulmonary-embolism-detection/test/' + df.StudyInstanceUID + '/'+ df.SeriesInstanceUID).unique())
dset = Lungs(dicom_folders)


# In[ ]:





# In[13]:


test = DataLoader(dset, batch_size=config.batch_size, shuffle=False, num_workers=config.WORKERS)


# In[14]:


from efficientnet_pytorch import EfficientNet


# In[16]:


from efficientnet_pytorch import EfficientNet
cnn = EfficientNet.from_pretrained('efficientnet-b0',num_classes=9).cuda()
cnn = torch.nn.DataParallel(cnn)


# In[18]:


from torch import nn
from torch.nn import functional as F

class NeuralNet(nn.Module):
    def __init__(self,cnn, embed_size=1280, LSTM_UNITS=64, DO = 0.3):
        super(NeuralNet, self).__init__()
        self.cnn = cnn.module
        self.cnn.eval()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)
        self.linear2 = nn.Linear(LSTM_UNITS*2, LSTM_UNITS*2)

        self.linear_pe = nn.Linear(LSTM_UNITS*2, 1)
        self.linear_global = nn.Linear(LSTM_UNITS*2, 9)

    def forward(self, x, lengths=None):
        with torch.no_grad():
            embedding = self.cnn.extract_features(x)
            embedding = self.avgpool(embedding)
            b,f,_,_ = embedding.shape
            embedding = embedding.reshape(1,b,f)
        self.lstm1.flatten_parameters()
        h_lstm1, _ = self.lstm1(embedding)
        self.lstm2.flatten_parameters()
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        h_conc_linear1  = F.relu(self.linear1(h_lstm1))
        h_conc_linear2  = F.relu(self.linear2(h_lstm2))
        
        hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2

        output = self.linear_pe(hidden)
        output_global = self.linear_global(hidden.mean(1))
        return output,output_global


# In[19]:


model = NeuralNet(cnn).cuda()


# In[20]:


model.load_state_dict(torch.load('../input/cnn-lstm-v2/best(1).pth'))


# In[31]:


classes = [
 '{}_negative_exam_for_pe',
 '{}_rv_lv_ratio_gte_1',
 '{}_rv_lv_ratio_lt_1',
 '{}_leftsided_pe',
 '{}_chronic_pe',
 '{}_rightsided_pe',
 '{}_acute_and_chronic_pe',
 '{}_central_pe',
'{}_indeterminate']


# In[38]:



model.eval()
pred_df = []
with torch.no_grad():
    predicted = model(imgs_batch) +torch.flip(model(imgs_batch),[-1])+torch.flip(model(imgs_batch),[-2])
    predicted = torch.sigmoid(predicted/3).cpu().numpy()


# In[64]:


sub.to_csv('submission.csv', index=None)

