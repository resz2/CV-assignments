#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
from fcmeans import FCM


# In[ ]:





# In[2]:


def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


# In[3]:


imgs = load_images('../../inputs/dd/')


# In[ ]:





# In[4]:


img = cv2.imread('../../inputs/15_19_s.jpg')
# pyplot uses rgb so image is reversed before displaying
plt.imshow(img[:,:,::-1])


# In[5]:


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap="gray")


# In[6]:


height = width = 256
scaled = cv2.resize(gray_img, (width, height), interpolation= cv2.INTER_LINEAR)
plt.imshow(scaled, cmap="gray")


# In[7]:


def rounder(img, pixel, i, j):
    ret = 0
    if(i<0 or j<0 or i>=height or j>=width):
        return ret
    val = img[i][j]
    if pixel==0 and val==0:
        return 0
    ret = round(min(pixel, val) / max(pixel, val))
    return int(ret)

def lbped(img, i, j):
    pixel = img[i][j]
    binary = []
    
    # modified lbp
    binary.append(rounder(img, pixel, i-1, j-1))
    binary.append(rounder(img, pixel, i-1, j))
    binary.append(rounder(img, pixel, i-1, j+1))
    binary.append(rounder(img, pixel, i, j+1))
    binary.append(rounder(img, pixel, i+1, j+1))
    binary.append(rounder(img, pixel, i+1, j))
    binary.append(rounder(img, pixel, i+1, j-1))
    binary.append(rounder(img, pixel, i, j-1))
    
    # converting to decimal
    ret = 0
    for bit in binary:
        ret = (ret << 1) | bit
    return ret


# In[8]:


def graysizer(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height = width = 256
    scaled = cv2.resize(gray_img, (width, height), interpolation= cv2.INTER_LINEAR)
    return scaled

def lbper(img):
    lbpimg = np.zeros_like(img)
    for i in range(0, height):
        for j in range(0, width):
            lbpimg[i][j] = lbped(img, i, j)
    return lbpimg


# In[9]:


lbpimg = lbper(scaled)


# In[10]:


plt.imshow(lbpimg, cmap="gray")


# In[ ]:





# In[11]:


histg = cv2.calcHist([lbpimg],[0],None,[256],[0,256])
histg = histg.astype(int)
plt.plot(histg)


# ## SPP

# In[12]:


# input is lbp image

def spp(img):
    feature = np.array((), int)
    size = 256
    while(size>63):
        i = j = 0
        while(i<256):
            while(j<256):
                patch = img[i:i+size, j:j+size]
                histo = cv2.calcHist([patch],[0],None,[256],[0,256])
                histo = histo.astype(int)
                arr = np.zeros(256, int)
                for k in range(256):
                    arr[k] = histo[k][0]
                feature = np.concatenate((feature, arr))
                j = j+size
            j = 0
            i = i+size
        size = size // 2
    return feature


# In[ ]:





# ### Performing spp to generate features for all images in dataset

# In[13]:


features = []

for image in imgs:
    feature = spp(lbper(graysizer(image)))
    features.append(feature)


# In[14]:


features = np.array(features)


# ### k decides the number of clusters

# In[15]:


k = 4

fcm = FCM(n_clusters=k)
fcm.fit(features)


# In[16]:


fcm_centers = fcm.centers
fcm_labels = fcm.predict(features)


# In[17]:


fcm_centers


# In[18]:


fcm_labels


# ### Writing images into subfolders

# In[19]:


# Run this cell to remove previously existing subfolders
shutil.rmtree('../../outputs/q2/')


# In[20]:


try:
    os.mkdir('../../outputs/q2')
except OSError as error:
    print(error)

for i in range(k):
    try:
        os.mkdir('../../outputs/q2/cluster{}'.format(i+1))
    except OSError as error:
        print(error)


# In[21]:


nums = [1]*k

for i in range(len(fcm_labels)):
    cv2.imwrite('../../outputs/q2/cluster{}/image{}.jpg'.format(fcm_labels[i]+1, nums[fcm_labels[i]]), imgs[i])
    nums[fcm_labels[i]] += 1


# In[ ]:





# In[ ]:




