#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil


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


# In[4]:


img = cv2.imread('../../inputs/15_19_s.jpg')
# pyplot uses rgb so image is reversed before displaying
plt.imshow(img[:,:,::-1])


# In[ ]:





# In[5]:


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')


# ## Corner detection

# In[6]:


corners = cv2.goodFeaturesToTrack(gray, 40, 0.01, 10)
corners = np.int0(corners)

# marking the corners
for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

plt.imshow(img[:,:,::-1])


# ### Creating patches and lbp features

# In[7]:


def rounder(img, pixel, i, j):
    ret = 0
    height, width = img.shape
    if(i<0 or j<0 or i>=height or j>=width):
        return ret
    val = img[i][j]
    if val>=pixel:
        ret = 1
    return int(ret)

def lbped(img, i, j):
    pixel = img[i][j]
    binary = []
    
    # lbp
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
    return int(ret)

def lbper(img):
    lbpimg = np.zeros_like(img)
    height, width = img.shape
    for i in range(0, height):
        for j in range(0, width):
            lbpimg[i][j] = lbped(img, i, j)
    # border values are removed
    return lbpimg[1:height-1, 1:width-1]


# In[8]:


# radius = (size of patch - 1) / 2
rad = 5
# patch is 11x11


# In[9]:


search_feature = np.array((), int)
search_hist = np.zeros(256, int)
l, b = gray.shape

# only the 25 best and viable corners are chosen
count = 0
for corner in corners:
    if(count>24):
        break
    w, h = corner[0]
    if(h-rad<0 or w-rad<0 or h+rad>l or w+rad>b):
        continue
    count += 1
    patch = gray[h-rad:h+rad+1, w-rad:w+rad+1]
    lbpatch = lbper(patch)
    histo = cv2.calcHist([lbpatch],[0],None,[256],[0,256])
    histo = histo.astype(int)
    arr = np.zeros(256, int)
    for k in range(256):
        arr[k] = histo[k][0]
        search_hist[k] += arr[k]
    search_feature = np.concatenate((search_feature, arr))


# In[10]:


search_feature.shape


# In[11]:


search_hist


# ### Creating histograms for dataset images

# In[12]:


def histomaker(gray, corners):
    feature = np.array((), int)
    cumu_hist = np.zeros(256, int)
    l, b = gray.shape
    count = 0
    
    for corner in corners:
        if(count>24):
            break
        w, h = corner[0]
        if(h-rad<0 or w-rad<0 or h+rad>l or w+rad>b):
            continue
        count += 1
        patch = gray[h-rad:h+rad+1, w-rad:w+rad+1]
        lbpatch = lbper(patch)
        histo = cv2.calcHist([lbpatch],[0],None,[256],[0,256])
        histo = histo.astype(int)
        arr = np.zeros(256, int)
        for k in range(256):
            arr[k] = histo[k][0]
            cumu_hist[k] += arr[k]
        feature = np.concatenate((feature, arr))
    
    return feature, cumu_hist

def process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 30, 0.01, 10)
    corners = np.int0(corners)
    
    # [0] for method 1, [1] for method 2
    return histomaker(gray, corners)[0]


# In[13]:


histos = []
for image in imgs:
    histos.append(process(image))


# 

# In[14]:


distances = dict()

# search_feature for method 1, search_hist for method 2
for i in range(len(histos)):
    distances[i] = ((histos[i] - search_feature)**2).sum()


# In[15]:


distances


# ### Choosing k nearest images

# In[16]:


sorted_index = sorted(distances, key=distances.get)
for i in sorted_index:
    print(i, distances[i])


# In[17]:


k = 5

for i in range(k):
    plt.imshow(imgs[sorted_index[i]][:,:,::-1])
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




