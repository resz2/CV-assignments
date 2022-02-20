#!/usr/bin/env python
# coding: utf-8

# ## Q2

# In[1]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


img = cv2.imread('data/horse.jpg')
plt.imshow(img)


# In[3]:


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap="gray")
gray_arr = np.array(gray_img)


# In[4]:


def find_means(threshold):
    gsum = gc = lsum = lc = 0

    for x in gray_arr:
        for y in x:
            if y < threshold:
                lsum += y
                lc += 1
            else:
                gsum += y
                gc += 1

    if(lc==0):
        less_mean = 0.0
    else:
        less_mean = lsum/lc
    more_mean = gsum/gc

    return more_mean, less_mean


# In[5]:


min_tss = np.inf
otsu_threshold = np.inf
tss_list = []


# In[6]:


thresholds = np.arange(256)

for t in thresholds:
    more_mean, less_mean = find_means(t)
    tss = 0
    for x in gray_arr:
        for y in x:
            if y > t:
                tss = tss + (more_mean - y)**2
            else:
                tss = tss + (less_mean - y)**2
    if tss < min_tss:
        min_tss = tss
        otsu_threshold = t
    tss_list.append((t, tss))


# In[ ]:





# In[7]:


otsu_threshold


# In[8]:


min_tss


# In[9]:


df = pd.DataFrame(tss_list)
df.columns = ['threshold', 'tss value']
df.to_csv('output/tss.csv')
df.head()


# In[10]:


mask = gray_img >= otsu_threshold
output_image = np.multiply(mask, 255)
plt.imshow(output_image, cmap="gray")
plt.show()


# In[11]:


cv2.imwrite('output/binary_horse.png', output_image)


# In[ ]:




