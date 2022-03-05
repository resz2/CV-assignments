#!/usr/bin/env python
# coding: utf-8

# In[105]:


import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage import color
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.measure import regionprops
from scipy.spatial.distance import euclidean


# In[2]:


img = img_as_float(io.imread('../../inputs/0002.jpg'))


# In[170]:


seglist = []
nums = (50, 100, 200, 300, 500, 800)

for num in (50, 100, 200, 300, 500, 800):
    segments = slic(img, n_segments = num, sigma = 5)
    seglist.append(segments)


# In[6]:


for segments in seglist:
    plt.imshow(mark_boundaries(img, segments, color=(255, 0, 0)))
    plt.show()


# In[ ]:





# In[7]:


superlist = []

for segments in seglist:
    superpixels = color.label2rgb(segments, img, kind='avg')
    superlist.append(superpixels)
    plt.imshow(superpixels)
    plt.show()


# In[ ]:





# In[101]:


loclist = []

for segments in seglist:
    locations = []
    regions = regionprops(segments)
    for props in regions:
        cx, cy = props.centroid
        locations.append(np.array([cx, cy]))
    loclist.append(locations)


# In[93]:


colorlist = []

for i in range(len(superlist)):
    colors = [0] * len(loclist[i])
    supers = superlist[i]
    segs = seglist[i]
    
    for j in range(len(supers)):
        for k in range(len(supers[0])):
            segnum = segs[j][k]
            color = supers[j][k]
            colors[segnum-1] = color
    colorlist.append(colors)


# In[ ]:





# ## Computing Saliency Maps

# In[127]:


width = img.shape[0]
height = img.shape[1]
denom = np.sqrt(width**2 + height**2)

def locex(locs, i, j):
    power = -euclidean(locs[i], locs[j]) / denom
    return np.exp(power)

def salcalc(i, locs, colors):
    ret = 0
    for j in range(len(locs)):
        expo = locex(locs, i, j)
        term = euclidean(colors[i], colors[j]) * expo
        ret += term
    return ret


# In[ ]:





# In[134]:


sallist = []

for k in range(len(loclist)):
    salvals = []
    locs = loclist[k]
    colors = colorlist[k]
    for i in range(len(locs)):
        sal = salcalc(i, locs, colors)
        salvals.append(sal)
    sallist.append(salvals)


# In[160]:


salmaps = []

for k in range(len(sallist)):
    salmap = np.zeros_like(img)
    for i in range(width):
        for j in range(height):
            salmap[i][j] = sallist[k][seglist[k][i][j] - 1]
    # scaling to [0-1]
    salmap = (salmap - np.min(salmap)) / (np.max(salmap) - np.min(salmap))
    salmaps.append(salmap)


# In[ ]:





# In[172]:


for i in range(len(salmaps)):
    print('Saliency map with '+ str(nums[i]) +' superpixels')
    plt.imshow(salmaps[i])
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




