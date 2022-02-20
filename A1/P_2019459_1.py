#!/usr/bin/env python
# coding: utf-8

# ## Q1

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import defaultdict


# In[2]:


img = cv2.imread('data/leaf.png')
# pyplot is rgb so image is reversed before displaying
plt.imshow(img[:,:,::-1])


# ### using kmeans clustering to reduce number of colors to 85

# In[3]:


# 85 colors
k = 85

i = np.float32(img).reshape(-1,3)
condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.8)
ret,label,center = cv2.kmeans(i, k , None, condition, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
img85 = center[label.flatten()]
img85 = img85.reshape(img.shape)
plt.imshow(img85[:,:,::-1])


# In[4]:


cv2.imwrite('data/leaf85.png', img85)


# In[5]:


unique_colors = set()
freqs = defaultdict(int)

img2 = Image.open('data/leaf85.png')
w, h = img2.size
for x in range(w):
    for y in range(h):
        pixel = img2.getpixel((x, y))
        unique_colors.add(pixel)
        freqs[pixel] += 1

num_colors = len(unique_colors)
print('number of colors in processed image:', num_colors)


# In[6]:


pnum = w*h
print('total pixels:', pnum)


# In[ ]:





# ### Eqn 3

# In[7]:


colors = list(unique_colors)
print(colors)


# In[8]:


def cdist(c1, c2):
    ret = (c1[0] - c2[0])**2
    ret += (c1[1] - c2[1])**2
    ret += (c1[2] - c2[2])**2
    return np.sqrt(ret)


# In[9]:


sals = defaultdict(float)
minsal = np.inf
maxsal = -1

for c1 in colors:
    val = 0
    for c2 in colors:
        val += (freqs[c2]/pnum) * cdist(c1, c2)
    sals[c1] = val
    if(val > maxsal):
        maxsal = val
    if(val < minsal):
        minsal = val


# In[10]:


sals


# In[11]:


print('min saliency:', minsal, '\nmax saliency:',maxsal)


# In[12]:


scaled = defaultdict(int)
for color in colors:
    scaled[color] = (sals[color] - minsal) / (maxsal - minsal) * 255


# In[13]:


scaled


# In[ ]:





# In[14]:


salmap3 = np.zeros((h, w), np.uint8)

for x in range(w):
    for y in range(h):
        pixel = img2.getpixel((x, y))
        salmap3[y][x] = scaled[pixel]

plt.imshow(salmap3, cmap='gray')


# In[15]:


cv2.imwrite('output/leafmap.png', salmap3)


# In[ ]:





# In[ ]:





# ### Eqn 5

# In[16]:


treeimg = cv2.imread('data/BigTree.jpg')
# pyplot is rgb so image is reversed before displaying
plt.imshow(treeimg[:,:,::-1])


# In[17]:


# 85 colors
k = 85

i = np.float32(treeimg).reshape(-1,3)
condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1)
ret,label,center = cv2.kmeans(i, k , None, condition, 10, cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
tree85 = center[label.flatten()]
tree85 = tree85.reshape(treeimg.shape)
plt.imshow(tree85[:,:,::-1])


# In[18]:


cv2.imwrite('data/tree85.jpg', tree85)


# In[19]:


unique2 = set()
freqs2 = defaultdict(int)

tree = Image.open('data/tree85.jpg')
w, h = tree.size
for x in range(w):
    for y in range(h):
        pixel = tree.getpixel((x, y))
        unique2.add(pixel)
        freqs2[pixel] += 1

num_colors = len(unique2)
print('number of colors in processed image:', num_colors)


# In[ ]:





# In[ ]:





# ### we use the segmented image

# In[20]:


segimg = cv2.imread('data/segmented.jpg')
plt.imshow(segimg[:,:,::-1])


# In[ ]:





# In[ ]:




