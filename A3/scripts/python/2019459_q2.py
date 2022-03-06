#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from skimage.util import img_as_float
from skimage import io
from scipy.spatial.distance import euclidean


# In[2]:


#img = img_as_float(io.imread('../../inputs/0002.jpg'))
img = cv2.imread('../../inputs/0002.jpg')


# In[3]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[4]:


x, y, z = img.shape


# In[5]:


# getting indices for appending
idxs = np.zeros((x, y, 2)).astype(np.uint32)
for i in range(x):
    for j in range(y):
        idxs[i][j] = [i, j]


# In[6]:


img2 = np.concatenate((img, idxs), axis=2)
img2 = img2.reshape(-1, 5)


# In[7]:


img2.shape


# In[ ]:





# In[ ]:





# In[8]:


def calcmean(labelimg, trainimg):
    meanvals = {}
    clusters = {}
    for i in range(x):
        for j in range(y):
            label = labelimg[i][j]
            if label in clusters:
                clusters[label].append(trainimg[i][j])
            else:
                clusters[label] = [trainimg[i][j]]

    for c in clusters:
        clusters[c] = np.array(clusters[c])
        meanvals[c] = clusters[c].mean(axis = 0)

    return meanvals

def dbscanner(samples, epss):
    cls = DBSCAN(min_samples=samples, eps=epss, algorithm ='auto', metric='euclidean')
    cls.fit(img2)
    labels = cls.labels_
    labelset = np.unique(labels)
    labelimg = labels.reshape(x, y)
    trainimg = img2.reshape(x, y, 5)
    meanvals = calcmean(labelimg, trainimg)

    for i in range(x):
        for j in range(y):
            if(labelimg[i][j] == -1):
                mindiff = np.inf
                for k in range(max(meanvals) + 1):
                    diff = euclidean(meanvals[k], trainimg[i][j])
                    if(diff < mindiff):
                        mindiff = diff
                        labelimg[i][j] = k

    result = img.copy()
    for i in range(x):
        for j in range(y):
            rgb = meanvals[labelimg[i][j]][:3]
            result[i][j] = rgb

    return result


# In[ ]:





# In[9]:


samplelist = [10, 20, 50, 100]
epslist = [5, 10, 15, 25]


# In[10]:


results = []

for sample in samplelist:
    for eps in epslist:
        print('DBSCAN with samples = {} and eps = {}'.format(sample, eps))
        results.append(dbscanner(sample, eps))


# In[ ]:





# In[11]:


i = 0

for sample in samplelist:
    for eps in epslist:
        print('DBSCAN with samples = {} and eps = {}'.format(sample, eps))
        plt.imshow(cv2.cvtColor(results[i], cv2.COLOR_BGR2RGB))
        plt.show()
        i += 1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




