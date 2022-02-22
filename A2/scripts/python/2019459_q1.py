#!/usr/bin/env python
# coding: utf-8

# In[178]:


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import tensorflow as tf
import tensorflow.keras as keras
import scipy
from sympy import *


# In[301]:


def load_images(folder):
    images = []
    names = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.resize(img, (224, 224))
        if img is not None:
            images.append(img)
            names.append(filename)
    return images, names

def load_images2(folder):
    images = []
    for filename in os.listdir(folder):
        img = keras.preprocessing.image.load_img(os.path.join(folder,filename), target_size=(224,224))
        if img is not None:
            images.append(img)
    return images


# ### Non-DL saliency maps

# In[302]:


imgs, names = load_images('../../inputs/dd/')


# In[4]:


nondl = []

for image in imgs:
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(image)
    saliencyMap = (saliencyMap * 255).astype(int)
    nondl.append(saliencyMap)


# In[ ]:





# ### DL saliency maps

# code and model used from https://usmanr149.github.io/urmlblog/cnn/2020/05/01/Salincy-Maps.html

# In[5]:


model = keras.applications.VGG16(weights='imagenet')


# In[6]:


model.summary()


# In[7]:


imgs2 = load_images2('../../inputs/dd/')


# In[ ]:





# In[8]:


def dlsalmap(image):
    img = keras.preprocessing.image.img_to_array(image)
    img = img.reshape((1, *img.shape))
    y_pred = model.predict(img)
    
    images = tf.Variable(img, dtype=float)
    with tf.GradientTape() as tape:
        pred = model(images, training=False)
        class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
        loss = pred[0][class_idxs_sorted[0]]
    
    grads = tape.gradient(loss, images)
    dgrad_abs = tf.math.abs(grads)
    dgrad_max_ = np.max(dgrad_abs, axis=3)[0]
    arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
    grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)
    return grad_eval


# In[107]:


dlog = []

for image in imgs2:
    saliencyMap = dlsalmap(image)
    dlog.append(saliencyMap)


# In[108]:


dl = np.array(dlog)
dl = (dl * 255).astype(np.uint8)


# In[ ]:





# In[109]:


plt.imshow(dl[11], cmap='gray')


# In[110]:


plt.imshow(nondl[11], cmap='gray')


# In[111]:


plt.imshow(imgs2[11])


# In[ ]:





# ### Based on eye-test, non DL saliency maps are better than DL based saliency maps

# In[112]:


for i in range(len(dl)):
    dl[i] = np.array(dl[i], np.uint8)
    nondl[i] = np.array(nondl[i], np.uint8)


# ### Otsu thresholding

# In[114]:


threshdl = []
threshnon = []
omapdl = []
omapnon = []

for i in range(len(dl)):
    ot, omap = cv2.threshold(dl[i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshdl.append(ot)
    omapdl.append(omap)
    ot, omap = cv2.threshold(nondl[i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshnon.append(ot)
    omapnon.append(omap)


# In[ ]:





# ### Measure 1

# In[179]:


from scipy import stats
from scipy import integrate


# In[226]:


def meanstd(salmap, ot):
    fg = []
    bg = []
    for y in salmap:
        for x in y:
            if(x<ot):
                bg.append(x)
            else:
                fg.append(x)
    
    fmean = np.mean(fg)
    bmean = np.mean(bg)
    fstd = np.std(fg)
    bstd = np.std(bg)
    return fmean, bmean, fstd, bstd

def gauss1(x):
    return scipy.stats.norm.pdf(x, mu1, sigma1)

def gauss2(x):
    return scipy.stats.norm.pdf(x, mu2, sigma2)

def mingauss(x):
    return min(gauss1(x), gauss2(x))

def overlap(fmean, bmean, fstd, bstd):
    global mu1
    global mu2
    global sigma1
    global sigma2
    mu1 = fmean
    mu2 = bmean
    sigma1 = fstd
    sigma2 = bstd
    integral = integrate.quad(mingauss, 0, 256)
    return integral[0]


# In[231]:


mu1 = None
mu2 = None
sigma1 = None
sigma2 = None


# In[232]:


def phi(ls):
    gamma = 256
    term = np.log10(1 + 256*ls)
    return 1 / (1 + term)


# In[ ]:





# In[244]:


phisdl = []
phisnon = []

for i in range(len(dl)):
    a, b, c, d = meanstd(dl[i], threshdl[i])
    ls = overlap(a, b, c, d)
    measure = phi(ls)
    phisdl.append(measure)
    a, b, c, d = meanstd(nondl[i], threshnon[i])
    ls = overlap(a, b, c, d)
    measure = phi(ls)
    phisnon.append(measure)


# In[ ]:





# ### Measure 2

# In[274]:


output = cv2.connectedComponentsWithStats(omapnon[19], 4, cv2.CV_32S)

num_labels = output[0]
labels = output[1]
stats = output[2]
centroids = output[3]


# In[275]:


num_labels


# In[283]:


ars = stats[:, -1]
ars


# In[284]:


ars.sum()


# In[287]:


def getcumax(stats):
    areas = stats[:, -1]
    areas = sorted(areas, reverse=True)
    # areas[0] is assumed to be background
    total = sum(areas) - areas[0]
    largest = areas[1]
    return largest / total

def compstats(omap):
    output = cv2.connectedComponentsWithStats(omap, 4, cv2.CV_32S)

    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    centroids = output[3]

    cumax = getcumax(stats)
    numcomps = num_labels - 1
    return cumax, numcomps


# In[289]:


def psi(cu, num):
    return cu + (1-cu)/num


# In[311]:


psisdl = []
psisnon = []

for i in range(len(dl)):
    cu, num = compstats(omapdl[i])
    measure = psi(cu, num)
    psisdl.append(measure)
    cu, num = compstats(omapnon[i])
    measure = psi(cu, num)
    psisnon.append(measure)


# In[ ]:





# ### Quality measure

# In[327]:


qualdl = []
qualnon = []

for i in range(len(dl)):
    qualdl.append(phisdl[i] * psisdl[i])
    qualnon.append(phisnon[i] * psisnon[i])


# In[ ]:





# In[ ]:





# ### Creating csv

# In[328]:


data = {'Image name': names, 'phi(DL)': phisdl, 'phi(non-DL)': phisnon, 'psi(DL)': psisdl,
        'psi(non-DL)': psisnon, 'quality(DL)': qualdl, 'quality(non-DL)':qualnon}


# In[329]:


df = pd.DataFrame(data)
df


# In[330]:


df.to_csv('../../outputs/q1/quality_stats.csv')


# In[ ]:





# ### Mean quality scores

# In[331]:


df.describe()


# In[332]:


print('MEANS\n')
print('phi(DL):', df['phi(DL)'].mean())
print('phi(non-DL):', df['phi(non-DL)'].mean())
print('psi(DL):', df['psi(DL)'].mean())
print('psi(non-DL):', df['psi(non-DL)'].mean())
print('quality(DL):', df['quality(DL)'].mean())
print('quality(non-DL):', df['quality(non-DL)'].mean())


# In[ ]:





# In[ ]:




