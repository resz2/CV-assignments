#!/usr/bin/env python
# coding: utf-8

# ## Q3

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


video = cv2.VideoCapture('data/shahar_walk.avi')
frames = []

while(True):
    found, frame = video.read()
    if(not found):
        break
    frames.append(frame)

print('Number of frames in video:', len(frames))

median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
print('\n\nMedian frame:')
plt.imshow(median_frame)
plt.show()


# In[3]:


video = cv2.VideoCapture('data/shahar_walk.avi')
gray_median = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_median, cmap=plt.cm.gray)
plt.show()


# In[4]:


img_array = []

for frame in frames:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_frame, gray_median)
    _, diff = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
    vid_height, vid_width, layers = frame.shape
    size = (vid_width, vid_height)

    copy = frame.copy()
    contours, _ = cv2.findContours(diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        xc = x + w//2
        yc = y + h//2
        radius = max(w//2,h//2)
        cv2.circle(copy, (xc ,yc), radius, (0,0,255), 2)
    img_array.append(copy)
    cv2.imshow('frame', copy)
    # input integer (ex - 40) for video to run automatically, blank to run frame by frame
    key = cv2.waitKey(40)
    # press escape keystop execution
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()


# In[5]:


out = cv2.VideoWriter('output/bounded.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()


# In[ ]:





# In[ ]:




