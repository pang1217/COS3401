import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

def resize_img(img, percent):
    dim = (int(img.shape[1] * percent) , int(img.shape[0] * percent))
    return  cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def intermean(im, t):
    [hist, _] = np.histogram(im, bins=256, range=(0, 255))
    
    prob = 1.0*hist/np.sum(hist)
    w0 = np.sum(prob[:t]) + 0.00000001
    w1 = np.sum(prob[t:]) + 0.00000001
    
    u0 = np.sum(np.array([i for i in range(t)])*prob[:t])/w0
    u1 = np.sum(np.array([i for i in range(t,256)])*prob[t:])/w1
    print(t, u0, u1)
    if (u0 == 0.0):
        thr = u1
    elif (u1 == 0.0):
        thr = u0
    else:
        thr = (u0 +u1) / 2

    return thr.astype('int16')

if __name__ == '__main__':
    im = cv2.imread("./images/bank6.jpg", 0)
    im_flat = np.reshape(im,(im.shape[0]*im.shape[1]))
    t0 = 128
    flag = True
    Tlist = []
    Tlist.append(t0)
    while (flag):
        t1 = intermean(im, t0)
        Tlist.append(t1)
        if (math.fabs(t1 -t0) < 1):
            flag = False
        else:
            t0 = t1
    #print(Tlist)
    print(t1)
    plt.subplot(131)
    plt.imshow(im, cmap = 'gray')
    plt.subplot(132)
    plt.hist(im_flat, bins=256, range=(0, 255))
    plt.subplot(133)
    plt.imshow(im > t1, cmap = 'gray')
    plt.show()
