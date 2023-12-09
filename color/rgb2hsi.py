import cv2
import numpy as np
import matplotlib.pyplot as plt

def ShowImage(img, gray):
    plt.axis("off")
    if gray:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()
    
def normalization(shade, scale):
    mn, mx = np.amin(shade), np.amax(shade)
    return (((shade - mn)/(mx-mn))*scale).astype(np.uint8)
    
def rgb_to_hsi(img):
    R = img[:, :, 2].astype(np.float16)
    G = img[:, :, 1].astype(np.float16)
    B = img[:, :, 0].astype(np.float16)

    I = (R + G + B) / 3.0

    min_rgb = np.minimum.reduce([R, G, B])
    S = 1 - (3.0 / (R + G + B + 1e-6)) * min_rgb

    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B) * (G - B))
    den[den == 0] = 0.0000001  
    theta = np.arccos(num / den)

    H = theta
    H[B > G] = 2 * np.pi - H[B > G]
    H = np.degrees(H)
    return H,S,I

if __name__ == '__main__':
    img = cv2.imread('./images/hsv.png')
    h,s,i = rgb_to_hsi(img)
    #h is in the range(0,360), s is in the range (0,1), i is in the range (0,255), then algorithms need to normalization 
    hn = normalization(h, 255)
    sn = normalization(s, 255)
    iin = normalization(i, 255)
    ShowImage(hn, 1) 
    ShowImage(sn, 1)
    ShowImage(iin, 1)  
# If you wish to visualize or save HSI image, you may need to normalize or scale it appropriately.
