import cv2
import numpy as np
 
def cv_show(img,msg):
    cv2.imshow(msg,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  
if __name__ == '__main__':
    img = cv2.imread("./images/cameraman.png", 0)
    height, width = img.shape[:2]
    
    #ty, tx = 20, 10 #height / 4, width / 4
    sx, sy = 0.5, 0.5
    #T = np.float32([[1, 0, tx], [0, 1, ty]])
    T = np.float32([[sx, 0, 0], [0, sy, 0]])
    new_img = cv2.warpAffine(img, T, (width, height))
    new_img = new_img.astype(np.uint8) 
    mul_img = cv2.hconcat([img, new_img])
    cv_show(mul_img,"the filtered image")
