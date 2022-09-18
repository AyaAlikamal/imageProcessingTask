import cv2
import numpy as np
import matplotlib.pyplot as plt
#Task1code:
img_1 = cv2.imread('cat1.jpg',0)
img_2 = cv2.imread('th1.jpg',0)
img_3 = cv2.imread('catwallpaper3.jpg',0)
img_4 = cv2.imread('cat2.jpg',0)
h1, w1 = img_1.shape[:2]
h2, w2 = img_2.shape[:2]
h3, w3 = img_3.shape[:2]
h4, w4 = img_4.shape[:2]
img_5 = np.zeros((max(h1, h2,h3,h4), w1+w2+w3+w4), dtype=np.uint8)
img_5[:h1, :w1] = img_1
img_5[:h2, w2:w1+w2] = img_2
img_5[:h3, w1+w2:w1+w2+w3] = img_3
img_5[:h4, w1+w2+w3:w1+w2+w3+w4] = img_4
cv2.imshow('Img_5',img_5)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Task2 first algorithem
img1 = cv2.imread('catwallpaper3.jpg',cv2.IMREAD_GRAYSCALE)          
img2 = cv2.imread('cat2.jpg',cv2.IMREAD_GRAYSCALE) 
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

#Task2 second algorithm
img1 = cv2.imread('cat2.jpg',cv2.IMREAD_GRAYSCALE)        
img2 = cv2.imread('catwallpaper 3.jpg',cv2.IMREAD_GRAYSCALE) 
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

#Task2 thired algorithm
img1 = cv2.imread('cat2.jpg',cv2.IMREAD_GRAYSCALE)         
img2 = cv2.imread('catwallpaper3.jpg',cv2.IMREAD_GRAYSCALE) 
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
matchesMask = [[0,0] for i in range(len(matches))]
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img3,),plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

#Task3 first algorithem with modefication
img1 = cv2.imread('catwallpaper3.jpg',cv2.IMREAD_GRAYSCALE)         
img2 = cv2.imread('cat2.jpg',cv2.IMREAD_GRAYSCALE) 
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:35],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

#Task2 questions:
#Brute-force Matching with ORB Descriptors
#it may take less time than the other matching ways as it work in specific destince 

#Task3 questions:
#Image smoothing is a LPF as it clear noise and making bluring for image
#2D convolution is LPF as it  clear noise and making bluring for image
#Gaussjan Blurring is LPF as it clear noise and making bluring for image
#Median Blurring is LPF as it clear noise and making bluring for image
#Bilateral Filtering is a HPS as it not only clear noise and making bluring for image, it also sharp the edges 

#Task4
image1 =cv2.imread('cat2.jpg',0)
image1[image1>255]=255
image1[image1<255]=0
print(image1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Task5
image_original = cv2.imread('cat1.jpg',0)
cv2.imshow('orignal image',image_original)
cv2.waitKey(0)
Sharpening_Filter=np.array([[-1,-1,-1],
                            [-1,9,-1],
                            [-1,-1,-1]])
sharpened_image = cv2.filter2D(image_original,-1,Sharpening_Filter)    
cv2.imshow('sharpened image',sharpened_image )                       
cv2.waitKey(0)
cv2.destroyAllWindows()

#Task6
img = cv2.imread('th.jpg',0)
canny= cv2.Canny(img,100,200)
titles =['th.jpg','canny']
images =[img,canny]
for i in range(2):
 plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
 plt.title(titles[i])
 plt.xticks([]),plt.yticks([])
 plt.show()
 cv2.waitKey(0)
 cv2.destroyAllWindows()
 #Task6 questions:
 #we choose the canny edge as it works on making the image pure as it required to pass on many filters as:
 #Noise reduction
 #Gradient Calculation
 #Non-maimum suppression
 #Double threshold
 #Edg Tracking b Hysteresis