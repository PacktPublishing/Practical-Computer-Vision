import cv2 

# loads and read an image from path to file
img =  cv2.imread('../figures/flower.png')

# displays previous image 
cv2.imshow("Image",img)

# keeps the window open untill a key is pressed
cv2.waitKey(0)

# clears all window buffers
cv2.destroyAllWindows()