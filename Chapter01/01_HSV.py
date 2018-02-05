import cv2 

# loads and read an image from path to file
img =  cv2.imread('../figures/flower.png')

# convert the color to hsv 
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# displays previous image 
cv2.imshow("Image",hsv)

# keeps the window open untill a key is pressed
cv2.waitKey(0)

# clears all window buffers
cv2.destroyAllWindows()