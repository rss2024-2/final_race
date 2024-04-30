import cv2
import numpy as np

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

lower_orange = np.array([4, 190, 60])
high_orange = np.array([15, 255, 255])
MIN_CONTOUR_AREA = 200

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def cd_color_segmentation(img, template):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	########## YOUR CODE STARTS HERE ##########
	# Convert the image to HSV
	blurred_image = cv2.GaussianBlur(img, (3,3), 0)
	blurred_image = cv2.erode(blurred_image, (3,3))
	blurred_image = cv2.dilate(blurred_image, (3,3))
	#use cv2.inRange to apply a mask to the image
	image_hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(image_hsv, lower_orange, high_orange)
	masked_image = cv2.bitwise_and(image_hsv,image_hsv,mask=mask)

	_, thresholded_image = cv2.threshold(mask, 40, 255,0)
	contours, _  = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	if len(contours) > 0:
		best_contour = max(contours, key=cv2.contourArea) # Choose contour of largest area
		x,y,w,h = cv2.boundingRect(best_contour)
		bounding_box = ((x,y), (x+w, y+h))
		return bounding_box


	return ((0,0),(0,0))

	
