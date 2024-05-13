import cv2
import numpy as np
from scipy.spatial import distance

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

def nothing(x):
	pass

cv2.namedWindow("Trackbars")
cv2.setWindowProperty("Trackbars", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#print("window made")

# Now create 6 trackbars that will control the lower and upper range of 
# H,S and V channels. The Arguments are like this: Name of trackbar, 
# window name, range,callback function. For Hue the range is 0-179 and
# for S,V its 0-255.
cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

cv2.setTrackbarPos("L - H", "Trackbars", 0)
cv2.setTrackbarPos("L - S", "Trackbars", 0)
cv2.setTrackbarPos("L - V", "Trackbars", 221)
cv2.setTrackbarPos("U - H", "Trackbars", 100)
cv2.setTrackbarPos("U - S", "Trackbars", 30)
cv2.setTrackbarPos("U - V", "Trackbars", 255)



# Create a window named trackbars.
def cv():
	while True:

		l_h = cv2.getTrackbarPos("L - H", "Trackbars")
		l_s = cv2.getTrackbarPos("L - S", "Trackbars")
		l_v = cv2.getTrackbarPos("L - V", "Trackbars")
		u_h = cv2.getTrackbarPos("U - H", "Trackbars")
		u_s = cv2.getTrackbarPos("U - S", "Trackbars")
		u_v = cv2.getTrackbarPos("U - V", "Trackbars")

		lower_orange = np.array([l_h, l_s, l_v])
		high_orange = np.array([u_h, u_s, u_v])
		#print(lower_orange, high_orange)

		key = cv2.waitKey(1)
		if key == 27:
			break
		# cv2.waitKey(0)
		return(lower_orange, high_orange)


MIN_CONTOUR_AREA = 200




# def image_print(img):
# 	"""
# 	Helper function to print out images, for debugging. Pass them in as a list.
# 	Press any key to continue.
# 	"""
# 	cv2.imshow("image", img)
# 	cv2.waitKey(0)


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
	lower_orange = cv()[0]
	high_orange = cv()[1]
	
	blurred_image = cv2.GaussianBlur(img, (3,3), 0)
	#use cv2.inRange to apply a mask to the image
	image_hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(image_hsv, lower_orange, high_orange)

	black_image = np.zeros_like(image_hsv, dtype=np.uint8)

# Set the pixels within the orange range to white in the black image using the mask
	black_image[mask != 0] = [255, 255, 255]
	#image_print(black_image)


	masked_image = cv2.bitwise_and(image_hsv,image_hsv,mask=mask)

	_, thresholded_image = cv2.threshold(mask, 40, 255,0)
	#blurred_image = cv2.erode(blurred_image, (3,3))
	kernel = np.ones((3, 3), np.uint8)
	thresholded_image = cv2.dilate(thresholded_image, kernel)
	contours, _  = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	if len(contours) > 2:
		image_center = (int(thresholded_image.shape[0]/2), int(thresholded_image.shape[1]/2))
		contours_list = dict()
		for c in contours:
			M = cv2.moments(c)
			center_X = int(M["m10"]/ M["m00"])
			center_Y = int(M["m10"]/ M["m00"])
			contour_center = (center_X, center_Y)
			distances_to_center = (distance.euclidean(image_center, contour_center))
			contours_list.append({'contour': c, 'distance': distances_to_center })
		sorted_contours= sorted(contours_list, key=cv2.contourArea, reverse=True)
		box_1 = sorted_contours[0]
		x1,y1,w1,h1 = cv2.boundingRect(box_1)
		center = (int(x1+w1/2), int(y1+h1/2))
		cv2.rectangle(black_image, (x1, y1), (x1+w1, y1+h1), (0,255,0), 2)

		if (len(sorted_contours)>1):
			box_2 = sorted_contours[1]
			x2,y2,w2,h2 = cv2.boundingRect(box_2)
			cv2.rectangle(black_image, (x2, y2), (x2+w2, y2+h2), (0,255,0), 2)
			center_2 = (x2+w2/2, y2+h2/2)
			center = (int((center[0]+center_2[0])/2), int((center[1]+center_2[1])/2))
			cv2.circle(black_image, center, 10, (255,0,0),2)
		best_contour = max(contours, key=cv2.contourArea) # Choose contour of largest area
		x,y,w,h = cv2.boundingRect(best_contour)
		bounding_box = ((x,y), (x+w, y+h))
		lookahead = transformUvToXy(center[0], center[1] + 190)
		#print('center', center)
		
		return (bounding_box, black_image, lookahead)


	return (((0,0),(0,0)), black_image, (1.3, 0.0))

PTS_IMAGE_PLANE = [[180, 317],
                   [416, 352],
                   [583, 385],
                   [299, 262],
                   [447, 245],
                   [203, 224],
                   [373, 218],
                   [87, 247],
                   [432, 276],
                   [236, 243],
                   [161, 282]] # dummy points
######################################################

# PTS_GROUND_PLANE units are in inches
# car looks along positive x axis with positive y axis to left

######################################################
## DUMMY POINTS -- ENTER YOUR MEASUREMENTS HERE
PTS_GROUND_PLANE = [[18, 8.],
                    [15, -4.25],
                    [23.25, -17],
                    [27, 1.5],
                    [32.25, -12.],
                    [39.25, 12.75],
                    [43.5, -7.],
                    [31., 21.],
                    [24.25, -8.],
                    [31.75, 7.5],
                    [22.5, 10.25]
                    ] # dummy points
# PTS_IMAGE_PLANE = [[235.0, 199.],
#                    [345., 248.],
#                    [109., 353.],
#                    [745., 213.]] # dummy points
# ######################################################

# # PTS_GROUND_PLANE units are in inches
# # car looks along positive x axis with positive y axis to left

# ######################################################
# ## DUMMY POINTS -- ENTER YOUR MEASUREMENTS HERE
# PTS_GROUND_PLANE = [[1.07, 0.35],
#                     [0.55, 0.04],
#                     [0.6, .43],
#                     [.84, -0.3]] # dummy points
# ######################################################

METERS_PER_INCH = 0.0254

#Initialize data into a homography matrix

np_pts_ground = np.array(PTS_GROUND_PLANE)
np_pts_ground = np_pts_ground * METERS_PER_INCH
# * METERS_PER_INCH
np_pts_ground = np.float32(np_pts_ground[:, np.newaxis, :])

np_pts_image = np.array(PTS_IMAGE_PLANE)
np_pts_image = np_pts_image * 1.0
np_pts_image = np.float32(np_pts_image[:, np.newaxis, :])

h, err = cv2.findHomography(np_pts_image, np_pts_ground)

def transformUvToXy(u, v):
        """
        u and v are pixel coordinates.
        The top left pixel is the origin, u axis increases to right, and v axis
        increases down.

        Returns a normal non-np 1x2 matrix of xy displacement vector from the
        camera to the point on the ground plane.
        Camera points along positive x axis and y axis increases to the left of
        the camera.

        Units are in meters.
        """
        homogeneous_point = np.array([[u], [v], [1]])
        xy = np.dot(h, homogeneous_point)
        scaling_factor = 1.0 / xy[2, 0]
        homogeneous_xy = xy * scaling_factor
        x = homogeneous_xy[0, 0]
        y = homogeneous_xy[1, 0]
        return x, y

