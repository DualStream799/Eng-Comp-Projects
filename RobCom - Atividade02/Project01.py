#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "DualStream799"

# Importing Libraries:
from matplotlib import pyplot as plt
from heapq import nlargest
from math import atan, degrees, sqrt
import numpy as np
import time
import cv2


def find_homography_draw_box(kp1, kp2, img_cena, img_original, good):
    
    out = img_cena.copy()
    
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)


    # Tenta achar uma trasformacao composta de rotacao, translacao e escala que situe uma imagem na outra
    # Esta transformação é chamada de homografia 
    # Para saber mais veja 
    # https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()


    
    h,w = img_original.shape
    # Um retângulo com as dimensões da imagem original
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    # Transforma os pontos do retângulo para onde estao na imagem destino usando a homografia encontrada
    dst = cv2.perspectiveTransform(pts,M)


    # Desenha um contorno em vermelho ao redor de onde o objeto foi encontrado
    img2b = cv2.polylines(out,[np.int32(dst)],True,(255,255,0),5, cv2.LINE_AA)
    
    return img2b



def frame_flip(capture, flip=False, flip_mode=0):
	"""Recieves a webcam frame and flips it or not, depending on 'flip' argument
	X-axis Flip (Vertical): 'flip_mode' = 0
	Y-axis Flip (Horizontal): 'flip_mode' > 0
	Both-axis Flip(Vertical and Horizontal): 'flip_mode' < 0 """

	# Capture frame-by-frame:
	ret, frame = capture.read()
	# Invert captured frame:
	if flip == True:
		return cv2.flip(frame, 1)
	else:
		return frame


def output_selector(window_name, output_frame):
	# Displays the resulting frame:
	cv2.imshow(window_name, output_frame)

def text_on_frame(frame, text, position, thickness, font_size=1, text_color=(255, 255, 255), shadow_color=(128, 128, 128), font_style=cv2.FONT_HERSHEY_SIMPLEX, line_style=cv2.LINE_AA):
	"""Displays a text on the frame with a shadow behind it for better visualization on any background"""
	cv2.putText(frame, text, position, font_style, font_size, shadow_color, thickness+1, line_style)
	cv2.putText(frame, text, position, font_style, font_size, text_color, thickness, line_style)

def angular_coefficient(point1, point2, decimals=0):
	"""Calculates the angular coefficient if a line between two points using the current formula: (y - y0) = m*(x - x0)"""
	return round(degrees(atan((point2[1] - point1[1])/(point2[0] - point1[0]))), decimals)


# Window name text:
window_name = "Computer Vision Detector Algorithm"
# Window width and height:
window_size = (640, 480)
# Display mode controller:
display_mode = 'default'
# Angular Coefficient start value:
m = 0.0
sheet_distance = 0.0
min_matches_val = 10
# Display mode dict for text display:
display_mode_text_dict = {'default': 'Nothing',
						  'mask': 'Cyan & Magenta',
						  'canny': 'Edges',
						  'hough': 'Circles',
						  'line': 'Line',
						  'angular_coef': 'Line & Inclination',
						  'distance': 'Sheet Distance',
						  'colorfull_circles': 'Circle Colors',
						  'brisk': 'Logo',
						  'all': 'Everything'}
# Color dict for "paint" elements:
color_dict = {'cyan': (255, 255, 0),
			  'magenta': (255, 0, 255)}


# Parameters to use when opening the webcam:
cap = cv2.VideoCapture(0)
# Setting a resolution limit for video capture:
cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_size[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_size[1])


while True:
	# Reads frame and flips it horizontally:
	bgr_frame = frame_flip(cap, True, 1)
	# Converts frame from BGR to Grayscale:
	gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
	# Converts frame from BGR to RGB:
	rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
	# Converts frame from RGB to HSV:
	hsv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)
	# Cyan's Hue value of HSV color space:
	cyan_hue = 180
	# Magenta's Hue value of HSV color space:
	magenta_hue = 300
	# Cyan's limit for creating mask (manually adjusted):
	cyan_high = (cyan_hue/2 + 40, 255, 255)
	cyan_low = (cyan_hue/2 - 40, 150, 70)
	# Magenta's limit for creating mask (manually adjusted):
	magenta_high = (magenta_hue/2 + 30, 255, 255)
	magenta_low = (magenta_hue/2 - 30, 140, 70)
	# Creating masks for each one of the target colors:
	cyan_mask = cv2.inRange(hsv_frame, cyan_low, cyan_high)
	magenta_mask = cv2.inRange(hsv_frame, magenta_low, magenta_high)
	# Creating a new mask combining the two previous ones:
	fused_mask = cv2.bitwise_or(cyan_mask, magenta_mask)
	# Filtering the original image using the mask (There's color only where the mask is white):
	masked_frame = cv2.bitwise_and(bgr_frame, bgr_frame, mask=fused_mask)
	# Bluring the grayscale frame in order to enhance edge detection:
	blured_frame = cv2.GaussianBlur(gray_frame, ksize=(5, 5), sigmaX=0)
	# Using Canny's algorithm to detect edges:
	edges_frame = cv2.Canny(blured_frame, 100, 200)
	# Using Hough Transform method to detect circles:
	circles = cv2.HoughCircles(edges_frame, cv2.HOUGH_GRADIENT, dp=2, minDist=40, param1=30, param2=100, minRadius=5, maxRadius=60)
	# Converting 'edges_frame' to BGR in order to get colorfull circles:
	color_edges_frame = cv2.cvtColor(edges_frame, cv2.COLOR_GRAY2BGR)	
	# Only if circles detected (avoids that 'None' reaches the 'for' loop and crashes the program):
	if circles is not None and (display_mode == 'hough' or display_mode == 'line' or display_mode == 'angular_coef' or display_mode == 'distance' or display_mode == 'colorfull_circles' or display_mode == 'all'):
		# Lists to store only valid circles (better explanation below):
		valid_circles = []
		# Draws all circles detected in the frame:
		for circle in circles[0]:
			# Parameters obtained by 'cv2.HoughCircles':
			x_pos, y_pos, radius_size = circle		

			# Checks if the center of the detected circles are inside the valid region of the color masks and stores the circle's data and wich mask were detected (otherwise, the circles are false and are not stored:
			if cyan_mask[int(y_pos), int(x_pos)] == 255:
				valid_circles.append(['cyan', list(circle)])
			elif magenta_mask[int(y_pos), int(x_pos)] == 255:
				valid_circles.append(['magenta', list(circle)])

			if display_mode != 'colorfull_circles':
				# Draws the circle on the frame:
				cv2.circle(color_edges_frame, center=(x_pos, y_pos), radius=radius_size, color=(0,255,255), thickness=2)
			
		# If two circles or more were detected, filter only the two largest ones (based on masks only discards false detections):
		if len(valid_circles) >= 2 and (display_mode == 'line' or display_mode == 'angular_coef' or display_mode == 'distance' or display_mode == 'colorfull_circles' or display_mode == 'all'):
			# Finds the two largest circles (the ones with bigger radius):
			largest_circles = nlargest(2, valid_circles, key=lambda x:x[1][2])

			if display_mode == 'line' or display_mode == 'angular_coef' or display_mode == 'all':
				# Draws a line between the circles' center:
				cv2.line(color_edges_frame, (largest_circles[0][1][0], largest_circles[0][1][1]), (largest_circles[1][1][0], largest_circles[1][1][1]), (0,0,255), 2)

			if display_mode == 'angular_coef' or display_mode == 'all':
				# Calculates the line's angular coeficient and updates the varible:
				m = angular_coefficient(largest_circles[0][1], largest_circles[1][1])

			if display_mode == 'colorfull_circles' or display_mode == 'all':
				# Draws the circles with the correspondent color:
				_ = [cv2.circle(color_edges_frame, center=(circle[1][0], circle[1][1]), radius=circle[1][2], color=color_dict[circle[0]], thickness=2)  for circle in largest_circles]
				# Draws the circles' color above the detected circles:
				_ = [text_on_frame(color_edges_frame, circle[0], (int(circle[1][0]), int(circle[1][1]+circle[1][2]+30)), 1, text_color=color_dict[circle[0]])  for circle in largest_circles]

			if display_mode == 'distance' or display_mode == 'all':
				# Calculates the lenght of the line:
				Ax = largest_circles[0][1][0]
				Ay = largest_circles[0][1][1]
				Bx = largest_circles[1][1][0]
				By = largest_circles[1][1][1]
				# Values obtained from calibration fixing D = 32cm (between the webcam and the sheet):
				sheet_height = 14 # centimeters [H] (between the center of the two circles in the sheet) 
				frame_distance = 628.5 # pixels [d] (virtual distance between the webcam and the frame)
				frame_height = sqrt((Bx - Ax)**2 + (By - Ay)**2) # pixels [h] (between the center of the two circles in the frame)
				# Calculates D using the Pinhole principle's formula:
				sheet_distance = round(sheet_height*frame_distance/frame_height, 2)

	if display_mode == 'brisk':
		# Flips briks again in order to let the words on the right direction (the image displayed is flipped):
		brisk_frame = cv2.flip(bgr_frame.copy(), 1)
		# Loads the logo:
		logo = cv2.imread('logo.png')
		# Converts Logo to Grayscale because of BRISK's needs:
		gray_logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)

		gray_brisk_frame = cv2.cvtColor(brisk_frame, cv2.COLOR_BGR2GRAY)
		# Initializes BRISK module:
		brisk_init = cv2.BRISK_create()
		# Finds unique keypoints on each of the images (logo and frame):
		keypoints_logo, descriptors_logo = brisk_init.detectAndCompute(gray_logo, None)
		keypoints_frame, descriptors_frame = brisk_init.detectAndCompute(gray_brisk_frame, None)
		# Sets the matching algorithim:
		bf = cv2.BFMatcher(cv2.NORM_HAMMING)
		# Finds matches comparing keypoints using KNN (K-Nearest Neighbors) algorithm:
		matches = bf.knnMatch(descriptors_logo, descriptors_frame, k=2)
		# Filters the valid matches as per Lowe's ratio test:
		valid_matches = [m for m, n in matches if m.distance < 0.7*n.distance]
		# 
		if len(valid_matches) > min_matches_val:
			#
			framed = find_homography_draw_box(keypoints_logo, keypoints_frame, gray_brisk_frame, gray_logo, valid_matches)
		else:
			print("Not enough matches are found {}/{}".format(len(valid_matches), min_matches_val))





	# Displays the resulting frame (Using keyboard to change visualization: simply change the image to be displayed):
	# Key '1':
	if display_mode == 'default':
		text_on_frame(bgr_frame, "Detecting " + display_mode_text_dict[display_mode], (0, 30), 1)
		cv2.imshow(window_name, bgr_frame)
		
	# Key '2':
	elif display_mode == 'mask':
		text_on_frame(masked_frame, "Detecting " + display_mode_text_dict[display_mode], (0, 30), 1)
		cv2.imshow(window_name, masked_frame)
	
	# Key '3':
	elif display_mode == 'canny':
		text_on_frame(edges_frame, "Detecting " + display_mode_text_dict[display_mode], (0, 30), 1)
		cv2.imshow(window_name, edges_frame)
	
	# Key '4':
	elif display_mode == 'hough':
		text_on_frame(color_edges_frame, "Detecting " + display_mode_text_dict[display_mode], (0, 30), 1)
		cv2.imshow(window_name, color_edges_frame)
	
	# Key '5':
	elif display_mode == 'line':
		text_on_frame(color_edges_frame, "Detecting " + display_mode_text_dict[display_mode], (0, 30), 1)
		cv2.imshow(window_name, color_edges_frame)
	
	# Key '6':
	elif display_mode == 'angular_coef':
		text_on_frame(color_edges_frame, "Inclination: {}".format(int(m)), (0, window_size[1] - 5), 1)
		text_on_frame(color_edges_frame, "Detecting " + display_mode_text_dict[display_mode], (0, 30), 1)
		cv2.imshow(window_name, color_edges_frame)
	
	# Key '7':
	elif display_mode == 'distance':
		text_on_frame(color_edges_frame, "Sheet Distance: {}cm".format(sheet_distance), (0, window_size[1] - 5), 1)
		text_on_frame(color_edges_frame, "Detecting " + display_mode_text_dict[display_mode], (0, 30), 1)
		cv2.imshow(window_name, color_edges_frame)

	# Key '8':
	elif display_mode == 'colorfull_circles':
		text_on_frame(color_edges_frame, "Detecting " + display_mode_text_dict[display_mode], (0, 30), 1)
		cv2.imshow(window_name, color_edges_frame)

	# Key '9':
	elif display_mode == 'brisk':
		img3 = cv2.drawMatches(logo,keypoints_logo, brisk_frame, keypoints_frame, valid_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
		cv2.imshow(window_name, img3)

	# Key '0':
	elif display_mode == 'all':
		text_on_frame(color_edges_frame, "Sheet Distance: {}cm".format(sheet_distance), (0, window_size[1] - 5), 1)
		text_on_frame(color_edges_frame, "Inclination: {}".format(int(m)), (0, window_size[1] - 35), 1)
		text_on_frame(color_edges_frame, "Detecting " + display_mode_text_dict[display_mode], (0, 30), 1)
		cv2.imshow(window_name, color_edges_frame)


	# Waits for a certain time (in milisseconds) for a key input ('0xFF' is used to handle input changes caused by NumLock):
	delay_ms = 60
	key_input = cv2.waitKey(delay_ms) & 0xFF

	# Display Mode Switch:
	# Exit the program:
	if  key_input == ord('q'):	
		break
	# Capture a frame and saves it as a .jpeg file:
	elif key_input == ord('c'):
		cv2.imwrite('single_frame.jpg', cv2.flip(bgr_frame, 1))
		print('frame aptured')
	# Default visualization:
	elif key_input == ord('1'):
		display_mode = 'default'
		print(display_mode)
	# Cyan and Magenta Mask visualization:
	elif key_input == ord('2'):
		display_mode = 'mask'
		print(display_mode)
	# Edge Detection visualization:
	elif key_input == ord('3'):
		display_mode = 'canny'
		print(display_mode)
	# Circles Detection visualization:
	elif key_input == ord('4'):
		display_mode = 'hough'
		print(display_mode)
	# Line Connecting Circles visualization:
	elif key_input == ord('5'):
		display_mode = 'line'
		print(display_mode)
	# Line's Angular Coefficient visualization:
	elif key_input == ord('6'):
		display_mode = 'angular_coef'
		print(display_mode)
	# Sheet Distance visualization:
	elif key_input == ord('7'):
		display_mode = 'distance'
		print(display_mode)
	# Circle Color Detection visualization:
	elif key_input == ord('8'):
		display_mode = 'colorfull_circles'
		print(display_mode)
	# Logo Finder visualization:
	elif key_input == ord('9'):
		display_mode = 'brisk'
		print(display_mode)
	# All Previous visualizations combined:
	elif key_input == ord('0'):
		display_mode = 'all'
		print(display_mode)
	
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()