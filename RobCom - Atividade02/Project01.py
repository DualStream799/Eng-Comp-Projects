#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "DualStream799"


from matplotlib import pyplot as plt
from heapq import nlargest
from math import degrees
import numpy as np
import time
import cv2


def frame_flip(capture, flip=False, flip_mode=0):
	"""Recieves a webcam frame and flips it or not, depending on 'flip' argument
	X-axis Flip (Vertical): 'flip_mode' = 0
	Y-axis Flip (Horizontal): 'flip_mode' > 0
	Both-axis Flip(Vertical and Horizontal): 'flip_mode' < 0 """

	# Capture frame-by-frame:
	ret, frame = capture.read()
	# Invert captured frame:
	if flip == True:
		return cv2.flip(frame, 20)
	else:
		return frame
	


def output_selector(window_name, output_frame):
	# Displays the resulting frame:
	cv2.imshow(window_name, output_frame)


def text_on_frame(frame, text, position, thickness, font_size=1, text_color=(255, 255, 255), shadow_color=(128, 128, 128), font_style=cv2.FONT_HERSHEY_SIMPLEX, line_style=cv2.LINE_AA):
	"""Displays a text on the frame with a shadow behind it for better visualization on any background"""
	cv2.putText(frame, text, position, font_style, font_size, shadow_color, thickness+1, line_style)
	cv2.putText(frame, text, position, font_style, font_size, text_color, thickness, line_style)
	

def capture_single_frame(frame, name='single_frame.jpg'):
	"""Saves a frame captured by OpenCV"""
	pass

def angular_coefficient(point1, point2, decimals=0):
	""""""
	return round(degrees((point2[1] - point1[1])/(point2[0] - point1[0])), decimals)


# Window name text:
window_name = "Computer Vision Detector Algorithm"
# Window width and height:
window_size = (640,480)
# Display mode controller:
display_mode = 'canny'
# Display mode dict for text display:
display_mode_text_dict = {'default': 'Nothing',
						  'mask': 'Cyan & Magenta',
						  'canny': 'Edges',
						  'hough': 'Circles'}
# 
color_dict = {'cyan': (255, 255, 0),
			  'magenta': (255, 0, 255)}
# Angular Coefficient start value:
m = 0.0


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
	if circles is not None and display_mode == 'hough':
		# Lists to store only valid circles (better explanation below):
		valid_circles = []
		# Draws all circles detected in the frame:
		for circle in circles[0]:
			# Parameters obtained by 'cv2.HoughCircles':
			x_pos, y_pos, radius_size = circle
			# Gets the 
			circle_color = (0,255,255) #tuple([int(val) for val in masked_frame[int(x_pos), int(y_pos), :]])
			# Draws the circle on the frame:
			cv2.circle(color_edges_frame, center=(x_pos, y_pos), radius=radius_size, color=circle_color, thickness=2)
			# if color in cyan or magenta ranges, print the color name on the frame
		
			# Checks if the center of the detected circles are inside the valid region of the color masks and stores the circle's data and wich mask were detected (otherwise, the circles are false and are not stored:
			if cyan_mask[int(y_pos), int(x_pos)] == 255:
				valid_circles.append(['cyan', list(circle)])
			elif magenta_mask[int(y_pos), int(x_pos)] == 255:
				valid_circles.append(['magenta', list(circle)])

		# If two circles or more were detected, filter only the two largest ones (based on masks only discards false detections):
		if len(valid_circles) >= 2:
			# Finds the two largest circles (the ones with bigger radius):
			largest_circles = nlargest(2, valid_circles, key=lambda x:x[1][2])	
			# Draws a line between the circles' center:
			cv2.line(color_edges_frame, (largest_circles[0][1][0], largest_circles[0][1][1]), (largest_circles[1][1][0], largest_circles[1][1][1]), (0,0,255), 2)
			# Calculates the line's angular coeficient and updates the varible:
			m = angular_coefficient(largest_circles[0][1], largest_circles[1][1])
			# Draws the circles' color above the detected circles:
			_ = [text_on_frame(color_edges_frame, circle[0], (int(circle[1][0]), int(circle[1][1]+circle[1][2]+30)), 1, text_color=color_dict[circle[0]])  for circle in largest_circles]



	# Display the resulting frame (USE KEYBOARD TO CHANGE VISUALIZATION: simply change the image to be displayed):
	if display_mode == 'default':
		text_on_frame(bgr_frame, "Detecting " + display_mode_text_dict[display_mode], (0, 30), 1)
		cv2.imshow(window_name, bgr_frame)

	elif display_mode == 'mask':
		text_on_frame(masked_frame, "Detecting " + display_mode_text_dict[display_mode], (0, 30), 1)
		cv2.imshow(window_name, masked_frame)

	elif display_mode == 'canny':
		text_on_frame(edges_frame, "Detecting " + display_mode_text_dict[display_mode], (0, 30), 1)
		cv2.imshow(window_name, edges_frame)

	elif display_mode == 'hough':
		text_on_frame(color_edges_frame, "{}*".format(int(m)), (window_size[0]-100, 30), 1)
		text_on_frame(color_edges_frame, "Detecting " + display_mode_text_dict[display_mode], (0, 30), 1)
		cv2.imshow(window_name, color_edges_frame)

	elif display_mode == 'line':
		pass
	elif display_mode == 'angular_coef':
		pass
	elif display_mode == 'distance':
		pass
	elif display_mode == 'all':
		pass
	else:
		cv2.imshow(window_name, bgr_frame)



	# Waits for a certain time (in milisseconds) for a key input ('0xFF' is used to handle input changes caused by NumLock):
	delay_ms = 60
	key_input = cv2.waitKey(delay_ms) & 0xFF

	# Display Mode Switch:

	# Exit the program:
	if  key_input == ord('q'):	
		break
	# Capture a frame and saves it as a .jpeg file:
	elif key_input == ord('c'):
		cv2.imwrite(name, 'single_frame.jpg')
	# Default visualization:
	elif key_input == ord('1'):
		display_mode = 'default'
	# Cyan and Magenta Mask visualization:
	elif key_input == ord('2'):
		display_mode = 'mask'
	# Edge Detection visualization:
	elif key_input == ord('3'):
		display_mode = 'canny'
	# Circles Detection visualization:
	elif key_input == ord('4'):
		display_mode = 'hough'

	

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()