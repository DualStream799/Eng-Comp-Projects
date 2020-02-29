#!/usr/bin/python
# -*- coding: utf-8 -*-

#### Projeto 01 - Circle Detector ####

__author__ = "DualStream799"

import cv2
import numpy as np
from matplotlib import pyplot as plt
import time


def frame_input(capture):
	"""Recieves a webcam frame, flips it horizontally and returns it in BGR (default), RGB, HSV and Grayscale color spaces"""
	# Capture frame-by-frame:
	ret, frame = capture.read()
	# Invert captured frame:
	fliped_frame = cv2.flip(frame, 1)
	# Convert frames from BGR to Grayscale:
	gray_frame = cv2.cvtColor(fliped_frame, cv2.COLOR_BGR2GRAY)
	# Convert frames from BGR to RGB:
	rgb_frame = cv2.cvtColor(fliped_frame, cv2.COLOR_BGR2RGB)
	# Convert frames from RGB to HSV:
	hsv_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)

	return fliped_frame, rgb_frame, hsv_frame, gray_frame


def cm_masks_module(frame):
	pass


def output_selector():
	pass


def capture_single_frame(frame, name='single_frame.jpg'):
	"""Saves a frame captured by OpenCV"""
	cv2.imwrite(name, frame)


# Parameters to use when opening the webcam:
cap = cv2.VideoCapture(0)
# Setting a resolution limit for video capture:
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Visualization controller:
display_mode = 'canny'

while(True):
	time.sleep(0.1)
	# Capture frame-by-frame:
	bgr_frame, rgb_frame, hsv_frame, gray_frame = frame_input(cap)


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



	# 
	canny_frame = cv2.Canny(gray_frame, 100, 200)


	# Display the resulting frame (USE KEYBOARD TO CHANGE VISUALIZATION: simply change the image to be displayed)
	if display_mode == 'default':
		cv2.imshow('frame', bgr_frame)
	elif display_mode == 'cm_mask':
		cv2.imshow('frame', masked_frame)
	elif display_mode == 'canny':
		cv2.imshow('frame', canny_frame)
	else:
		cv2.imshow('frame', bgr_frame)

	# Does a command depending of the keyboard key pressed (useful to mode switch):
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	elif cv2.waitKey(1) & 0xFF == ord('c'):
		capture_single_frame(fliped_frame)
		break
	
	elif cv2.waitKey(1) & 0xFF == ord('1'):
		display_mode = 'default'
		print("display_mode: {}".format(display_mode))

	elif cv2.waitKey(1) & 0xFF == ord('2'):
		display_mode = 'cm_mask'
		print("display_mode: {}".format(display_mode))

	elif cv2.waitKey(1) & 0xFF == ord('3'):
		display_mode = 'canny'
		print("display_mode: {}".format(display_mode))

	

print(display_mode)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()