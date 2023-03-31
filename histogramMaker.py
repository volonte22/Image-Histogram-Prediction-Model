# AUTHOR : JACK VOLONTE
# CS 429 HW 2 - PART 1
# 9/22/2022
# This program reads in images of Aurora images and creates a histogram of size 256+256+256 bins that are written to the file "AurHistogram.txt" - then 20% have been
# moved manually to file "validationAurs.txt" for later use.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import time



AurNo = glob.glob('./images/AurNo/*.png')
AurYes = glob.glob('./images/AurYes/*.png')

fileWrite = open("AurHistogram.txt", "w")
fileWrite2 = open("NonNormalized.txt", "w")
np.set_printoptions(suppress=True)

# No images histograms
for b in AurNo:
	x = b
	# load image using cv2
	image = cv2.imread(b)

	# create r,g,b arrays
	r = np.array(np.zeros(256))
	g = np.array(np.zeros(256))
	b = np.array(np.zeros(256))
	for i, j in np.ndindex(image.shape[:-1]):  # image.shape = (150,150)

		# get each r g b from each image
		r1 = image[i][j][0]
		g2 = image[i][j][1]
		b3 = image[i][j][2]

		# up the count for each pixel #
		r[r1] = r[r1] + 1
		g[g2] = g[g2] + 1
		b[b3] = b[b3] + 1

	# normalize arrays
	r2 = r
	g4 = g
	b5 = b
	r = ((r - min(r2)) / (max(r2) - min(r2))).round(4)
	g = ((g - min(g4)) / (max(g4) - min(g4))).round(4)
	b = ((b-min(b5)) / (max(b5) - min(b5))).round(4)




	# send to one array of 256*3 length
	rgb = np.array([])
	rgb = np.concatenate((r,g,b),axis=0)
	#print(rgb)
	fileWrite.write('-1' + ' ')
	for a in rgb:
		fileWrite.write(str(a) + ' ')
	fileWrite.write('#<' + str(x) + '>\n')

# Yes images histograms
for b in AurYes:
	x = b
	# load image using cv2
	image = cv2.imread(b)

	# create r,g,b arrays
	r = np.array(np.zeros(256))
	g = np.array(np.zeros(256))
	b = np.array(np.zeros(256))
	for i, j in np.ndindex(image.shape[:-1]):  # image.shape = (150,150)

		# get each r g b from each image
		r1 = image[i][j][0]
		g2 = image[i][j][1]
		b3 = image[i][j][2]

		# up the count for each pixel #
		r[r1] = r[r1] + 1
		g[g2] = g[g2] + 1
		b[b3] = b[b3] + 1

	# normalize arrays
	r2 = r
	g4 = g
	b5 = b
	r = ((r - min(r2)) / (max(r2) - min(r2))).round(4)
	g = ((g - min(g4)) / (max(g4) - min(g4))).round(4)
	b = ((b-min(b5)) / (max(b5) - min(b5))).round(4)


	# send to one array of 256*3 length
	rgb = np.array([])
	rgb = np.concatenate((r,g,b),axis=0)
	#print(rgb)

	fileWrite.write('1' + ' ')
	for a in rgb:
		fileWrite.write(str(a) + ' ')
	fileWrite.write('#<' + str(x) + '>\n')





plt.show()
