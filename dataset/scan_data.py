# -*- coding: utf-8 -*-
"""
Created on 2019/7/4

@author: chengli
"""
import cv2
import os
import os.path as osp
from tqdm import *
import numpy as np


def sort_points(points):
	left = sorted([point[0] for point in points])[:2]
	right = sorted([point[0] for point in points])[2:]
	up = sorted([point[1] for point in points])[:2]
	down = sorted([point[1] for point in points])[2:]

	sorted_points = {}

	for point in points:
		if point[0] in left and point[1] in up:
			sorted_points[3] = point
		elif point[0] in left and point[1] in down:
			sorted_points[4] = point
		elif point[0] in right and point[1] in down:
			sorted_points[1] = point
		elif point[0] in right and point[1] in up:
			sorted_points[2] = point

	keys = sorted(list(sorted_points.keys()))
	return [list(sorted_points[k]) for k in keys]




def transfer_loc(line):
	splits = line.strip().split(",")
	points = np.array([[int(float(splits[2 * i])), int(float(splits[2 * i + 1]))] for i in range(4)])
	points = sort_points(points)
	label = splits[-1]
	return points, label


if __name__ == "__main__":
	img_dir = "/home/pc/competition/ocr/image_train"
	label_dir = "/home/pc/competition/ocr/txt_train"

	im = cv2.imread("../sample/T1.AK_XX8hXXbnu_Z1_042512.jpg.jpg", cv2.IMREAD_GRAYSCALE)
	with open("../sample/T1.AK_XX8hXXbnu_Z1_042512.jpg.txt", "r")as f:
		readlines = f.readlines()
	loc_points, labels = zip(*[transfer_loc(line) for line in readlines])

	for i, label in enumerate(labels):
		if label != "###":
			rect = cv2.minAreaRect(np.array(loc_points[i]))

			im_copy = im.copy()

			center = tuple(rect[0])

			rot_mat = cv2.getRotationMatrix2D(center, rect[-1], 1.0)

			rot_image = cv2.warpAffine(im_copy, rot_mat, (im.shape[1], im.shape[0]))

			width = rect[1][0]
			height = rect[1][1]

			start_w = int(center[0] - (width // 2))
			end_w = int(center[0] + (width // 2))
			start_h = int(center[1] - (height // 2))
			end_h = int(center[1] + (height // 2))

			cut_imd = rot_image[start_h:end_h, start_w:end_w]
			cv2.imshow("pic", cut_imd)
			cv2.waitKey(0)

# img_names = os.listdir(img_dir)
#
# for img_name in tqdm(img_names, desc = "scan img"):
# 	img_path = osp.join(img_dir, img_name)
# 	imread = cv2.imread(img_path)
# 	txt_name = img_name[:-4] + ".txt"
# 	txt_path = osp.join(label_dir, txt_name)
# 	if not osp.exists(txt_path):
# 		print("{} 的label 文件不存在".format(img_name))
# 		continue
#
# 	with open(txt_path, "r")as f:
# 		readlines = f.readlines()
#
# 	loc_points, labels = zip(*[transfer_loc(line) for line in readlines])
#
# 	for i, label in enumerate(labels):
# 		if label != "###":
# 			rect = cv2.minAreaRect(loc_points[i])
#
# 			rotate = cv2.rotate(imread, int(rect[-1]))
#
# 			cut_imd = rotate[int(rect[0][0]):int(rect[1][0]), int(rect[0][1]):int(rect[1][1])]
# 			cv2.imshow("pic", cut_imd)
# 			cv2.waitKey(0)
#
# 	cv2.imshow("pic", imread)
# 	cv2.waitKey(0)
