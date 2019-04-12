import cv2
import numpy as np


def resize_image(image, out_width, out_height):
    """
        Resize an image to the "good" input size
    """
    im_arr = image
    h, w = np.shape(im_arr)[:2]
    ratio = out_height / h

    im_arr_resized = cv2.resize(im_arr, (int(w * ratio), out_height))
    re_h, re_w = np.shape(im_arr_resized)[:2]

    if re_w >= out_width:
        final_arr = cv2.resize(im_arr, (out_width, out_height))
    else:
        final_arr = np.ones((out_height, out_width), dtype=np.uint8) * 255
        final_arr[:, 0:np.shape(im_arr_resized)[1]] = im_arr_resized
    return final_arr
