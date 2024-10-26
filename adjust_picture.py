# -*- coding: utf-8 -*-
"""
Created on Thu May 13 16:43:35 2021

@author: USER
"""

import cv2
import numpy as np
import pytesseract

# 讀取影像
image = cv2.imread('0.jpg', cv2.IMREAD_COLOR)

# 查看原始圖片尺寸大小
size = image.shape
sz1, sz2, sz3 = size[0], size[1], size[2]
print('Original Image Dimensions:')
print(f'Width: {sz1}\nHeight: {sz2}\nChannels: {sz3}')

# 縮放圖片至 100x100 像素大小
res_img = cv2.resize(image, (100, 100), interpolation=cv2.INTER_CUBIC)

# 查看縮放後圖片的尺寸大小
size = res_img.shape
sz4, sz5, sz6 = size[0], size[1], size[2]
print('Resized Image Dimensions:')
print(f'Width: {sz4}\nHeight: {sz5}\nChannels: {sz6}')

# 轉為灰階圖片
gray_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2GRAY)

# 二值化處理
ret, sim_inv = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 模糊化處理
mblur = cv2.medianBlur(sim_inv, 3)

# 開運算操作 (去除小的白點雜訊)
kernel = np.ones((2, 2), np.uint8)
open_img = cv2.morphologyEx(mblur, cv2.MORPH_OPEN, kernel)

# 顯示各階段處理後的圖片
cv2.imshow("Original Image", image)
cv2.imshow("Resized Image", res_img)
cv2.imshow("Gray Image", gray_img)
cv2.imshow("Binary Image", sim_inv)
cv2.imshow("Blurred Image", mblur)

# 等待按鍵後關閉所有顯示的圖片窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
