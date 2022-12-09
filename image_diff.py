# @Author==>QingQiangJia
# @Time  ==>2022/3/4 14:39
# @Email ==>jiaqingqiang@foxmail.com
# pip install structural_similarity -i https://pypi.douban.com/simple
# pip install imutils -i https://pypi.douban.com/simple
# pip install opencv-python -i https://pypi.douban.com/simple
from skimage.metrics import structural_similarity
import cv2
import numpy as np
from osgeo import gdal

# print('输入影像1路径')
#
# print('输入阈值：此值越小越精细')
# areavalue = ''
# while (areavalue==''):
#     areavalue = input()
#     if(areavalue==''):
#         print('检测到未输入阈值 请再次输入')
# try:
#     areavalue = int(areavalue)
# except:
#     print('输入阈值不合法 程序已退出')
#     exit()
before = cv2.imread('[the first image]')
after = cv2.imread('[the second image]')
dataset = gdal.Open('[the tif with coordinate system]')

 def calculateXY(dataset, x, y):
     minx, xres, xskew, maxy, yskew, yres = dataset.GetGeoTransform()
     x = minx + xres * x
     y = maxy + yres * y
     return 'x:'+str(x)+',y:'+str(y)

# Convert images to grayscale
before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# Compute SSIM between two images
(score, diff) = structural_similarity(before_gray, after_gray, full=True)
# print("Image similarity", score)

# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type in the range [0,1]
# so we must convert the array to 8-bit unsigned integers in the range
# [0,255] before we can use it with OpenCV
diff = (diff * 255).astype("uint8")

# Threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

mask = np.zeros(before.shape, dtype='uint8')
filled_after = after.copy()
record = 0
recordArr = []
for c in contours:
    area = cv2.contourArea(c)
    if area > 40:
        record += 1
        x, y, w, h = cv2.boundingRect(c)
        recordArr.append('第{}处范围为：{}  {}'.format(record, calculateXY(dataset, x, y), calculateXY(dataset, x+w, y+h)))
        cv2.rectangle(before, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.rectangle(after, (x, y), (x + w, y + h), (36, 255, 12), 2)
        cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
        cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)

 with open('report.txt', 'w') as f:
     for item in recordArr:
         f.write(item+'\n')

print('报告已生成 程序已退出')
cv2.imshow('before', before)
cv2.imshow('after', after)
cv2.imshow('diff', diff)
cv2.imshow('mask', mask)
cv2.imshow('filled after', filled_after)
cv2.waitKey(0)
