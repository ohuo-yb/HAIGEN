import cv2
import os
path='./test_new_edge'
img_all = os.listdir(path)
for i in img_all:
    img_g = cv2.imread(os.path.join(path,i), 0)  # 获取路径img/0.jpg的图像，图像类型为RGB图像

    t, img_bin = cv2.threshold(img_g, 200, 255, cv2.THRESH_BINARY)  # 二值化

    cv2.imshow("img_g", img_g)  # 显示RGB图像
    cv2.imshow("img_bin", img_bin)  # 显示二值化图像

    cv2.waitKey(0)  # 等待时间
#
# mapfile = input("Map file? ")
# img = cv2.imread(mapfile);
# if img is None:
#     print ("Error opening image!")
#     input("Press Enter to exit...")
#     exit(-1);
# xys = []
# t_xys = []
# w = len(img)
# h = len(img[0,:])
# for i in range(w):
#     for j in range(h):
#         if sum(img[i,j,:]) == 0:
#             xys.append(str(i) + "\n")
#             xys.append(str(j) + "\n")
#         elif img[i,j,2] - img[i,j,1] > 50 and img[i,j,2] - img[i,j,0] > 50:
#             t_xys.append(str(i) + "\n")
#             t_xys.append(str(j) + "\n")
