import numpy as np
import cv2

from sp_vo import VisualOdometry as sp_VisualOdometry
from sp_vo import PinholeCamera

cam = PinholeCamera(640, 480, 611.164, 609.028, 316.961, 247.770)
sp_vo = sp_VisualOdometry(cam)

traj = np.zeros((1000,1000,3), dtype = np.uint8)
log_fopen = open("results/result.txt", mode='a')

sp_errors=[]
norm_errors = []
sp_feature_nums = []
norm_feature_num = []
cv2.namedWindow('Trajectory',1)
for img_id in range(263,3606):
    img_name = '/media/zhibo/storage/home/dataset/orbbec/camera/data/' + str(img_id) + '.png'
    img_id = img_id - 262
    img = cv2.imread(img_name, 0)
    img = cv2.resize(img, (320, 240))

    sp_vo.update(img, img_id)
    sp_cur_t = sp_vo.cur_t
    if(img_id > 2):
        sp_x, sp_y, sp_z = sp_cur_t[0], sp_cur_t[1], sp_cur_t[2]
    else:
        sp_x, sp_y, sp_z = 0., 0., 0.

    sp_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for(u, v) in sp_vo.px_ref:
        cv2.circle(sp_img, (u,v), 3, (0, 255, 0))

    # sp_est_point = np.array([sp_x, sp_y]).reshape(2)
    sp_draw_x, sp_draw_y = int(sp_x) + 500, int(sp_z) + 200

    cv2.circle(traj, (sp_draw_x, sp_draw_y), 1, (255, 0, 0), 1)
    # cv2.rectangle(traj, (10,20), (600,60), (0,0,0), -1)

    cv2.imshow('Feature detection', sp_img)

    cv2.imshow('Trajectory', traj)
    raw_key = cv2.waitKey(1)
    if (raw_key == 27):
        break
    if (raw_key == 32):
        continue

# live cam
# cap = cv2.VideoCapture(2)
# img_id = 0
# pauseTime = 0
# while(cap.isOpened()):
#         ret, img_name = cap.read()
#         # if ret is True:
#         sp_img = cv2.resize(img_name, (320, 240))
#         sp_vo.update(sp_img, img_id)
#         sp_cur_t = sp_vo.cur_t
#         if(img_id > 2):
#             sp_x, sp_y, sp_z = sp_cur_t[0], sp_cur_t[1], sp_cur_t[2]
#         else:
#             sp_x, sp_y, sp_z = 0., 0., 0.

#         # sp_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#         for(u, v) in sp_vo.px_ref:
#             cv2.circle(sp_img, (u,v), 3, (0, 255, 0))

#         # sp_est_point = np.array([sp_x, sp_y]).reshape(2)
#         sp_draw_x, sp_draw_y = int(sp_x) + 500, int(sp_z) + 200

#         cv2.circle(traj, (sp_draw_x, sp_draw_y), 1, (255, 0, 0), 1)
#         # cv2.rectangle(traj, (10,20), (600,60), (0,0,0), -1)
#         print (sp_vo.cur_R)

#         cv2.imshow('Feature detection', sp_img)

#         cv2.imshow('Trajectory', traj)

#         raw_key = cv2.waitKey(pauseTime)
#         if (raw_key == 27):
#             break
#         if (raw_key == 32):
#             pauseTime = 1
#         if (pauseTime == 1):
#             img_id= img_id +1


cv2.imwrite('map.png', traj)
