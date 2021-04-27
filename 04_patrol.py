# -*-coding:utf-8-*-
# Copyright (c) 2020 DJI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import cv2
import robomaster
from robomaster import robot, flight
from numpy import *
import threading

# from multiprocessing import Process

import numpy as np
from PIL import Image

from yolo import YOLO

def patrol():
    # robomaster.config.LOCAL_IP_STR = "192.168.31.44"
    # tl_drone = robot.Drone()
    # tl_drone.initialize(conn_type="sta")

    # tl_flight = tl_drone.flight
    # tl_camera = tl_drone.camera

    # yolo = YOLO()  # For YOLO

    # # initialize the camera
    # t = time.time()
    # t_list = [0]*20

    # tl_camera.start_video_stream(display=False)
    # tl_camera.set_fps("low")  # 默认high
    # tl_camera.set_resolution("low")  # 默认high
    # tl_camera.set_bitrate(6)  # 默认6

    # # -------------------------------------#
    # #   调用摄像头
    # #   capture=cv2.VideoCapture("1.mp4")
    # # -------------------------------------#
    # fps = 0.0

    # while (True):
    #     # 读取某一帧
    #     img = tl_camera.read_cv2_image()

    #     t1 = time.time()

    #     # 格式转变，BGRtoRGB
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     # 转变成Image
    #     img = Image.fromarray(np.uint8(img))
    #     # 进行检测
    #     img = np.array(yolo.detect_image(img))
    #     # RGBtoBGR满足opencv显示格式
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    #     # 缩小原图
    #     # height, width = img.shape[:2]
    #     # size = (int(width*0.3), int(height*0.5))
    #     # img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    #     # 放大原图，输入尺寸格式为（宽，高）
    #     fx = 3
    #     fy = 2.1
    #     img = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

    #     fps = (fps + (1. / (time.time() - t1))) / 2
    #     print("fps= %.2f" % (fps))
    #     img = cv2.putText(img, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #     cv2.imshow("video", img)

    #     c = cv2.waitKey(1) & 0xff
    #     if c == 27:
    #         capture.release()
    #         break

    # 起飞
    tl_flight.takeoff().wait_for_completed()
    

    # patrol
    for i in range(0,2):
        tl_flight.go(x=150, y=0, z=0, speed=50).wait_for_completed()
        tl_flight.go(x=0, y=40, z=0, speed=50).wait_for_completed()
        tl_flight.go(x=-150, y=0, z=0, speed=50).wait_for_completed()
        tl_flight.go(x=0, y=40, z=0, speed=50).wait_for_completed()
    
    # back to origin
    tl_flight.go(x=0, y=-200, z=0, speed=50).wait_for_completed()

    # look for nearest mid card 
    tl_flight.go(x=0, y=0, z=100, speed=30, mid1="m-2").wait_for_completed()
    
    # pre-landing
    tl_flight.go(x=0, y=0, z=70, speed=20, mid1="m-2").wait_for_completed()
    time.sleep(2)
    tl_flight.go(x=0, y=0, z=40, speed=20, mid1="m-2").wait_for_completed()
    time.sleep(2)

    # 降落
    tl_flight.land().wait_for_completed()

def camStream():
    # robomaster.config.LOCAL_IP_STR = "192.168.31.44"
    # tl_drone = robot.Drone()
    # tl_drone.initialize(conn_type="sta")

    # tl_flight = tl_drone.flight
    # tl_camera = tl_drone.camera

    # yolo = YOLO()  # For YOLO

    # # initialize the camera
    # t = time.time()
    # t_list = [0]*20

    # tl_camera.start_video_stream(display=False)
    # tl_camera.set_fps("low")  # 默认high
    # tl_camera.set_resolution("low")  # 默认high
    # tl_camera.set_bitrate(6)  # 默认6

    # # -------------------------------------#
    # #   调用摄像头
    # #   capture=cv2.VideoCapture("1.mp4")
    # # -------------------------------------#
    fps = 0.0

    while (True):
        # 读取某一帧
        img = tl_camera.read_cv2_image()

        t1 = time.time()

        # 格式转变，BGRtoRGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 转变成Image
        img = Image.fromarray(np.uint8(img))
        # 进行检测
        img = np.array(yolo.detect_image(img))
        # RGBtoBGR满足opencv显示格式
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 缩小原图
        # height, width = img.shape[:2]
        # size = (int(width*0.3), int(height*0.5))
        # img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        # 放大原图，输入尺寸格式为（宽，高）
        fx = 3
        fy = 2.1
        img = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))
        img = cv2.putText(img, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", img)

        c = cv2.waitKey(1) & 0xff
        if c == 27:
            capture.release()
            break

if __name__ == '__main__':
    robomaster.config.LOCAL_IP_STR = "192.168.31.44"
    tl_drone = robot.Drone()
    tl_drone.initialize(conn_type="sta")

    tl_flight = tl_drone.flight
    tl_camera = tl_drone.camera

    yolo = YOLO()  # For YOLO

    # initialize the camera
    t = time.time()
    t_list = [0]*20

    tl_camera.start_video_stream(display=False)
    tl_camera.set_fps("low")  # 默认high
    tl_camera.set_resolution("low")  # 默认high
    tl_camera.set_bitrate(6)  # 默认6

    # -------------------------------------#
    #   调用摄像头
    #   capture=cv2.VideoCapture("1.mp4")
    # -------------------------------------#
    # fps = 0.0

    p = threading.Thread(name='patrol', target=patrol)
    s = threading.Thread(name='stream', target=camStream)

    p.start()
    s.start()

    tl_drone.close()
