from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import argparse
import pickle as pkl

import robomaster
from robomaster import robot

def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.25)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "160", type = str)
    return parser.parse_args()

if __name__ == '__main__':
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    num_classes = 80

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()

    num_classes = 80
    bbox_attrs = 5 + num_classes
    
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()


    model.eval()
    
    # videofile = 'video.avi'
    
    # cap = cv2.VideoCapture(0)
    
    # assert cap.isOpened(), 'Cannot capture source'

    # fill in your lan address:
    robomaster.config.LOCAL_IP_STR = "192.168.10.2"
    # robomaster.config.ROBOT_IP_STR = "192.168.31.143"
    # robomaster.config.DEFAULT_CONN_TYPE = "sta"
    tl_drone = robot.Drone()
    tl_drone.initialize()

    # 获取飞机电池电量信息
    tl_battery = tl_drone.battery
    battery_info = tl_battery.get_battery()
    print("Drone battery soc: {0}".format(battery_info))

    # start motor spinning
    # tl_flight = tl_drone.flight
    # tl_flight.motor_on()

    # initialize the camera
    tl_camera = tl_drone.camera

    tl_camera.start_video_stream(display=False)
    tl_camera.set_fps("low")
    tl_camera.set_resolution("low")
    tl_camera.set_bitrate(6)

    frames = 0
    start = time.time()
    # i = 0

    # cap = cv2.VideoCapture('udp://192.168.10.1:11111')
    # assert cap.isOpened(), 'Cannot capture source'

    while (True):
        ret = 1
        frame = tl_camera.read_video_frame(strategy="newest")
        if ret:
            
            img, orig_im, dim = prep_image(frame, inp_dim)
            output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
                    
            output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
            
#            im_dim = im_dim.repeat(output.size(0), 1)
            output[:,[1,3]] *= frame.shape[1]
            output[:,[2,4]] *= frame.shape[0]

            
            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))
            
            list(map(lambda x: write(x, orig_im), output))

            # # start patrol
            # i += 1
            # if i == 1:
            #     flight_action = tl_flight.takeoff()
            #     cv2.imshow("frame", orig_im)
            # # 前进300cm
            # elif i == 20:
            #     flight_action.wait_for_completed()
            #     flight_action = tl_flight.go(x=400, y=0, z=0, speed=60)
            #     cv2.imshow("frame", orig_im)
            # # 右移200cm
            # elif i == 50:
            #     flight_action.wait_for_completed()
            #     flight_action = tl_flight.go(x=0, y=-150, z=0, speed=60)
            #     cv2.imshow("frame", orig_im)
            # # 后退300cm
            # elif i == 70:
            #     flight_action.wait_for_completed()
            #     flight_action = tl_flight.go(x=-400, y=0, z=0, speed=60)
            #     cv2.imshow("frame", orig_im)
            # # 右移200cm
            # elif i == 100:
            #     flight_action.wait_for_completed()
            #     flight_action = tl_flight.go(x=0, y=-150, z=0, speed=60)
            #     cv2.imshow("frame", orig_im)
            # # 前进300cm
            # elif i == 120:
            #     flight_action.wait_for_completed()
            #     flight_action = tl_flight.go(x=400, y=0, z=0, speed=60)
            #     cv2.imshow("frame", orig_im)
            # # 右移200cm
            # elif i == 150:
            #     flight_action.wait_for_completed()
            #     flight_action = tl_flight.go(x=0, y=-150, z=0, speed=60)
            #     cv2.imshow("frame", orig_im)
            # # 后退300cm
            # elif i == 170:
            #     flight_action.wait_for_completed()
            #     flight_action = tl_flight.go(x=-400, y=0, z=0, speed=60)
            #     cv2.imshow("frame", orig_im)
            # # 左移600cm
            # elif i == 200:
            #     flight_action.wait_for_completed()
            #     flight_action = tl_flight.go(x=0, y=450, z=0, speed=60)
            #     cv2.imshow("frame", orig_im)
            
            # # look for nearest mid card 
            # elif i == 230:
            #     flight_action.wait_for_completed()
            #     flight_action = tl_flight.go(x=0, y=0, z=100, speed=40, mid1="m-2")
            #     cv2.imshow("frame", orig_im)

            # # 降落
            # elif i == 250:
            #     flight_action.wait_for_completed()
            #     flight_action = tl_flight.land()
            #     cv2.imshow("frame", orig_im)
            # else:
            #     cv2.imshow("frame", orig_im)
            # # 向前飞50厘米，向后飞50厘米
            # # tl_flight.forward(distance=50).wait_for_completed()
            # # tl_flight.backward(distance=50).wait_for_completed()

            # # Action End
            
            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            
        else:
            break   

    # for i in range(0, 302):
    #     img = tl_camera.read_cv2_image()
    #     cv2.imshow("Drone", img)
    #     cv2.waitKey(1)
    cv2.destroyAllWindows()
    tl_camera.stop_video_stream()

    #stop motor spinning
    # tl_flight.motor_off()

    print("test successfully!")

    tl_drone.close()
