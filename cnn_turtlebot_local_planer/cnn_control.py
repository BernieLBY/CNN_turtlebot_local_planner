#!/usr/bin/python
# coding=UTF-8
import cv2
import numpy as np
import rospy
import torch
import torch.nn as nn
import torch.utils.data.dataloader as DataLoader
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from PIL import Image
from sensor_msgs.msg import Image
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
import new_predict
import PIL
move_cmd = Twist()

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (3, 160, 120)
            nn.Conv2d(
                in_channels=3,      # input height
                out_channels=32,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape (32, 160, 120)
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (32, 80, 60)
        )
        self.conv2 = nn.Sequential(  # input shape (32, 80, 60)
            nn.Conv2d(32, 32, 5, 1, 2),  # output shape (32, 80, 60)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # output shape (32, 40, 30)
        )
        self.conv3 = nn.Sequential(  # input shape (32, 80, 60)
            nn.Conv2d(32, 64, 5, 1, 2),  # output shape (32, 80, 60)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # output shape (32, 80, 60)
        )
        self.out = nn.Linear(64 * 20 * 15, 5)   # fully connected layer, output 10 classes
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output


class image_converter:
    def __init__(self):
        self.image_pub=rospy.Publisher("/cmd_vel_mux/input/navi",Twist,queue_size=10)
        self.bridge=CvBridge()
        self.image_sub=rospy.Subscriber("/camera/depth/image_raw",Image,self.callback)
    

    def callback(self,data):
        depth_image = self.bridge.imgmsg_to_cv2(data, "passthrough")
        #depth_image=depth_image*1000

        # Convert the depth image to a Numpy array since most cv2 functions require Numpy arrays.
        # depth_array = np.array(depth_image, dtype=np.float32)
        # cv_image_resized = cv2.resize(depth_array, (160,120), interpolation = cv2.INTER_CUBIC)
        
        depth_min = np.nanmin(depth_image)
        depth_max = np.nanmax(depth_image)
        #depth_min = 120
        #depth_max = 4000
        


        depth_img = depth_image.copy()
        depth_img[np.isnan(depth_image)] = depth_max
        depth_img = ((depth_img - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        cv2.imshow("Depth Image", depth_img)
        cv2.waitKey(1)
        # for row in range(120):            #遍历高
        #     for col in range(160):         #遍历宽
        #         depth = cv_image_resized[row, col]
        #         if depth<0.08:
        #             depth=0.08
        #         elif depth>5000:
        #             depth=5000
        #         else:
        #             cv_image_resized[row, col]=depth
        #             #max_v=np.max(cv_image_resized) 
        #             cv_image_resized=cv_image_resized/6
        #cv_image_resized=cv_image_resized/10
        #where_are_nan = np.isnan(cv_image_resized)
        #cv_image_resized[where_are_nan] = 0
        #cv_image_resized[row, col]=depth
        depth_img = cv2.resize(depth_img, (160,120), interpolation = cv2.INTER_CUBIC)
        cv_image = cv2.cvtColor(depth_img,cv2.COLOR_GRAY2BGR)
        #cv_image = cv2.resize(depth_array, (160,120), interpolation = cv2.INTER_CUBIC)
        #cv2.imshow('test',cv_image)
        #cv2.waitKey(1)
        #print(cv_image.shape)
        class_out=new_predict.predit_output(cv_image)
        if class_out==0:
            move_cmd.linear.x=0.0
            move_cmd.angular.z=-0.5
            print("全向右转")
        elif class_out==1:
            move_cmd.linear.x=0.2
            move_cmd.angular.z=-0.3
            print("右转")
        elif class_out==2:
            move_cmd.linear.x=0.2
            move_cmd.angular.z=0
            print("直走")
        elif class_out==3:
            move_cmd.linear.x=0.2
            move_cmd.angular.z=0.3
            print("左转")
        elif class_out==4:
            move_cmd.linear.x=0.0
            move_cmd.angular.z=0.5
            print("全向左转")
        self.image_pub.publish(move_cmd)

if __name__ == '__main__':
    rospy.init_node("new_cnn_control")
    rospy.loginfo("starting")
    image_converter()
    rospy.spin()
