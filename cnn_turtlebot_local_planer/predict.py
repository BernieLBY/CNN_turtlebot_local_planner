#!/usr/bin/python
# coding=UTF-8
import torch
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import numpy

loader = transforms.Compose([transforms.ToTensor()])
def image_loader(cv_image):
    """load image, returns cuda tensor"""
    #image = Image.open(image_name).convert('RGB')
    image = loader(cv_image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image  #assumes that you're using GPU

''' net1=nn.Sequential(
    nn.Conv2d(3,32,5,1,2),      # output shape (32, 160, 120)
    nn.ReLU(),    # activation
    nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (32, 80, 60)
    nn.Conv2d(32, 32, 5, 1, 2),  # output shape (32, 80, 60)
    nn.ReLU(),  # activation
    nn.MaxPool2d(kernel_size=2),  # output shape (32, 40, 30)
    nn.Conv2d(32, 64, 5, 1, 2),  # output shape (32, 80, 60)
    nn.ReLU(),  # activation
    nn.MaxPool2d(kernel_size=2),  # output shape (32, 80, 60)
    nn.Linear(64 * 20 * 15, 5)   # fully connected layer, output 10 classes
) '''
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

def predit_output(cv_image):
    net2 = CNN()
    #net2=torch.load("/home/ethan/leiTai/cnn.pkl")
    net2.load_state_dict(torch.load("./cnn_param.pkl"))
    #loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
    image = image_loader(cv_image)
    output =net2(image)
    pred = torch.max(output, 1)[1]#1表示1维 ，把那一行全部提取出来，第二个1是提取的类别名称，也就是第2列
    pred_numpy=pred.data.numpy()
    
    #return (pred_numpy[0])
    #print (pred_numpy[0])
    return pred_numpy[0]