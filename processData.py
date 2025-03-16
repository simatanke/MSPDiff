import os
import cv2
import numpy as np
import math


path_0 = r'E:\DataSet\PI-ND dataset\0_image'
path_45 = r'E:\DataSet\PI-ND dataset\45_image'
path_90 = r'E:\DataSet\PI-ND dataset\90_image'
path_135 = r'E:\DataSet\PI-ND dataset\135_image'



## 获取 I Q U V 值
def getIQUV(img_0, img_45, img_90, img_135):
    I = (img_0 + img_45 + img_90 + img_135) * 0.5
    Q = (img_0 - img_90)
    U = (img_45 - img_135)
    V = (img_0 + img_90 - img_45 - img_135)
    return I, Q, U, V


def getDoP(I, Q, U, V):
    Dop = np.sqrt(Q ** 2 + U ** 2 + V ** 2) / (I + 1e-10)  # 添加小量以避免除以零
    return Dop


def getAoP(I, Q, U, V):
    Aop = 0.5 * np.arctan2(U, Q)  # 使用 np.arctan2 处理 AoP
    return Aop


def getDoLP(I, Q, U, V):
    DoLp = np.sqrt(Q ** 2 + U ** 2) / (I + 1e-10)  # 添加小量以避免除以零
    return DoLp


def getAoLP(I, Q, U, V):
    AoLp = 0.5 * np.arctan2(U, Q)  # 使用 np.arctan2 处理 AoLP
    return AoLp


## 循环处理文件夹中的所有图像 *.bmp
for file in os.listdir(path_0):
    if file.endswith(".bmp"):
        img_0 = cv2.imread(os.path.join(path_0, file), cv2.IMREAD_GRAYSCALE).astype(np.float32)  # 将图像转换为浮点型
        img_45 = cv2.imread(os.path.join(path_45, file), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img_90 = cv2.imread(os.path.join(path_90, file), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img_135 = cv2.imread(os.path.join(path_135, file), cv2.IMREAD_GRAYSCALE).astype(np.float32)




        I, Q, U, V = getIQUV(img_0, img_45, img_90, img_135)
        DoP = getDoP(I, Q, U, V)
        AoP = getAoP(I, Q, U, V)
        DoLP = getDoLP(I, Q, U, V)
        # AoLP = getAoLP(I, Q, U, V)

        ## 可视化 I,Q,U,V DoP AoP DoLP AoLP
        cv2.imshow('I', (I / I.max() * 255).astype(np.uint8))  # 缩放并转换数据类型为 uint8
        cv2.imshow('Q', (Q / Q.max() * 255).astype(np.uint8))
        cv2.imshow('U', (U / U.max() * 255).astype(np.uint8))
        cv2.imshow('V', (V / V.max() * 255).astype(np.uint8))
        cv2.imshow('DoP', (DoP / DoP.max() * 255).astype(np.uint8))
        cv2.imshow('AoP', ((AoP + np.pi / 2) / (np.pi) * 255).astype(np.uint8))  # 将 AoP 转换到 0-255 范围
        cv2.imshow('DoLP', (DoLP / DoLP.max() * 255).astype(np.uint8))
        # cv2.imshow('AoLP', ((AoLP + np.pi / 2) / (np.pi) * 255).astype(np.uint8))  # 将 AoLP 转换到 0-255 范围
        cv2.waitKey(0)
