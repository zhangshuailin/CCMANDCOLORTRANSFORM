# -*- coding: utf-8 -*-
"""
http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html
Created on Tue Oct 10 09:32:32 2024
@author: Mr Zhang
"""

import rawpy
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
import os
import struct
from PIL import Image




M_xyz2rgb=np.array([[3.24096994,-1.53738318,-0.49861076],
                   [-0.96924364,1.8759675,0.04155506],
                   [0.05563008,-0.20397695,1.05697151]])
M_rgb2xyz=np.array([[0.4123908 , 0.35758434, 0.18048079],
                    [0.21263901, 0.71516868, 0.07219231],
                    [0.01933082, 0.11919478, 0.95053216]])

#%% x-rite 色彩标准
lab_ideal=np.array( # X-Rite官网提供的LAB色彩真值，D50
     [[37.986,13.555,14.059],
      [65.711,18.13,17.81],
      [49.927,-4.88,-21.925],
      [43.139,-13.095,21.905],
      [55.112,8.844,-25.399],
      [70.719,-33.397,-0.199],
      [62.661,36.067,57.096],
      [40.02,10.41,-45.964],
      [51.124,48.239,16.248],
      [30.325,22.976,-21.587],
      [72.532,-23.709,57.255],
      [71.941,19.363,67.857],
      [28.778,14.179,-50.297],
      [55.261,-38.342,31.37],
      [42.101,53.378,28.19],
      [81.733,4.039,79.819],
      [51.935,49.986,-14.574],
      [51.038,-28.631,-28.638],
      [96.539,-0.425,1.186],
      [81.257,-0.638,-0.335],
      [66.766,-0.734,-0.504],
      [50.867,-0.153,-0.27],
      [35.656,-0.421,-1.231],
      [20.461,-0.079,-0.973]],dtype='float32')
      
def gamma(x,colorspace='sRGB'): #Gamma变换
    y=np. zeros (x. shape)
    y[x>1]=1
    if colorspace in ( 'sRGB', 'srgb'):
      y[(x>=0)&(x<=0.0031308)]=(323/25*x[ (x>=0)&(x<=0.0031308)])
      y[(x<=1)&(x>0.0031308)]=(1.055*abs(x[ (x<=1)&(x>0.0031308)])**(1/2.4)-0.055)
    return y

def gamma_reverse(x,colorspace= 'sRGB'): #逆Gamma变换
    y=np.zeros(x.shape)
    y[x>1]=1
    if colorspace in ('sRGB', 'srgb'):
       y[(x>=0)&(x<=0.04045)]=x[(x>=0)&(x<=0.04045)]/12.92
       y[(x>0.04045)&(x<=1)]=((x[(x>0.04045)&(x<=1)]+0.055)/1.055)**2.4
    return y


#其它参考“http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html”
#“http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html”
#%%将colorchecker D50 xyY/Lab 值转成srgb值,参考“https://zhuanlan.zhihu.com/p/693596337”
if __name__=='__main__':
    #矩阵逆
    #C1 = np.dot(C, np.linalg.inv(A))
    XYZ_WP_C  = np.array([[98.074, 100, 118.232]])
    XYZ_WP_D50= np.array([[96.422, 100, 82.521]])
    XYZ_WP_D65= np.array([[95.047, 100, 108.883]])
    #xyz白点转换矩阵，参考“RGB coordinates of the Macbeth ColorChecker”
    XYZ_WP_D50_2_XYZ_WP_D65_M=np.array([[0.9556, -0.0230, 0.0632],
                                        [-0.0283, 1.0099, 0.0210],
                                        [0.0123, -0.0205, 1.3299]])

    #print(XYZ_WP_D50_2_XYZ_WP_D65_M@XYZ_WP_D50.T)
    #@@@@@@D50 Lab2XYZ
    f_reverse=lambda t : (t>(6/29))*(t**3)+(t<=(6/29))*(3*((6/29)**2)*(t-4/29))
    Xn=XYZ_WP_D50[0,0]/100
    Yn=XYZ_WP_D50[0,1]/100
    Zn=XYZ_WP_D50[0,2]/100
    lab=lab_ideal.transpose()
    xyz_d50=np.vstack((Xn*f_reverse((lab[0,:]+16)/116+lab[1,:]/500),
                   Yn*f_reverse((lab[0,:]+16)/116),
                   Zn*f_reverse((lab[0,:]+16)/116-lab[2,:]/200)))
    #print(XYZ_d50.T)
    #@@@@@@计算Bradford 3by3矩阵 d50->d65
    M=np.array([[0.9870, -0.1471, 0.1600],
                [0.4323, 0.5184, 0.0493],
                [-0.0085, 0.0400, 0.9685]])
    M_T =  np.linalg.inv(M)
    XYZdw = np.array([[96.422/100, 100/100, 82.521/100]]).T
    XYZsw = np.array([[95.047/100, 100/100, 108.883/100]]).T
    RGBdw = M_T@XYZdw
    RGBsw = M_T@XYZsw
    RGBdsoverdw = RGBsw/RGBdw
    #print(RGBdwoversw)
    RGBdsoverdw_3by3=np.array([[RGBdsoverdw[0,0], 0, 0],
                       [0, RGBdsoverdw[1,0], 0],
                       [0, 0, RGBdsoverdw[2,0]]])
    bradford3by3_d50_2_d65=M@RGBdsoverdw_3by3@np.linalg.inv(M)
    #也可直接查表“http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html”
    #print(bradford3by3_d50_2_d65)           
    
    #@@@@@@XYZ_D50->D65
    xyz_d65=bradford3by3_d50_2_d65@xyz_d50
    #print(xyz_d65)
    
    
    #@@@@@@XYZ_D65->Lab_D65
    f=lambda t : (t>((6/29)**3))*(t**(1/3))+(t<=(6/29)**3)*(29*29/6/6/3*t+4/29)
    Xn=XYZ_WP_D65[0,0]/100
    Yn=XYZ_WP_D65[0,1]/100
    Zn=XYZ_WP_D65[0,2]/100
    L=116*f(xyz_d65[1,:]/Yn)-16
    a=500*(f(xyz_d65[0,:]/Xn)-f(xyz_d65[1,:]/Yn))
    b=200*(f(xyz_d65[1,:]/Yn)-f(xyz_d65[2,:]/Zn))
    Lab=np.vstack((L,a,b))
    #print(Lab.T)
    
    
    #@@@@@@XYZ_D50->SRGB-LINEAR
    SRGB_Linear=M_xyz2rgb@xyz_d65
    #print(SRGB_Linear.T*255)
    SRGB_gamma=gamma(SRGB_Linear,colorspace='sRGB')
    #print(SRGB_gamma.T*255)
    




   