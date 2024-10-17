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

M_rgb2xyz=np.array([[0.4123908 , 0.35758434, 0.18048079],
                    [0.21263901, 0.71516868, 0.07219231],
                    [0.01933082, 0.11919478, 0.95053216]])

#其它参考“http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html”
#“http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html”
#%%将colorchecker D50 xyY/Lab 值转成srgb值,参考“https://zhuanlan.zhihu.com/p/693596337”
if __name__=='__main__':
    
    CAMERARGB2SRGBLINE_D65=np.array([[1.332749, -0.40975, 0.077011],
                                     [-0.03831, 1.120957, -0.08263],
                                     [0.124537, -0.71389, 1.589355]])
    CAMERARGB2SRGBLINE_A  =np.array([[1.678215, -0.52838, -0.14983],
                                     [-0.19981, 1.184738, 0.015072],
                                     [-0.03011, -1.11765, 2.147771]])
                                     
    CAMERARGB2XYZ_D65 = CAMERARGB2SRGBLINE_D65@M_rgb2xyz
    CAMERARGB2XYZ_A   = CAMERARGB2SRGBLINE_A  @M_rgb2xyz
    print(CAMERARGB2XYZ_D65)
    print(CAMERARGB2XYZ_A)
    print(np.linalg.inv(CAMERARGB2XYZ_D65))
    print(np.linalg.inv(CAMERARGB2XYZ_A))
    #A=np.array([[1, 2, 3],
    #            [4, 5, 6],
    #            [7, 7, 9]])
    #B=np.array([[2, 3, 4],
    #            [7, 8, 9],
    #            [1, 1, 3]])
    #AB=A@B
    #print(AB)
    #print(AB@np.linalg.inv(B))






   