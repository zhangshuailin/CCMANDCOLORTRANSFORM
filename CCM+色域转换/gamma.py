# -*- coding: utf-8 -*-
"""
https://zhuanlan.zhihu.com/p/413851281
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

#%% 读取Raw图，预处理，转浮点，OB
if __name__=='__main__':
    x = [i/255 for i in range(256)]
    y = [0 for i in range(256)]
    for i in range(256):
        if((x[i]>=0)&(x[i]<=0.0031308)):
            y[i]=323/25*x[i]
        elif((x[i]<=1)&(x[i]>0.0031308)):
            y[i]=(1.055*abs(x[i])**(1/2.4)-0.055)
    print(x)
    print(y)