# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:50:23 2023

@author: Mi Xiaoshi
"""

#%% 二维查找表
from scipy.interpolate import interp2d,interpn,LinearNDInterpolator
import numpy as np
import matplotlib.pyplot as plt
import time

def gamma(x,colorspace='sRGB'): #Gamma变换
    y=np. zeros (x. shape)
    y[x>1]=1
    if colorspace in ( 'sRGB', 'srgb'):
        y[(x>=0)&(x<=0.0031308)]=(323/25*x[ (x>=0)&(x<=0.0031308)])
        y[(x<=1)&(x>0.0031308)]=(1.055*abs(x[ (x<=1)&(x>0.0031308)])**(1/2.4)-0.055)
    elif colorspace in ('my'):  
        y[ (x>=0)&(x<=1)]=(1.42*(1-(0.42/(x[(x>=0)&(x<=1)]+0.42))))
    elif colorspace in ('P3'):  
        y[ (x>=0)&(x<=1)]=x[ (x>=0)&(x<=1)]**(1/2.6)
    elif (type(colorspace)==float)|(type(colorspace)==int):
        beta=colorspace
        y[ (x>=0)&(x<=1)]=((1+beta)*(1-(beta/(x[(x>=0)&(x<=1)]+beta))))
    return y

def gamma_reverse(x,colorspace= 'sRGB'): #逆Gamma变换
    y=np.zeros(x.shape)
    y[x>1]=1
    if colorspace in ('sRGB', 'srgb'):
        y[(x>=0)&(x<=0.04045)]=x[(x>=0)&(x<=0.04045)]/12.92
        y[(x>0.04045)&(x<=1)]=((x[(x>0.04045)&(x<=1)]+0.055)/1.055)**2.4
    elif colorspace in ('my'):
        y[(x>=0)&(x<=1)]=0.42/(1-(x[(x>=0)&(x<=1)]/1.42))-0.42         
    return y


M_xyz2rgb=np.array([[3.24096994,-1.53738318,-0.49861076],
                    [-0.96924364,1.8759675,0.04155506],
                    [0.05563008,-0.20397695,1.05697151]])
M_rgb2xyz=np.array([[0.4123908 , 0.35758434, 0.18048079],
                    [0.21263901, 0.71516868, 0.07219231],
                    [0.01933082, 0.11919478, 0.95053216]])
def rgb2lab(img,whitepoint='D65'): #rgb转lab
    if (img.ndim==3):
        if (img.shape[2]==3):
            (rgb,func_reverse)=im2vector(img)
    elif (img.ndim==2):
        if (img.shape[1]==3):
            rgb=img
            func_reverse=lambda x : x
        elif (img.shape[0]>80)&(img.shape[1]>80):
            img=np.dstack((img,img,img))
            (rgb,Func_reverse)=im2vector(img)
    rgb=rgb.transpose()
    rgb=gamma_reverse(rgb,colorspace='sRGB')
    xyz=M_rgb2xyz@rgb
    xyz=xyz.transpose()
    f=lambda t : (t>((6/29)**3))*(t**(1/3))+\
        (t<=(6/29)**3)*(29*29/6/6/3*t+4/29)
    if whitepoint=='D65':
        Xn=95.047/100
        Yn=100/100
        Zn=108.883/100
    L=116*f(xyz[:,1]/Yn)-16
    a=500*(f(xyz[:,0]/Xn)-f(xyz[:,1]/Yn))
    b=200*(f(xyz[:,1]/Yn)-f(xyz[:,2]/Zn))
    Lab=np.vstack((L,a,b)).transpose()
    img_out=func_reverse(Lab)
    return img_out

def lab2rgb(img,whitepoint='D65'): #lab转rgb
    if (img.ndim==3):
        if (img.shape[2]==3):
            (lab,func_reverse)=im2vector(img)
    elif (img.ndim==2):
        if (img.shape[1]==3):
            lab=img
            func_reverse=lambda x : x
        elif (img.shape[0]>80)&(img.shape[1]>80):
            img=np.dstack((img,img,img))
            (lab,Func_reverse)=im2vector(img)
    lab=lab.transpose()
    if whitepoint=='D65':
        Xn=95.047/100
        Yn=100/100
        Zn=108.883/100
    f_reverse=lambda t : (t>(6/29))*(t**3)+\
        (t<=(6/29))*(3*((6/29)**2)*(t-4/29))
    xyz=np.vstack((Xn*f_reverse((lab[0,:]+16)/116+lab[1,:]/500),
                   Yn*f_reverse((lab[0,:]+16)/116),
                   Zn*f_reverse((lab[0,:]+16)/116-lab[2,:]/200) ))
    rgb=M_xyz2rgb@xyz
    rgb=rgb.transpose()
    rgb=gamma(rgb,colorspace='sRGB')
    rgb_out=func_reverse(rgb)
    return rgb_out


np.random.seed(0)
u,v=np.meshgrid(np.linspace(-0.5,0.5,11),np.linspace(-0.5,0.5,11))
z=(np.random.rand(11,11)-0.5)*0.04
f=LinearNDInterpolator((u.flatten(),v.flatten()),z.flatten())
plt.figure()
uu=np.linspace(-0.5,0.5,6001)
vv=np.linspace(-0.5,0.5,4001)
u_new,v_new=np.meshgrid(uu,vv)
start_time=time.time()
z_new=f(u_new,v_new)
print('插值耗时:{}s'.format(time.time()-start_time))
plt.plot(u_new+z_new,v_new,'r.')
plt.plot(u+z,v,'k.')

#%%
#%% x-rite 色彩标准

lab_ideal=np.array( # X-Rite官网提供的LAB色彩真值
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

rgb_ideal=lab2rgb(lab_ideal)
M_rgb2yuv=np.array([[0.299,0.587,0.114],
                    [-0.169,-0.331,0.499],
                    [0.499,-0.418,-0.081]])
M_yuv2rgb=np.array([[9.99999554e-01, -4.46062343e-04,1.40465882],
                     [9.99655449e-01, -3.44551299e-01,-7.15683665e-01],
                     [1.00177531e+00,1.77530689,9.94081794e-04]])

yuv_ideal=(M_rgb2yuv@gamma_reverse(rgb_ideal).T).T
#%%
def fig_yuv(N=501):
    u,v=np.meshgrid(np.linspace(-0.5,0.5,N),np.linspace(0.5,-0.5,N))
    Y=np.ones(u.shape)*0.5
    img_background=M_yuv2rgb @ np.vstack((Y.flatten(),u.flatten(),v.flatten()))
    img_background=img_background.T.reshape((501,501,3))
    fig=plt.figure(tight_layout=True)
    h_ax=plt.axes(xlim=[-0.5,0.5],ylim=[-0.5,0.5],xticks=np.arange(-0.5,0.5,0.1),yticks=np.arange(-0.5,0.5,0.1))
    plt.grid()
    plt.imshow(isp.gamma(img_background),extent=(-0.5,0.5,-0.5,0.5))
    
    return fig
#%% YUV色谱图
import me_CCM_optimizer as isp
fig_yuv()
for idx,yuv in enumerate(yuv_ideal):
    plt.plot(yuv[1],yuv[2],'ks')
    plt.text(yuv[1]+0.01,yuv[2]+0.01,'{}'.format(idx+1))
#%%
rgb_mean=np.array([
       [0.05604649, 0.03763649, 0.03084474],
       [0.23760724, 0.16160099, 0.13566643],
       [0.1005693 , 0.12852928, 0.17241962],
       [0.0717919 , 0.08149751, 0.054678  ],
       [0.13842899, 0.13839421, 0.1935168 ],
       [0.13314316, 0.21936157, 0.20805185],
       [0.21592635, 0.09212222, 0.04138732],
       [0.05916373, 0.08043833, 0.16416494],
       [0.26455184, 0.09372092, 0.08674786],
       [0.06830533, 0.04834968, 0.07834507],
       [0.22371823, 0.25566936, 0.1190641 ],
       [0.28020669, 0.16114731, 0.06404834],
       [0.02021428, 0.03257127, 0.08527481],
       [0.07551719, 0.13500342, 0.07635126],
       [0.19093955, 0.05140478, 0.03901855],
       [0.44907791, 0.32956755, 0.12193429],
       [0.22170259, 0.09314077, 0.14076718],
       [0.0540875 , 0.1253151 , 0.17184836],
       [0.32161676, 0.31624249, 0.31230679],
       [0.26459358, 0.26259166, 0.26249043],
       [0.18669077, 0.18759722, 0.18778614],
       [0.09848111, 0.09848111, 0.09848111],
       [0.04052817, 0.04142432, 0.04216452],
       [0.01271575, 0.0130849 , 0.01328809]])    

ccm=np.array([[ 1.8693276 , -0.53901856, -0.33030905],
              [-0.20339718,  1.50823653, -0.30483935],
              [-0.00366836, -0.60574533,  1.60941369]])
awb_para=[rgb_mean[21,1]/rgb_mean[21,0],1,rgb_mean[21,1]/rgb_mean[21,2]]
rgb_mean=isp.awb(rgb_mean,awb_para)
ae_comp=1.86942235
yuv_mean=(M_rgb2yuv@(isp.ccm(rgb_mean*ae_comp,ccm).T)).T
for idx,yuv in enumerate(yuv_mean):
    plt.plot(yuv[1],yuv[2],'ko')
    plt.text(yuv[1]+0.01,yuv[2]+0.01,'{}'.format(idx+1))

