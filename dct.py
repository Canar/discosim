#!/usr/bin/python3
from scipy.fftpack import dct, idct

# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')    

from skimage.io import imread
from skimage.io import imsave
#from skimage.exposure import equalize_hist
import skimage.exposure as ex
#from skimage.exposure import equalize_adapthist
from skimage import img_as_uint
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pylab as plt
from sys import argv
import cv2
import random
import math

def to_cs(image): return cv2.cvtColor(image,cv2.COLOR_RGB2Lab)
def from_cs(image): return cv2.cvtColor(image,cv2.COLOR_Lab2RGB)

imrgb=to_cs(imread(argv[1])).transpose(2,0,1)
dcta=np.empty(imrgb.shape)
print('image loaded into colorspace')

for c in range(3):
    dcta[c]=dct2(imrgb[c])
print('dct calculated')

calc_extents=False
if calc_extents:
    extent=np.empty((3,4,2))
    tll=99999
    tlh=0


    for c in range(3):
        extent[c][0][1]=extent[c][0][0]=dcta[c][0][0]
        extent[c][1][1]=extent[c][1][0]=dcta[c][w][0]
        extent[c][2][1]=extent[c][2][0]=dcta[c][0][h]
        extent[c][3][1]=extent[c][3][0]=dcta[c][w][h]

    checkbox=8
    for c in range(3):
        for x in range(checkbox):
            for y in range(checkbox):
                extent[c][0][0]=min(extent[c][0][0],dcta[c][ x ][ y ])
                extent[c][0][1]=max(extent[c][0][1],dcta[c][ x ][ y ])
                extent[c][1][0]=min(extent[c][1][0],dcta[c][w-x][ y ])
                extent[c][1][1]=max(extent[c][1][1],dcta[c][w-x][ y ])
                extent[c][2][0]=min(extent[c][2][0],dcta[c][ x ][h-y])
                extent[c][2][1]=max(extent[c][2][1],dcta[c][ x ][h-y])
                extent[c][3][0]=min(extent[c][3][0],dcta[c][w-x][h-y])
                extent[c][3][1]=max(extent[c][3][1],dcta[c][w-x][h-y])

    print(extent)

busted_lerp=False
if busted_lerp:
    adj=np.empty((3,3,2))

    for c in range(3):
        ro=extent[c][0][1]-extent[c][0][0]

        r=extent[c][1][1]-extent[c][1][0]
        m=ro/r
        b=extent[c][0][0]-(m*extent[c][1][0])
        adj[c][0][0]=m
        adj[c][0][1]=b

        r=extent[c][2][1]-extent[c][2][0]
        m=ro/r
        b=extent[c][0][0]-(m*extent[c][2][0])
        adj[c][1][0]=m
        adj[c][1][1]=b

        r=extent[c][3][1]-extent[c][3][0]
        m=ro/r
        b=extent[c][0][0]-(m*extent[c][3][0])
        adj[c][2][0]=m
        adj[c][2][1]=b


    dctadj=dcta.copy()
    for c in range(dctadj.shape[0]):
        for x in range(dctadj.shape[1]):
            for y in range(dctadj.shape[2]):
                xf=x/w
                yf=y/h
                of=1-((xf+yf)/2)
                v=dcta[c][x][y]
                dctadj[c][x][y]=v*of + xf*(v*adj[c][0][0] + adj[c][0][1])/2 + yf*(v*adj[c][1][0]+adj[c][1][1])/2

    imsave('dctadj.png',dctadj.transpose(1,2,0))


proc_thresh=False
if proc_thresh:
    thresh=int(argv[2])
    p=3
    rtf=1.1
    rt=min(h,w)/rtf/1.1
    for c in range(imrgb.shape[0]):
        for x in range(imrgb.shape[1]):
           for y in range(imrgb.shape[2]):
                #test=(x*x+y*y)>(thresh*thresh) #circle
                #test=(pow(x,p)*pow(y,p))>pow(thresh,p) #power
                #test=x>thresh or y>thresh #square
                #test=((x+y)/2)>thresh #diag
                #test=x*y>thresh*thresh #inverse
                #test=(pow(x,p)+pow(y,p))>pow(thresh,p) #power
                #test=min(pow(x,2)*pow(y,2)*4,pow(x,4)+pow(y,4))>pow(thresh,4) #idklul
                #test=abs(dcta[c][x][y])<thresh #smallval
                t=rt if c==0 else rt/rtf
                #test=(x*x+y*y)>(t*t) #circle
                test=x>t or y>t #square
                if test:
                    dcta[c][x][y]=0


w=dcta.shape[1]-1
h=dcta.shape[2]-1
match 'soft': #filter
    case 'soft':
        for (c,x,y),v in np.ndenumerate(dcta):
            d=1-((x/w)**2 + (y/h)**2)**0.5
            d=0 if d < 0 else d
            dv=d**0.5
            dcta[c][x][y]=dcta[c][x][y]*dv
        print('filter applied')
    case 'noop':
        print('meow')


write_dct=True
write_dct_log=True
extra_rescaling_approaches=False
if write_dct:    
    dctri=dcta.copy()

    if write_dct_log:
        for (c,x,y),v in np.ndenumerate(dcta):
            dctri[c][x][y]=math.log(dcta[c][x][y]**2+1)**0.8

    #dctri=ex.equalize_hist(dctri)

    if extra_rescaling_approaches:
        for c in range(3):
            #dctout[c]=ex.equalize_adapthist(ex.rescale_intensity(dcta[c]).astype(np.float64),kernel_size=[h/3,w/3])
            dctri=ex.rescale_intensity(dctout[c],out_range=(1.0,1000000.0))
            dctri=ex.equalize_hist(dctri)
            #dctri=ex.rescale_intensity(dcta[c],out_range=(0.0,1.0))
            #dctout[c]=ex.adjust_gamma(dctri,0.1)
            #dctout[c]=ex.adjust_log(ex.rescale_intensity(dcta[c],out_range=(0.0,1.0)))
            #dctout[c]=ex.adjust_log(dctri)
            #dctout[c]=ex.equalize_hist(dctri)
            #dctout[c]=dctri
            #dctout[c]=ex.adjust_gamma(ex.equalize_hist(dcta[c]),4)
            xr=int(w*random.random())
            yr=int(h*random.random())
            print([dcta[c][xr][yr],dctout[c][xr][yr]])
            #dctout[c]=ex.adjust_sigmoid(ex.rescale_intensity(dcta[c],out_range=(0.0,1.0)).astype(float))
            #dctout[c]=ex.equalize_adapthist(ex.rescale_intensity(dcta[c],in_range=(0.0,100.0)).astype(float))
            #dctout[c]=ex.equalize_adapthist(ex.rescale_intensity(dcta[c],in_range=(0.0,100.0)).astype(float))
            #dctout[c]=ex.equalize_adapthist(ex.rescale_intensity(dcta[c],in_range=(0.0,1.0)))
            #dctout[c]=equalize_adapthist(equalize_hist(dcta[c]))

    imsave('dctout.png',dctri.transpose(1,2,0))
    #done

post=np.empty(imrgb.shape)
for c in range(imrgb.shape[0]):
    post[c]=idct2(dcta[c])

post=ex.rescale_intensity(post,out_range=(0,255)).astype(np.uint8)
imsave('post.png',from_cs(post.transpose(1,2,0)))

# print(imF.ndim)
# print(imF.shape)
# im1 = idct2(imF)

# check if the reconstructed image is nearly equal to the original image
# np.allclose(im, im1)
# True

# plot original and reconstructed images with matplotlib.pylab
# plt.gray()
#plt.subplot(121), plt.imshow(im), plt.axis('off'), plt.title('input', size=20)
# plt.subplot(122), plt.imshow(im1), plt.axis('off'), plt.title('output', size=20)
# plt.show()
