#this program needs python 2.7, matplotlib and numpy


import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline


## z=z**2+z0
##buzzsaw 0.001643721971153+0.822467633298876i scale 4 x 10-11
##curreent best algoritm
## warning : code takes a long time to run

xcen=0.001643721971153
ycen=0.822467633298876
scale=5e-11
import time

xbeg=xcen-(scale/2)
xend=xcen+(scale/2)
ybeg=ycen-(scale/2)
yend=ycen+(scale/2)
pi=np.pi
n=2048
n2=2048
xx = np.linspace(xbeg,xend,n2, dtype=np.complex128)
yy = np.linspace(ybeg*1j,yend*1j,n, dtype=np.complex128)
xx,yy=np.meshgrid(xx,yy)

z0=xx+yy
z=xx*0
z2=xx*0
mset=xx*0
msetbol=xx*0
j=0
qq=10000
breakpixellim=100
t1=time.time()

qqq=np.array([1000,1000])

while qq > breakpixellim and j < 2500:
    j+=1
    z=(z**2)+z0
    z2=np.abs(z)>2
    if np.sum(z2)>0:
        jj=np.sin((2*j*pi)/100)
        msetbol=mset != 0
        msetbol=(~msetbol)*z2 
        mset=mset+(msetbol*jj)
        z=z*(~z2)
        qq=mset==0
        qq=np.sum(qq)

print time.time()-t1, j
# mset[0:10]=20
plt.figure(figsize=(32,32),dpi=100)
plt.imshow(np.abs(mset))