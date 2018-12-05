## z=z**2+z0

##dueling dragon -1.6735519336462-0.0003236318510i scale 16 x 10-9

import matplotlib.pyplot as plt
import numpy as np
xcen=-1.6735519336462
ycen=0.0003236318510
scale=16e-9
import time

xbeg=-1.6735519336462-(scale/2)
xend=-1.6735519336462+(scale/2)
ybeg=0.0003236318510-(scale/2)
yend=0.0003236318510+(scale/2)
pi=np.pi
n=800
n2=800
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


while qq > breakpixellim :
    j+=1
    z=(z**2)+z0
    z2=np.abs(z)>2
    if np.sum(z2)>0:
        jj=np.sin((2*j*pi)/100)
        msetbol=mset > 0
        msetbol=(~msetbol)*z2 
        mset=mset+(msetbol*jj)
        z=z*(~z2)
        qq=mset==0
        qq=np.sum(qq)

print time.time()-t1, j
# mset[0:10]=20
plt.figure(figsize=(10,10),dpi=100)
plt.imshow(np.abs(mset))

