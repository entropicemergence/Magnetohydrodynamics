import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from matplotlib import style
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pywt

start_time = time.time()

n=768

x = np.linspace(0.0, 8.0, n)
y = np.linspace(0.0, 8.0, n)
x,y=np.meshgrid(x,y)
pi=np.pi
dx=8.0/(n-1)
dy=8.0/(n-1)
dt=0.0002
eta=0.1

# initial condition

# Velocity
u = np.ones((n, n))  ##create a 1xn vector of 1's
v = np.ones((n, n))
u=u*0
v=v*0
# u[int(4.0 / dy): int(5.0 / dy + 1),
# int(4.0 / dx): int(5.0 / dx + 1)] = 2.0
# v[int(4.0 / dy): int(5.0 / dy + 1),
# int(4.0 / dx): int(5.0 / dx + 1)] = 2.0
# u[int(0.0 / dy): int(2.0 / dy + 1),
# int(0.0 / dx): int(2.0 / dx + 1)] = 2.5
# v[int(0.0 / dy): int(2.0 / dy + 1),
# int(0.0 / dx): int(2.0 / dx + 1)] = 2.5
# u[int(7.0 / dy): int(8.0 / dy + 1),
# int(2.0 / dx): int(3.0 / dx + 1)] = 2.5
# v[int(7.0 / dy): int(8.0 / dy + 1),
# int(2.0 / dx): int(3.0 / dx + 1)] = 2.5
# u[int(3.0 / dy): int(4.0 / dy + 1),
# int(6.0 / dx): int(7.0 / dx + 1)] = 2.5
# v[int(3.0 / dy): int(4.0 / dy + 1),
# int(6.0 / dx): int(7.0 / dx + 1)] = 2.5
xx = np.linspace(0.0, n, n)
yy = np.linspace(0.0, n, n)
xx,yy=np.meshgrid(xx,yy)
# Xo=40
# Yo=75
# Xo1=60
# Yo1=25
# rs=((xx-Xo)**2)+((yy-Yo)**2)
# rs1=((xx-Xo1)**2)+((yy-Yo1)**2)
# u=((yy-Yo)/rs)-((yy-Yo1)/rs1)   #double vortx
# v=-((xx-Xo)/rs)+((xx-Xo1)/rs1)
# u=((yy-Yo)/rs)                #single vortex
# v=-((xx-Xo)/rs)
u=-np.sin(y*pi/2)
v=np.sin(x*pi/2)

def Vcalc(u,v,rho,dt,dx,dy,UiP,UiN,VjP,VjN,UjP,UjN,ViP,ViN,p,Bx,By,rand1,rand2):
    Dux = (UiP - (2 * Uij) + UiN) / (dx ** 2)
    Duy = (UjP - (2 * Uij) + UjN) / (dy ** 2)
    Dvx = (ViP - (2 * Vij) + ViN) / (dx ** 2)
    Dvy = (VjP - (2 * Vij) + VjN) / (dy ** 2)
    aux = (Uij - UiN) / dx
    auy = (Uij - UjN) / dy
    avx = (Vij - ViN) / dx
    avy = (Vij - VjN) / dy
    PjP = p[2:, 1:(n - 1)]
    PjN = p[:(n - 2), 1:(n - 1)]
    PiP = p[1:(n - 1), 2:]
    PiN = p[1:(n - 1), :(n - 2)]
    Pi=(PiP-PiN)*(dt/(rho*2*dx))
    Pj=(PjP-PjN)*(dt/(rho*2*dy))

    Babs = ((Bx**2)+(By**2))/2
    Bxij = Bx[1:(n - 1), 1:(n - 1)]
    Byij = By[1:(n - 1), 1:(n - 1)]
    BxjN = Bx[:(n - 2), 1:(n - 1)]
    BxiN = Bx[1:(n - 1), :(n - 2)]
    ByjN = By[:(n - 2), 1:(n - 1)]
    ByiN = By[1:(n - 1), :(n - 2)]
    Bxx = (Bxij - BxiN) / dx
    Bxy = (Bxij - BxjN) / dy
    Byx = (Byij - ByiN) / dx
    Byy = (Byij - ByjN) / dy

    Bij = Babs[1:(n - 1), 1:(n - 1)]
    BjN = Babs[:(n - 2), 1:(n - 1)]
    BiN = Babs[1:(n - 1), :(n - 2)]
    Bdx = (dt*(Bij - BiN)) / dx
    Bdy = (dt*(Bij - BjN)) / dy
    Bupart=(dt * Bxij * Bxx) + (dt * Byij * Bxy)-Bdx
    Bvpart=(dt * Bxij * Byx) + (dt * Byij * Byy)-Bdy


    u[1:(n - 1), 1:(n - 1)]= Uij - (dt * Uij * aux) - (dt * Vij * auy) + (eta * ((dt * Dux) + (dt * Duy)))-Pi+Bupart+rand1
    v[1:(n - 1), 1:(n - 1)] = Vij - (dt * Uij * avx) - (dt * Vij * avy) + (eta * ((dt * Dvx) + (dt * Dvy)))-Pj+Bvpart+rand2
##    u[0,:]=u[-2,:];u[-1,:]=u[1,:]
##    u[:,0]=u[:,-2];u[:,-1]=u[:,1]
##    v[0, :] = v[-2, :];v[-1, :] = v[1, :]
##    v[:, 0] = v[:, -2];v[:, -1] = v[:, 1]
##
    u[0,:]=0;u[-1,:]=0
    u[:,0]=0;u[:,-1]=0
    v[0, :] = 0;v[-1, :] =0
    v[:, 0] = 0;v[:, -1] =0
    



    
    return u,v

# Density
rho=25/(36*pi)

# Pressure
p=x*0
b=y*0
def pbound(p):
    # p = np.lib.pad(U, ((1, 1), (1, 1)), 'constant', constant_values=((0, 0), (0, 0)))
    # p[:, -1] = p[:, -2]  ##dp/dy = 0 at x = 2
    # p[0, :] = p[1, :]  ##dp/dy = 0 at y = 0
    # p[:, 0] = p[:, 1]  ##dp/dx = 0 at x = 0
    # p[-1, :] = 0  ##p = 0 at y = 2
    p[0,:]=p[1,:];p[-1,:]=p[-2,:]
    p[:,0]=p[:,1];p[:,-1]=p[:,-2]

    return p

def bcalc(b,rho,dt,dx,dy,UiP,UiN,VjP,VjN,UjP,UjN,ViP,ViN):
    Dux = (UiP - UiN) / (2 * dx)
    Dvy = (VjP - VjN) / (2 * dy)
    Duy = (UjP - UjN) / (2 * dy)
    Dvx = (ViP - ViN) / (2 * dx)
    b[1:(n - 1), 1:(n - 1)]=rho*(((1/dt)*(Dux+Dvy))-(Dux**2)-(2*Duy*Dvx)-(Dvy**2))
    return b
def pcalc(p,dx,dy,b,npress):
    divider=2*((dx**2)+(dy**2))
    if i == 0:
        npress=1500
    for k in range(npress):                             #relaxing pressure

        PjP = p[2:, 1:(n - 1)]
        PjN = p[:(n - 2), 1:(n - 1)]
        PiP = p[1:(n - 1), 2:]
        PiN = p[1:(n - 1), :(n - 2)]
        p[1:(n - 1), 1:(n - 1)]=((((PiP+PiN)*(dy**2))+((PjP+PjN)*(dx**2)))/divider)-((((dx**2)*(dy**2))/divider)*b[1:(n - 1),1:(n - 1)])
        pbound(p)
    return p
# Vector Potential
A=np.cos(y*1*pi/2)+0.5*np.cos(x*2*pi/2)
Bx=A*0
By=A*0
eta2=0.1
def Acalc(A,dt,dx,dy,Uij,Vij,eta2):
    Aij = A[1:(n - 1), 1:(n - 1)]
    AjP = A[2:, 1:(n - 1)]
    AjN = A[:(n - 2), 1:(n - 1)]
    AiP = A[1:(n - 1), 2:]
    AiN = A[1:(n - 1), :(n - 2)]

    aAx = (Aij - AiN) / dx
    aAy = (Aij - AjN) / dy
    DAx = (AiP - (2 * Aij) + AiN) / (dx ** 2)
    DAy = (AjP - (2 * Aij) + AjN) / (dy ** 2)

    A[1:(n - 1), 1:(n - 1)] = Aij - (dt * Uij * aAx) - (dt * Vij * aAy) + (eta2 * ((dt * DAx) + (dt * DAy)))
    A[0,:]=A[-2,:];A[-1,:]=A[1,:]
    A[:,0]=A[:,-2];A[:,-1]=A[:,1]

    return A

#magnetic field
def Bcalc(A,dx,dy):
    Aij = A[1:(n - 1), 1:(n - 1)]
    AjN = A[:(n - 2), 1:(n - 1)]
    AiN = A[1:(n - 1), :(n - 2)]
    Bx[1:(n - 1), 1:(n - 1)]=(Aij-AjN)/dy
    By[1:(n - 1), 1:(n - 1)]= (Aij - AiN) / dx
    Bx[0,:]=Bx[-2,:];Bx[-1,:]=Bx[1,:]
    Bx[:,0]=Bx[:,-2];Bx[:,-1]=Bx[:,1]
    By[0,:]=By[-2,:];By[-1,:]=A[1,:]
    By[:,0]=By[:,-2];By[:,-1]=A[:,1]
    return Bx,By

#mathematical ops

h=500000
kkk=-1
lll=0
for i in range(h):
    Uij = u[1:(n - 1), 1:(n - 1)]
    Vij = v[1:(n - 1), 1:(n - 1)]
    UjP = u[2:, 1:(n - 1)]
    UjN = u[:(n - 2), 1:(n - 1)]
    UiP = u[1:(n - 1), 2:]
    UiN = u[1:(n - 1), :(n - 2)]
    VjP = v[2:, 1:(n - 1)]
    VjN = v[:(n - 2), 1:(n - 1)]
    ViP = v[1:(n - 1), 2:]
    ViN = v[1:(n - 1), :(n - 2)]
    A=Acalc(A, dt, dx, dy, Uij, Vij, eta2)
    Bx,By= Bcalc(A, dx, dy)
    npress = 50
    b=bcalc(b,rho,dt,dx,dy,UiP,UiN,VjP,VjN,UjP,UjN,ViP,ViN)
    p=pcalc(p,dx,dy,b,npress)
    rand1=0
    rand2=0
    if kkk== 1:
        rand1=(np.random.random_sample([n-2,n-2])-0.5)*0.7
        rand2=(np.random.random_sample([n-2,n-2])-0.5)*0.7

    
    u,v= Vcalc(u, v, rho, dt, dx, dy, UiP, UiN, VjP, VjN, UjP, UjN, ViP, ViN, p,Bx,By,rand1,rand2)
    print i

    
    

    kkk = kkk + 1
##    if kkk == 10:
##        # z = ((u ** 2) + (v ** 2)) ** 0.5
##        # z = z[2:-2, 2:-2]
##        # z=p
##        Bx = np.flipud(Bx)
##        By=np.flipud(By)
##        z=(Bx**2)+(By**2)
##        z = z[2:-2, 2:-2]
##
##        # plt.figure(1)
##        plt.figure(figsize=(32, 24), dpi=100)
##        coeffs = pywt.dwt2(z, 'haar')
##        cA, (cH, cV, cD) = coeffs
##
##        plt.subplot(221)
##        plt.title("wavelet t= %s"%i)
##        zz = 1 * (cV * 50) ** 2 + 1 * (cH * 50) ** 2 + 1 * (cD * 350) ** 2
##        I = plt.imshow(zz, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], cmap='jet',
##                       norm=colors.Normalize(vmin=zz.min(), vmax=zz.max()))
##        # plt.colorbar(I, extend='max')
##
##        plt.subplot(222)
##        plt.title("original B t= %s"%i)
##        II = plt.imshow(z, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], cmap='jet',
##                        norm=colors.Normalize(vmin=z.min(), vmax=z.max()))
##        # plt.colorbar(II, extend='max')
##

    if kkk == 40 :
        kkk = 0
        lll=lll+1
        # z = ((u ** 2) + (v ** 2)) ** 0.5
        # z = z[2:-2, 2:-2]
##        Bx = np.flipud(Bx)
##        By=np.flipud(By)
        z=(Bx**2)+(By**2)
        z = z[2:-2, 2:-2]
##        coeffs = pywt.dwt2(z, 'haar')
##        cA, (cH, cV, cD) = coeffs

##        plt.subplot()
##        plt.title('wavelet t= %s' % i)
##        zz = 1 * (cV * 50) ** 2 + 1 * (cH * 50) ** 2 + 1 * (cD * 350) ** 2
##        I = plt.imshow(zz, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], cmap='jet',
##                   norm=colors.Normalize(vmin=zz.min(), vmax=zz.max()))
        # plt.colorbar(I, extend='max')
        
        Bxxx= np.flipud(Bx)
        Byyy=np.flipud(By)
        bind=np.linspace(0,(n-1),(n-1),dtype=np.int16)
        bind=bind[0::16]
        bind1,bind=np.meshgrid(bind,bind)
        buu=Bxxx[bind1,bind]
        bvv=Byyy[bind1,bind]
        bxx=x[bind1,bind]
        byy=y[bind1,bind]
        plt.figure(figsize=(16, 12), dpi=100)
        plt.plot()       
        plt.quiver(bxx,byy,buu,bvv,color='b')       
        plt.title("Magnetic Field B t= %s" % i)
        II = plt.imshow(z, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], cmap='jet',
                    norm=colors.Normalize(vmin=z.min(), vmax=z.max()))
        plt.colorbar(II, extend='max')
        plt.savefig('Picture/B %s .png' % lll)
        plt.clf()
        plt.close()




        plt.figure(figsize=(16, 12), dpi=100)
        z=p
        z = z[2:-2, 2:-2]
        plt.plot()
        plt.title("Pressure P t= %s" % i)
        II = plt.imshow(z, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], cmap='jet',
                    norm=colors.Normalize(vmin=z.min(), vmax=z.max()))
        plt.colorbar(II, extend='max')
        plt.savefig('Picture/P %s .png' % lll)
        plt.clf()
        plt.close()




        plt.figure(figsize=(16, 12), dpi=100)
        z = ((u ** 2) + (v ** 2)) ** 0.5
        z = z[2:-2, 2:-2]
        plt.plot()
        uu = np.flipud(u)
        vv=np.flipud(v)
        ind=np.linspace(0,(n-1),(n-1),dtype=np.int16)
        ind=ind[0::16]
        ind1,ind=np.meshgrid(ind,ind)
        uu=uu[ind1,ind]
        vv=vv[ind1,ind]
        xx=x[ind1,ind]
        yy=y[ind1,ind]      
        plt.quiver(xx,yy,uu,vv,color='b')
        plt.title("Velocity V t= %s" % i)
        II = plt.imshow(z, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], cmap='jet',
                    norm=colors.Normalize(vmin=z.min(), vmax=z.max()))
        plt.colorbar(II, extend='max')
        plt.savefig('Picture/V %s .png' % lll)
        plt.clf()
        plt.close()
        
        
        

        plt.figure(figsize=(16, 12), dpi=100)
        z=A
        z = z[2:-2, 2:-2]
        plt.plot()
        plt.title("Potential A t= %s" % i)
        II = plt.imshow(z, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], cmap='jet',
                    norm=colors.Normalize(vmin=z.min(), vmax=z.max()))
        plt.colorbar(II, extend='max')
        plt.savefig('Picture/A %s .png' % lll)
        plt.clf()
        plt.close()
        





# z=((u**2)+(v**2))**0.5
# z=z[2:-2,2:-2]
# z=A
# Bx = np.flipud(Bx)
# By=np.flipud(By)
# z=(Bx**2)+(By**2)

# coeffs = pywt.dwt2(z, 'haar')
# cA, (cH, cV, cD) = coeffs

# plt.figure(1)
# plt.subplot(223)
# plt.title('vertical+horizontal+diagonal 2 t= %s'%h)
# zz=1*(cV*50)**2+1*(cH*50)**2+1*(cD*350)**2
# I=plt.imshow(zz,extent=[np.min(x),np.max(x),np.min(y),np.max(y)],cmap='jet',norm=colors.Normalize(vmin=zz.min(),vmax=zz.max()))
# plt.colorbar(I,extend='max')
#
# plt.subplot(224)
# plt.title("original 2 t= %s"%h)
# II=plt.imshow(z,extent=[np.max(x),(np.max(x))*2,np.min(y),np.max(y)],cmap='jet',norm=colors.Normalize(vmin=z.min(),vmax=z.max()))
# plt.colorbar(II,extend='max')

# plt.show()

# plt.savefig('V %s .png'%h)

# print("--- %s seconds ---" % (time.time() - start_time))
# plt.quiver(x,y,u,v,color='b')
# #plt.quiver(x,y,Bx,By,color='b')
# u=np.flipud(u)
# v=np.flipud(v)
# z=((u**2)+(v**2))**0.5
# Bx=np.flipud(Bx)
# By=np.flipud(By)
# B=(Bx**2)+(By**2)
# I=plt.imshow(z,extent=[np.min(x),np.max(x),np.min(y),np.max(y)],cmap='jet',norm=colors.Normalize(vmin=z.min(),vmax=z.max()))
#
# plt.colorbar(I,extend='max')
# plt.subplots_adjust(left=0.1, right=0.9, top=0.97, bottom=0.05)
# plt.title(h)
# plt.show()
