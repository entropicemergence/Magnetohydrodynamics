import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from matplotlib import style
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pywt
import skimage.measure


def Vcalc(u,v,rho,dt,dx,dy,UiP,UiN,VjP,VjN,UjP,UjN,ViP,ViN,p,Bx,By,rand1,rand2,Uij,Vij):
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

    u[1:(n - 1), 1:(n - 1)]= Uij - (dt * Uij * aux) - (dt * Vij * auy) + (eta * ((dt * Dux) + (dt * Duy)))-Pi+Bupart
    v[1:(n - 1), 1:(n - 1)] = Vij - (dt * Uij * avx) - (dt * Vij * avy) + (eta * ((dt * Dvx) + (dt * Dvy)))-Pj+Bvpart

    return u,v

# Density

def pbound(p):

    p=np.lib.pad(p,((1,1),(1,1)),'constant',constant_values=((1,1),(1,1)))

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
    for k in range(npress):                             #relaxing pressure

        PjP = p[2:, 1:(n - 1)]
        PjN = p[:(n - 2), 1:(n - 1)]
        PiP = p[1:(n - 1), 2:]
        PiN = p[1:(n - 1), :(n - 2)]
        p[1:(n - 1), 1:(n - 1)]=((((PiP+PiN)*(dy**2))+((PjP+PjN)*(dx**2)))/divider)-((((dx**2)*(dy**2))/divider)*b[1:(n - 1),1:(n - 1)])

    return p
# Vector Potential

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


    return A

#magnetic field
def Bcalc(A,dx,dy,Bx,By):
    Aij = A[1:(n - 1), 1:(n - 1)]
    AjN = A[:(n - 2), 1:(n - 1)]
    AiN = A[1:(n - 1), :(n - 2)]

    Bx[1:(n - 1), 1:(n - 1)]=(Aij-AjN)/dy
    By[1:(n - 1), 1:(n - 1)]= (Aij - AiN) / dx

    return Bx,By

#mathematical ops

def solve(u,v,A,p,Bx,By,b,dx,dy,ggg):
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
    Bx,By= Bcalc(A, dx, dy,Bx,By)
    npress = 50
    if ggg==0:
        npress = 1000
    b=bcalc(b,rho,dt,dx,dy,UiP,UiN,VjP,VjN,UjP,UjN,ViP,ViN)
    p=pcalc(p,dx,dy,b,npress)
    rand1=0
    rand2=0
    u,v= Vcalc(u,v,rho,dt,dx,dy,UiP,UiN,VjP,VjN,UjP,UjN,ViP,ViN,p,Bx,By,rand1,rand2,Uij,Vij)
    return u,v,A,p,Bx,By,b
    
  
    
def scaleup(value):
    new_value=value.repeat(2,axis=0).repeat(2,axis=1)
    return new_value
def max_pool(value):
    new_value=skimage.measure.block_reduce(value, (2,2), np.max)    
    return new_value
def scaledown(value1,value2,value3,value4):
    value1=max_pool(value1)
    value2=max_pool(value2)
    value3=max_pool(value3)
    value4=max_pool(value4)
    new_value=np.zeros([10,10])
    new_value[indy1,indx1]=value1
    new_value[indy2,indx2]=value2
    new_value[indy3,indx3]=value3
    new_value[indy4,indx4]=value4
    return new_value

def refine(current,unit_location):
    new_value=current[:unit_location]
    unit=current[unit_location]
    unit=unit[1:-1,1:-1]    
    a=scaleup(unit[indy1,indx1])
    b=scaleup(unit[indy2,indx2])
    c=scaleup(unit[indy3,indx3])
    d=scaleup(unit[indy4,indx4])
    a=np.lib.pad(a,((1,1),(1,1)),'constant',constant_values=((1,1),(1,1)))
    b=np.lib.pad(b,((1,1),(1,1)),'constant',constant_values=((1,1),(1,1)))
    c=np.lib.pad(c,((1,1),(1,1)),'constant',constant_values=((1,1),(1,1)))
    d=np.lib.pad(d,((1,1),(1,1)),'constant',constant_values=((1,1),(1,1)))
    new_value=np.append(new_value,[a],axis=0)
    new_value=np.append(new_value,[b],axis=0)
    new_value=np.append(new_value,[c],axis=0)
    new_value=np.append(new_value,[d],axis=0)
    new_value=np.append(new_value,current[(unit_location+1):],axis=0)
    return new_value   

n=128
inda=np.arange(n/2)
indb=np.arange(n/2,n)
indx1,indy1=np.meshgrid(inda,inda)
indx2,indy2=np.meshgrid(indb,inda)
indx4,indy4=np.meshgrid(inda,indb)
indx3,indy3=np.meshgrid(indb,indb)
x = np.linspace(0.0, 8.0, n)
y = np.linspace(0.0, 8.0, n)
x,y=np.meshgrid(x,y)
t=np.array([[0,0,4,4],[4,0,8,4],[4,4,8,8,],[0,4,4,8]],dtype=np.float)

for i in range(4):
    s1=t[i,0]
    s2=t[i,1]
    d1=t[i,2]
    d2=t[i,3]
    xx = np.linspace(s1, d1, n)
    yy = np.linspace(s2, d2, n)
    xx,yy=np.meshgrid(xx,yy)
    if i==0:
        x=np.append([x],[xx],axis=0)
        y=np.append([y],[yy],axis=0)
    if i > 0:
        x=np.append(x,[xx],axis=0)
        y=np.append(y,[yy],axis=0)

def addsec(existing,new):
    new_existing=np.append(existing,[new],axis=0)
    return new_existing


def padding(vari1,vari2,vari3,vari4,vari5,vari6,vari7):
    vari1=np.lib.pad(vari1,((1,1),(1,1)),'constant',constant_values=((1,1),(1,1)))
    vari2=np.lib.pad(vari2,((1,1),(1,1)),'constant',constant_values=((1,1),(1,1)))
    vari3=np.lib.pad(vari3,((1,1),(1,1)),'constant',constant_values=((1,1),(1,1)))
    vari4=np.lib.pad(vari4,((1,1),(1,1)),'constant',constant_values=((1,1),(1,1)))
    vari5=np.lib.pad(vari5,((1,1),(1,1)),'constant',constant_values=((1,1),(1,1)))
    vari6=np.lib.pad(vari6,((1,1),(1,1)),'constant',constant_values=((1,1),(1,1)))
    vari7=np.lib.pad(vari7,((1,1),(1,1)),'constant',constant_values=((1,1),(1,1)))
    return vari1,vari2,vari3,vari4,vari5,vari6,vari7

def flowvar(var1,var2,var3,var4,var5):
    var1[0,:]=var5[-2,:];var1[-1,:]=var3[2,:]
    var1[:,0]=var2[:,-2];var1[:,-1]=var4[:,2]
    return var1



pi=np.pi
dx=8.0/(n-1)
dy=8.0/(n-1)
dx2=dx/2
dy2=dx2
dx3=dx/4
dx4=dx/8
dx5=dx/16
dt2=0.0002/2
dt3=0.0002/4
dt4=0.0002/8
dt5=0.0002/16


xm=x[0]
ym=y[0]
dt=0.0002
eta=0.1
eta2=0.1
rho=25/(36*pi)
u=-np.sin(ym*pi/2)
v=np.sin(xm*pi/2)
A=np.cos(ym*1*pi/2)+0.5*np.cos(xm*2*pi/2)
Bx=A*0
By=A*0
p=xm*0
b=ym*0

A1=scaleup(A[indy1,indx1])
Bx1=scaleup(Bx[indy1,indx1])
By1=scaleup(By[indy1,indx1])
p1=scaleup(p[indy1,indx1])
b1=scaleup(b[indy1,indx1])
u1=scaleup(u[indy1,indx1])
v1=scaleup(v[indy1,indx1])

A2=scaleup(A[indy2,indx2])
Bx2=scaleup(Bx[indy2,indx2])
By2=scaleup(By[indy2,indx2])
p2=scaleup(p[indy2,indx2])
b2=scaleup(b[indy2,indx2])
u2=scaleup(u[indy2,indx2])
v2=scaleup(v[indy2,indx2])

A3=scaleup(A[indy3,indx3])
Bx3=scaleup(Bx[indy3,indx3])
By3=scaleup(By[indy3,indx3])
p3=scaleup(p[indy3,indx3])
b3=scaleup(b[indy3,indx3])
u3=scaleup(u[indy3,indx3])
v3=scaleup(v[indy3,indx3])

A4=scaleup(A[indy4,indx4])
Bx4=scaleup(Bx[indy4,indx4])
By4=scaleup(By[indy4,indx4])
p4=scaleup(p[indy4,indx4])
b4=scaleup(b[indy4,indx4])
u4=scaleup(u[indy4,indx4])
v4=scaleup(v[indy4,indx4])

u1,v1,A1,p1,Bx1,By1,b1=padding(u1,v1,A1,p1,Bx1,By1,b1)
u2,v2,A2,p2,Bx2,By2,b2=padding(u2,v2,A2,p2,Bx2,By2,b2)
u3,v3,A3,p3,Bx3,By3,b3=padding(u3,v3,A3,p3,Bx3,By3,b3)
u4,v4,A4,p4,Bx4,By4,b4=padding(u4,v4,A4,p4,Bx4,By4,b4)

Amas=np.append([A1],[A2],axis=0)
Bxmas=np.append([Bx1],[Bx2],axis=0)
Bymas=np.append([By1],[By2],axis=0)
pmas=np.append([p1],[p2],axis=0)
bmas=np.append([b1],[b2],axis=0)
umas=np.append([u1],[u2],axis=0)
vmas=np.append([v1],[v2],axis=0)

Amas=addsec(Amas,A3)
Bxmas=addsec(Bxmas,Bx3)
Bymas=addsec(Bymas,By3)
pmas=addsec(pmas,p3)
bmas=addsec(bmas,b3)
umas=addsec(umas,u3)
vmas=addsec(vmas,v3)

Amas=addsec(Amas,A4)
Bxmas=addsec(Bxmas,Bx4)
Bymas=addsec(Bymas,By4)
pmas=addsec(pmas,p4)
bmas=addsec(bmas,b4)
umas=addsec(umas,u4)
vmas=addsec(vmas,v4)

#target=1
#Amas=refine(Amas,target)
#Bxmas=refine(Bxmas,target)
#Bymas=refine(Bymas,target)
#pmas=refine(pmas,target)
#bmas=refine(bmas,target)
#umas=refine(umas,target)
#vmas=refine(vmas,target)



def calctotal(Am,Bxm,Bym,pm,bm,um,vm,unit_count,hhh):
    for i in range(unit_count):
        a1A=Am[i]
        a1Bx=Bxm[i]
        a1By=Bym[i]
        a1p=pm[i]
        a1b=bm[i]
        a1u=um[i]
        a1v=vm[i]
        a1u,a1v,a1A,a1p,a1Bx,a1By,a1b=solve(a1u,a1v,a1A,a1p,a1Bx,a1By,a1b,dx2,dy2,hhh)
        Am[i]=a1A
        Bxm[i]=a1Bx
        Bym[i]=a1By
        pm[i]=a1p
        bm[i]=a1b
        um[i]=a1u
        vm[i]=a1v
    return Am,Bxm,Bym,pm,bm,um,vm


masL1=np.ones(4)
masL2=np.zeros([4,4])
masL3=np.zeros([4,4,4])
masL4=np.zeros([4,4,4,4])
masL5=np.zeros([4,4,4,4,4])
masindex=np.array([[1,0,0,0,0,0,0],[1,1,1,0,0,0,0],[1,2,2,0,0,0,0],[1,3,3,0,0,0,0]],dtype=np.int16)
indexleft=np.array([[1,1,1,0,0,0,0],[1,1,0,0,0,0,0],[1,1,3,0,0,0,0],[1,1,2,0,0,0,0]],dtype=np.int16)
indextop=np.array([[1,1,3,0,0,0,0],[1,1,2,0,0,0,0],[1,1,1,0,0,0,0],[1,1,0,0,0,0,0]],dtype=np.int16)
indexright=np.array([[1,1,1,0,0,0,0],[1,1,0,0,0,0,0],[1,1,3,0,0,0,0],[1,1,2,0,0,0,0]],dtype=np.int16)
indexbottom=np.array([[1,1,3,0,0,0,0],[1,1,2,0,0,0,0],[1,1,1,0,0,0,0],[1,1,0,0,0,0,0]],dtype=np.int16)
masindex=np.append([masindex],[indexleft],axis=0)
masindex=np.append(masindex,[indextop],axis=0)
masindex=np.append(masindex,[indexright],axis=0)
masindex=np.append(masindex,[indexbottom],axis=0)

#print masindex

def transfervar(var,index):
    
    
    
    
    




nnn=0
n=130        
       #after padding the size increase
for i in range(0):

    count=7

#    Bx1=flowvar(Bx1,Bx2,Bx1,Bx2,Bx1)
#    By1=flowvar(By1,By2,By1,By2,By1)
#    A1=flowvar(A1,A2,A1,A2,A1)
#    p1=flowvar(p1,p2,p1,p2,p2)
#    u1=flowvar(u1,u2,u1,u2,u1)
    
#    v1=flowvar(v1,v2,v1,v2,v1)
#    
#    Bx2=flowvar(Bx2,Bx2,Bx1,Bx2,Bx1)
#    By2=flowvar(By2,By2,By1,By2,By1)
#    A2=flowvar(A2,A1,A1,A2,A1)
#    p2=flowvar(p2,p1,p1,p2,p2)
#    u2=flowvar(u2,u2,u1,u2,u1)
#    v2=flowvar(v2,v2,v1,v2,v1)
#    
#    Bx3=flowvar(Bx3,Bx2,Bx1,Bx2,Bx1)
#    By3=flowvar(By3,By2,By1,By2,By1)
#    A3=flowvar(A3,A1,A1,A2,A1)
#    p3=flowvar(p3,p1,p1,p2,p2)
#    u3=flowvar(u3,u2,u1,u2,u1)
#    v3=flowvar(v3,v2,v1,v2,v1)
#    
#    Bx4=flowvar(Bx4,Bx2,Bx1,Bx2,Bx1)
#    By4=flowvar(By4,By2,By1,By2,By1)
#    A4=flowvar(A4,A1,A1,A2,A1)
#    p4=flowvar(p4,p1,p1,p2,p2)
#    u4=flowvar(u4,u2,u1,u2,u1)
#    v4=flowvar(v4,v2,v1,v2,v1)


    iii=i
    Amas,Bxmas,Bymas,pmas,bmas,umas,vmas=calctotal(Amas,Bxmas,Bymas,pmas,bmas,umas,vmas,count,iii)
#    print Amas.shape
    
    nnn=nnn+1
    
#    if nnn==40:
#        nnn=0
#        z=(Bx1**2)+(By1**2)
#        z2=(Bx2**2)+(By2**2)
#        z3=(Bx3**2)+(By3**2)
#        z4=(Bx4**2)+(By4**2)
#        
#        plt.figure(figsize=(16, 12), dpi=100)
#        plt.plot()       
#        plt.title("Magnetic Field B1 t= %s" % i)
#        II = plt.imshow(z, extent=[np.min(x[1]), np.max(x[1]), np.min(y[1]), np.max(y[1])], cmap='jet',
#                    norm=colors.Normalize(vmin=z.min(), vmax=z.max()))
#        plt.savefig('Picture/8/%s B1.png' % i)
#        plt.clf()
#        plt.close()
#        
#        plt.figure(figsize=(16, 12), dpi=100)
#        plt.plot()       
#        plt.title("Magnetic Field B2 t= %s" % i)
#        II = plt.imshow(z2, extent=[np.min(x[2]), np.max(x[2]), np.min(y[2]), np.max(y[2])], cmap='jet',
#                    norm=colors.Normalize(vmin=z2.min(), vmax=z2.max()))
#        plt.savefig('Picture/8/%s B2.png' % i)
#        plt.clf()
#        plt.close()
#        
#        plt.figure(figsize=(16, 12), dpi=100)
#        plt.plot()       
#        plt.title("Magnetic Field B3 t= %s" % i)
#        II = plt.imshow(z3, extent=[np.min(x[3]), np.max(x[3]), np.min(y[3]), np.max(y[3])], cmap='jet',
#                    norm=colors.Normalize(vmin=z3.min(), vmax=z3.max()))
#        #plt.colorbar(II, extend='max')
#        plt.savefig('Picture/8/%s B3.png' % i)
#        plt.clf()
#        plt.close()
#        
#        
#        plt.figure(figsize=(16, 12), dpi=100)
#        plt.plot()       
#        plt.title("Magnetic Field B4 t= %s" % i)
#        II = plt.imshow(z4, extent=[np.min(x[4]), np.max(x[4]), np.min(y[4]), np.max(y[4])], cmap='jet',
#                    norm=colors.Normalize(vmin=z4.min(), vmax=z4.max()))
#        #plt.colorbar(II, extend='max')
#        plt.savefig('Picture/8/%s B4.png' % i)
#        plt.clf()
#        plt.close()
#
#        print i



