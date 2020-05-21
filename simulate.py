#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 19:02:10 2020

@author: SimonL
"""
import numpy as np
import matplotlib.pyplot as plt 
import lfdfiles as lfd
import matplotlib.cbook as cbook
from scipy.optimize import minimize 
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy  import pi
   
    
import numpy as np


def phasorit(tau,omega,harmonics):
    a=np.zeros((2,harmonics,tau.size))
#    for j in range(0,tau.size)
    for i in range(1,harmonics+1):
        a[0,i-1,:]=1/(1+((i*omega)**2)*(tau**2))
        a[1,i-1,:]=(i*omega*tau)/(1+(((i*omega)**2)*(tau**2)));
    return a
#
#def funcFs(f,species,signal):
#    chisq=sum((signal-(f*species))**2) #+f[1]*species2+f[2]*species3+f[3]*species4))**2 )
#    return chisq
#t=np.linspace(0,10e-9,50e-9);

def makeandplotphasor(ax):
    i=0
    sz=3000
    tau= np.linspace(1e-10,5e-7,sz)
    species=np.zeros((2,sz))
    for t in tau:
        species[:,i]=np.squeeze(phasorit(t,omega,1))
        i+=1
    ax.plot(species[0,:],species[1,:],'k-',)
def funcFs(x):
            chisq=np.sum(np.sum((signal-np.sum(x*species,2))**2))
            
            
            
            #+f[1]*species2+f[2]*species3+f[3]*species4))**2 )
    #                chisq=weightLife*(np.sum(np.sum((signal-np.sum(f*species,2))**2)))\
    #        +weightSpec*(np.sum(np.sum((specsignal-np.sum(fspec*specspecies,2))**2)))
            return chisq
    
def findFractions(signal,species): 
    #signal is in the form [(s,g),harmonics] ie shape = (2,har,256,256)
    #species is in form [(s,g),Pure components,harmonics] ie shape = (2,nComponents,harmonics)
    optimize=False
    for a in (signal,species):
        m, n = a.shape[-2:]
        if m != n:
            optimize=True# this is wrong
    if optimize:
        sz=np.shape(species)[2]
        bnds = [(0,1) for x in range(sz)]
        init= [0.1 for x in range(sz)]
        #init=[0.08,0.12,0.3,0.15,0.22,0.13]
        cons = ({'type': 'eq', 'fun': lambda x:  1-np.sum(x)})
        res = minimize(funcFs,init , method='SLSQP',bounds=bnds,constraints = cons,tol=1e-10)
        res=res.x  
    elif not optimize:
        har=signal.shape[1]
        signal = signal.flatten('F')
        comps=species.shape[2]
        species=species.reshape((2*har),comps,order = 'F')
        res=np.linalg.solve(species,signal)
        res=res
    return res

def findFractionsNOCHECK(signal,species): 
    #signal is in the form [(s,g),harmonics] ie shape = (2,har,256,256)
    #species is in form [(s,g),Pure components,harmonics] ie shape = (2,nComponents,harmonics)

        sz=np.shape(species)[2]
        bnds = [(0,1) for x in range(sz)]
        init= [0.1 for x in range(sz)]
        #init=[0.08,0.12,0.3,0.15,0.22,0.13]
        cons = ({'type': 'eq', 'fun': lambda x:  1-np.sum(x)})
        res = minimize(funcFs,init , method='SLSQP',bounds=bnds,constraints = cons,tol=1e-10)
        res=res.x  
        return res

#%%

#tau[2]=6.50e-9;
#tau[3]=10.0e-9;
#def findFractionsLinalg(signal,species): 
##signal is in the form [(s,g),harmonics] ie shape = (2,har)
##species is in form [(s,g),harmonics,Pure components] ie shape = (2,harmonics,nComponents)
#
##    np.linalg.lstsq(species,signal)
#    #init=[0.08,0.12,0.3,0.15,0.22,0.13]
##    cons = ({'type': 'eq', 'fun': lambda x:  1-np.sum(x)})
##    def funcFs(x):
##        chisq=np.sum(np.sum((signal-np.sum(x*species,2))**2)) #+f[1]*species2+f[2]*species3+f[3]*species4))**2 )
##    #                chisq=weightLife*(np.sum(np.sum((signal-np.sum(f*species,2))**2)))\
##    #        +weightSpec*(np.sum(np.sum((specsignal-np.sum(fspec*specspecies,2))**2)))
##        return chisq
##    res = minimize(funcFs,init , method='SLSQP',bounds=bnds,constraints = cons,tol=1e-10)
#    return res.x





def testAlgo(f,taus,har):
    species=phasorit(taus,omega,har)
    for i in range(0,f.size):
        plt.plot(species[0,0,i],species[1,0,i],'bd')#,label='s%i'%(i))
    signal = np.sum(f*species,2)
    
    plt.plot(signal[0,0],signal[1,0],'y2',label='Sig:f1=%1.3f,f2=%1.3f,f3=%1.3f,f4=%1.3f'%(f[0],f[1],f[2],f[3]))
    plt.title('%i Harmonics'%(har))

    res=findFractions(signal,species)
    #res = minimize(funcFs,init , method='Nelder-Mead')
    print('Predicted fractional contribution:')
    print(res)

    comboPredict = np.sum(res*species,2)
    plt.plot(comboPredict[0,0],comboPredict[1,0],'r 1',\
             label='Calc.:f1=%1.3f,f2=%1.3f,f3=%1.3f,f4=%1.3f'%(res[0],res[1],res[2],res[3]))
    return res,comboPredict

har = 2
omega=2*pi*80e6;

f=np.zeros(4)
f[0]=0.15;
f[1]=0.1;
f[2]=0.4;
f[3]=0.35;
#f[4]=0.25;
#f[5]=0.1;

tau=np.zeros(4)
tau[0]=0.1e-9;
tau[1]=0.75e-9;
tau[2]=6.50e-9;
tau[3]=10.0e-9;
#tau[4]=2.0e-9;
#tau[5]=3.40e-9;
species=phasorit(tau,omega,har)
for i in range(0,f.size):
    plt.plot(species[0,0,i],species[1,0,i],'bd')#,label='s%i'%(i))
signal = np.sum(f*species,2)
#

def findspecies(signal):
#    two species:
    f=np.zeros(2)
    f[0]=0.15;
    f[1]=0.45;
#    f[2]=0.4;
#    f[3]=0.35;
    tau=np.zeros(2)
    tau[0]=0.1e-9;
    tau[1]=0.75e-9;
#    tau[2]=6.50e-9;
#    tau[3]=10.0e-9;
    species=phasorit(tau,omega,1)
    signal = np.sum(f*species,2)
    res = findFractions(signal,species)
            
    

#def test():
#    res = minimize(funcFs,init , method='SLSQP',bounds=bnds,constraints = cons,tol=1e-1)
def test():
    findFractions(signal,species)



for har in range(1,5):
    fig, ax = plt.subplots()
    res,comboPredict=testAlgo(f,tau,har)
    makeandplotphasor(ax)
    plt.legend()
    plt.show()
    plt.figure(2)
    plt.plot(har,f[0]-res[0],'*',label='f1-predicted')
    plt.plot(har,f[1]-res[1],'o',label='f2-predicted')
    plt.plot(har,f[2]-res[2],'+',label='f3-predicted')
    plt.plot(har,f[3]-res[3],'x',label='f4-predicted')
#    plt.plot(har,f[4]-res.x[4],'x',label='f5-predicted')
#    plt.plot(har,f[5]-res.x[5],'x',label='f6-predicted')
    plt.show()
#plt.figure(2)
#plt.legend()

#Images
def genImage(tau,sgnoise,sz,har,binning):
    fs=[]
    a=[]
    im=np.zeros((256,256,3,har));
    im[:,:,0,:]=100*np.random.poisson(100,(256,256,1));
    species=phasorit(tau,omega,har);
#    fig, ax = plt.subplots()
    f=np.zeros(4)
    f[0]=0.15;
    f[1]=0.1;
    f[2]=0.4;
    f[3]=0.35;
    #f[4]=0.25;
    #f[5]=0.1;    
    group=np.sum(f*species,2)
    fs.append(f)
#    ax.plot(group[0,0],group[1,0],'k 1')
    im[:128,:128,1:,:]=group+np.random.normal(0.0,sgnoise,(128,128,2,har));
    
    f=np.zeros(4)
    f[0]=0.3;
    f[1]=0.05;
    f[2]=0.2;
    f[3]=0.55;
    group=np.sum(f*species,2)
    fs.append(f)
#    ax.plot(group[0,0],group[1,0],'k 1')
    im[128:,:128,1:,:]=group+np.random.normal(0,sgnoise,(128,128,2,har));
    
    f=np.zeros(4)
    f[0]=0.4;
    f[1]=0.2;
    f[2]=0.2;
    f[3]=0.2;
    group=np.sum(f*species,2)
    fs.append(f)
#    ax.plot(group[0,0],group[1,0],'k1')
    im[:128,128:,1:,:]=group+np.random.normal(0,sgnoise,(128,128,2,har));

    f=np.zeros(4)
    f[0]=0.1;
    f[1]=0.6;
    f[2]=0.2;
    f[3]=0.1;
    group=np.sum(f*species,2)
    fs.append(f)
#    ax.plot(group[0,0],group[1,0],'k1')
    im[128:,128:,1:,:]=group+np.random.normal(0,sgnoise,(128,128,2,har));
    xedges = np.linspace(0,1,binning)
    yedges = np.linspace(0,1,binning)
#    his=np.histogram2d(np.reshape(im[:,:,2,0],256*256),np.reshape(im[:,:,1,0],256*256),bins=(256,256))
#    plt.imshow(his[0],
##               extent=[0, 1, 0, 1],
#           cmap='jet')
    for i in range(0,har):
        H, xedges, yedges = np.histogram2d(np.reshape(im[:,:,2,0],256*256),np.reshape(im[:,:,1,0],256*256), bins=(xedges, yedges))
        H = H.T
        a.append([H,xedges, yedges])
    return im,a,fs

#%%
for sgnoise in np.arange(0,0.1,0.01):
    isize=256
    harmonics=3
    har=harmonics
    binning=256
#    sgnoise=.01
    tau=np.zeros(4)
    tau[0]=0.1e-9;
    tau[1]=0.75e-9;
    tau[2]=6.50e-9;
    tau[3]=10.0e-9;
    species=phasorit(tau,omega,har);
    #fig, ax = plt.subplots()
    
    im,a,fs=genImage(tau,sgnoise,isize,harmonics,binning)
    xedges = np.linspace(0,1,binning)
    yedges = np.linspace(0,1,binning)
    
    
    fig, ax = plt.subplots()
    
    #H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    
    ax.imshow(a[0][0].T, interpolation='nearest', origin='low',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    
    
    
    #plt.hist2d(np.reshape(im[:,:,1,0],256*256),np.reshape(im[:,:,2,0],256*256),bins=(xedges,yedges),cmap=plt.cm.jet);
    makeandplotphasor(ax)
    plt.show()
    
    
    #im=np.zeros((256,256,3,har)); (isize,isize,[intensity,s,g],har)
    
    saveres=[]
    for i in np.arange(0, 250,10):
        for j in np.arange(0,250,10):
            res=[]
            signal=np.zeros((2,har))
            signal[0,:]=im[i,j,1,:]
            signal[1,:]=im[i,j,2,:]
            res=findFractions(signal,species)  
            #print(signal[0,0],signal[1,0],)     
            saveres.append([res,i,j])     
            
    #res,comboPredict=findFractions(signal,species)
    #plt.figure()
    error=[]
    for i in saveres:
        plt.figure(1)
        comboPredict = np.sum(i[0]*species,2)    
        plt.plot(comboPredict[0,0],comboPredict[1,0],'.') 
        if i[1]<128 and i[2]<128:
            error.append(i[0]-fs[0])
        if i[1]>128 and i[2]<128:
            error.append(i[0]-fs[1])
        if i[1]<128 and i[2]>128:
            error.append(i[0]-fs[2])
        if i[1]>128 and i[2]>128:
            error.append(i[0]-fs[3])
            
    for f in fs:
       group=np.sum(f*species,2)
       ax.plot(group[0,0],group[1,0],'k2', label='Signal')
    #ax.show()
       
    error = np.reshape(error,np.size(error))
    sigma=np.std(error)
    print('making histo')  
    fig, ax = plt.subplots()
    
    #    ax.figure(6)    
    ax.hist(error,100)
    ax.set_title('sigma = %f'%(sigma))    
    
    plt.show()
    




#plt.figure(4)

#%%
threshold=20
saveres=[] # in the form of [fractional contributions,  image location xy] => [0..n, x,y]
d=0
signal=np.zeros((2,har));
for h in a: #0 to number of harmonics
    for j in range(h[1].size-1):  #0 to bin size ie phasor size
        for k in range(h[2].size-1):  # bin size ie phasor size
            if h[0][j,k]>threshold:
                res=[]
                signal[0,d]=h[1][j]
                signal[1,d]=h[2][k]
#                print(signal[:,0])
                res=findFractions(signal,species)
                saveres.append([res,j,k])
    d+=1
#    print(res.x)

for f in fs:
    group=np.sum(f*species,2)
    plt.plot(group[0,0],group[1,0],'r2')
#    print(group[:,0])

for i in saveres:
    comboPredict = np.sum(i[0]*species,2)    
    plt.plot(comboPredict[1,0],comboPredict[0,0],'g.')    

#%% baby t
import numpy as np
import lfdfiles as lfd
import matplotlib.pyplot as plt

def phasorit(tau,omega,harmonics):
    a=np.zeros((2,harmonics,tau.size))
#    for j in range(0,tau.size)
    for i in range(1,harmonics+1):
        a[0,i-1,:]=1/(1+((i*omega)**2)*(tau**2))
        a[1,i-1,:]=(i*omega*tau)/(1+(((i*omega)**2)*(tau**2)));
    return a
#

def makeandplotphasor(ax):
    i=0
    sz=3000
    tau= np.linspace(1e-10,5e-7,sz)
    species=np.zeros((2,sz))
    for t in tau:
        species[:,i]=np.squeeze(phasorit(t,omega,1))
        i+=1
    ax.plot(species[0,:],species[1,:],'k-',)

omega=2*pi*80e6;
binning=100
xedges = np.linspace(0,1,binning)
yedges = np.linspace(0,1,binning)
fig, ax = plt.subplots()

myfile='031320_PA14chunk_nocoverslip_fov55_p7_780nm_20xA_CH1_CH2000$CC0S_ch1_h1_h2.R64';
f=lfd.SimfcsR64(myfile)
with lfd.SimfcsR64(myfile) as f:
    x=np.reshape(f.asarray()[2,:,:]*np.sin(np.deg2rad(f.asarray()[1,:,:])),256*256);
    y=np.reshape(f.asarray()[2,:,:]*np.cos(np.deg2rad(f.asarray()[1,:,:])),256*256);
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    H = H.T
    ax.imshow(H, interpolation='nearest', origin='low',
               extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    
    
#    
makeandplotphasor(ax)

plt.show()



#subplot_kw = dict(xlim=(0, 1), ylim=(0, 1), autoscale_on=False)
#fig, ax = plt.subplots(subplot_kw=subplot_kw)
#xedges = np.linspace(0,1,256)
#yedges = np.linspace(0,1,256)
#a = ax.hist2d(np.reshape(im[:,:,1,0],256*256), np.reshape(im[:,:,2,0],256*256),bins=(xedges,yedges),cmap=plt.cm.jet);
#selector = SelectFromCollection(ax,a[3])
#
#def accept(event):
#    if event.key == "enter":
#        print("Selected points:")
#        print(selector.xys[selector.ind])
#        selector.disconnect()
#        ax.set_title("")
#        fig.canvas.draw()
#
#fig.canvas.mpl_connect("key_press_event", accept)
#ax.set_title("Press enter to accept selected points.")
#
#plt.show()
