
import os
import sys
from fluorObjects import *
import matplotlib.pyplot as plt
import numba
from numba import jit
import argparse
import numpy as np


# fluorobjects
# harmonics option for functions and methods


# sys.path.append("/Users/Tara/Desktop/HyperFluo/")

NADH_bound=FluorObject('NADH_bound')
tau=np.zeros(1)
tau[0]=3.2e-9
NADH_bound=phasorit(tau, 2*pi*80e6, 2)

NADH=FluorObject('NADH')
NADH.add_refFile("/Users/Tara/Desktop/FLIM_Spectral/Biofilm_June18_2020/NADH10mM_8p_fov132_20xA_740nmEx_CH1_460bw80_CH2_540bw50-lp495000$CC0S_ch1_h1_h2.R64")

PYO=FluorObject('PYO')
PYO.add_refFile("/Users/Tara/Desktop/FLIM_Spectral/Biofilm_June18_2020/dyes/1mmPYO_1375umTCEP_30p000$CG0T_0-1__ch1_h1_h2.R64")

PVD=FluorObject('PVD')
PVD.add_refFile("/Users/Tara/Desktop/FLIM_Spectral/Biofilm_June18_2020/dyes/PVD500umPBS501$CG0T_0-1__ch1_h1_h2.R64")

myob=FluorObject('myob')
myob.add_refFile("/Users/Tara/Desktop/FLIM_Spectral/Biofilm_June18_2020/WT_ASM_3D_plate1/26p_100d101$CG0T_0-1__ch1_h1_h2.R64")
#%%
LifeSpecies=LifetimeAsPureComponentSG([NADH,PYO,PVD],2)
LifeSpecies=np.concatenate((LifeSpecies,NADH_bound),axis=2)
har=2
LifeSignal=np.zeros((2,har,256,256)) 
LifeSignal=LifetimeAsPureComponentSG([myob],har)
plt.scatter(LifeSignal[1,0,:],LifeSignal[0,0,:],marker='x')

#%%
#poor boo
myob.lifetimethreshold=0


if len(myob.lifetimeIm)>0:
    ans='n'
    myob.showLifeImage(0)
    while ans!='y':
        plt.ion()
        plt.hist(myob.lifetimeIm[0].flatten()*(myob.lifetimeIm[0].flatten()>myob.lifetimethreshold),bins=100,range=[2, np.max(myob.lifetimeIm[0])])
        plt.show()
        myob.lifetimethreshold=np.round(float(input('Where (on x axis) do you want to set threshold')))
        plt.imshow(myob.lifetimeIm[0]*(myob.lifetimeIm[0]>myob.lifetimethreshold))
        plt.show()
        ans=input('Satisfied? answer y or n')

##
#%%


LifeSpecies=LifetimeAsPureComponentSG([NADH,PYO,PVD],2)
LifeSpecies=np.concatenate((LifeSpecies,NADH_bound),axis=2)
# LifeSpecies=np.concatenate((LifeSpecies,NADHPerf),axis=2)

    # LifeSpecies=LifetimeAsPureComponentSG([NADH,PYO,PVD],1) #perfectnadh
# LifeSpecies=np.concatenate((LifeSpecies,NADH_bound),axis=2)#perfectnadh

LifeSignal=np.zeros((2,2,256,256))
LifeSignal[0,0,:,:]=myob.lifetimePhasor[0][0]
LifeSignal[1,0,:,:]=myob.lifetimePhasor[0][1]
LifeTimeResde=np.zeros((4,256,256))
        # SpectralRes=np.zeros((4,256,256))
LifeTimeRes=np.zeros((4,256,256))
# SpecLifeRes=np.zeros((4,256,256))
sz=np.size(LifeSpecies,2)
init= [0.2 for x in range(sz)]
for i in numba.prange(256):
    for j in numba.prange(256):
        # SpectralRes[:,i,j] = findFractions(SpecSignal[:,:,i,j].reshape(2,har,1), SpecSpecies)
        if i>2 and i<253:
            if j>2 and j<253:
                if myob.lifetimethreshold<myob.lifetimeIm[0][i,j]:
                    LifeTimeResde[:,i,j],LifeTimeRes[:,i,j]= findFractions(LifeSignal[:,:,i,j].reshape(2,2,1), LifeSpecies)
                else: LifeTimeResde[:,i,j],LifeTimeRes[:,i,j]=np.nan,np.nan
            else: LifeTimeResde[:,i,j],LifeTimeRes[:,i,j]=np.nan,np.nan
        else: LifeTimeResde[:,i,j],LifeTimeRes[:,i,j]=np.nan,np.nan
        # SpecLifeRes[:,i,j]= findFractionsComplete(SpecSignal[:,:,i,j].reshape(2,har,1),SpecSpecies,LifeSignal[:,:,i,j].reshape(2,har,1),LifeSpecies)
        # init=np.mean([SpectralRes[:,i,j],LifeTimeRes[:,i,j],SpecLifeRes[:,i,j]],0)
    print('{} lines left'.format(255-i))
   #perfect nadh print('LIFETIME: \n Mean fractions in line {}: \n NADH={}, \n PYO={}, \n PVD={}, \n NADH_bound={}'.format(i, np.nanmean(LifeTimeRes[0,i,:]),np.nanmean(LifeTimeRes[1,i,:]),np.nanmean(LifeTimeRes[2,i,:]),np.nanmean(LifeTimeRes[3,i,:])))
    print('LIFETIME: \n Mean fractions in line {}: \n NADH={}, \n PYO={}, \n PVD={}, \n NADH_bound={}'.format(i, np.nanmean(LifeTimeRes[0,i,:]),np.nanmean(LifeTimeRes[1,i,:]),np.nanmean(LifeTimeRes[2,i,:]),np.nanmean(LifeTimeRes[3,i,:])))
 
#########################

#%%


liferes= imDecomposition_life(NADH,PYO,PVD,NADH_bound,myob)





LifeSignal=np.zeros((2,1,256,256))
# lifetime phasor =[s1,g1,s2,g2] 
# 2 arrays

# populate s1
LifeSignal[0,0,:,:]=myob.lifetimePhasor[0][0]

# populate g1
LifeSignal[1,0,:,:]=myob.lifetimePhasor[0][1]




LifeSignal=np.zeros((2,1,256,256))
LifeSignal[0,0,:,:]=myob.lifetimePhasor[0][0]
lifesignal=LifetimeAsPureComponentSG([NADH,PYO,PVD,myob],2)




for i in numba.prange(256):
    for j in numba.prange(256):
        LifeTimeResde[:,i,j],LifeTimeRes[:,i,j]=findFractions(LifeSignal[:,:,i,j].reshape(2,1,1), LifeSpecies)
