# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:16:41 2020

@author: Simon Leemans
"""
import numpy as np
import os
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt   
import matplotlib
import pandas as pd 
import seaborn as sns
import numpy.fft as ft
from numpy import pi
import scipy.signal as sc
import tkinter.filedialog
import lfdfiles
from scipy.optimize import minimize 
from scipy import optimize
import scipy.stats as stats
import tkinter.messagebox
from scipy.optimize import LinearConstraint
import numba
from numba import jit
# class FluorSignal:
    
def meanAlong(lsAx,MyArray,order):
    #run through lsAx to find axes that you want to squeeze:
    ls=[]
    for i in order[::-1]:
        if i in lsAx:
          ls.append(order.find(i))
    MyArray=MyArray.mean(axis=tuple(ls))
    return MyArray

class FluorObject:
    def __init__(self, name):
        self.name = name
        self.spectralthreshold=[]
        self.spectralIm=[]
        self.spectralRange=[]
        # self.spectralPhasor=[]
        self.SpecfileName=[]
        self.order=[]
        
        self.lifetimethreshold=[]
        self.lifetimeIm=[]
        self.lifetimePhasor=[]
        self.lifetimefileName=[]
        
        print('Please add data using methods .addLSM, .addRefFile')  

    def info(self):
        if len(self.spectralIm)>0:
            print('{} has {} Spectral (lsm) images. \n \n Filename and path = {}'.format(self.name, len(self.spectralIm),self.SpecfileName))
        if len(self.lifetimeIm)>0:
            print('\n Object {}, has {} Lifetime images. \n \n Filename and path = {}'.format(self.name,len(self.lifetimeIm),self.lifetimefileName))
            
    def addLSM(self,ch):
        #this should be a LSM file containing ch spectral channels
        # ch=9
        valerr=False
        LSMfile=tkinter.filedialog.askopenfilename(title='Select a .LSM file',filetypes=[("LSM files", "*.LSM")])
        if os.path.splitext(LSMfile)[1] =='.lsm':
            ti=tiff.TiffFile(LSMfile) # read in first file in directory
            self.SpecfileName.append(LSMfile)
            self.order.append(ti.series[0].axes.lower())            
            # self.meta.append(ti.lsm_metadata())
            try:
                self.spectralRange.append(np.asarray(ti.lsm_metadata['ChannelColors']['ColorNames']).astype(float)[:ch])
            except ValueError:
                valerr=True
                ct=0
                for i in ti.lsm_metadata['ChannelColors']['ColorNames']:
                    try:
                        float(i)
                        ct+=1
                    except: break
                # self.spectralRange.append(np.asarray(ti.lsm_metadata['ChannelColors']['ColorNames'])[:ct].astype(float))           
                self.spectralRange.append(np.asarray(ti.lsm_metadata['ChannelColors']['ColorNames'])[:ch].astype(float)) 
                c=ti.series[0].axes.lower().find('c')
                self.spectralIm.append(np.delete(ti.asarray()[:,:,:ch+1,:,:],-1,axis=c))
            if valerr==False:                     
                self.spectralIm.append(ti.asarray()[:,:,:ch,:,:])

    def addLSMauto(self,LSMfile,ch):
            #this should be a LSM file containing ch spectral channels
        # ch=9
        valerr=False
        # LSMfile=tkinter.filedialog.askopenfilename(title='Select a .LSM file',filetypes=[("LSM files", "*.LSM")])
        if os.path.splitext(LSMfile)[1] =='.lsm':
            ti=tiff.TiffFile(LSMfile) # read in first file in directory
            self.SpecfileName.append(LSMfile)
            self.order.append(ti.series[0].axes.lower())            
            # self.meta.append(ti.lsm_metadata())
            try:
                self.spectralRange.append(np.asarray(ti.lsm_metadata['ChannelColors']['ColorNames']).astype(float)[:ch])
            except ValueError:
                valerr=True
                ct=0
                for i in ti.lsm_metadata['ChannelColors']['ColorNames']:
                    try:
                        float(i)
                        ct+=1
                    except: break
                # self.spectralRange.append(np.asarray(ti.lsm_metadata['ChannelColors']['ColorNames'])[:ct].astype(float))           
                self.spectralRange.append(np.asarray(ti.lsm_metadata['ChannelColors']['ColorNames'])[:ch].astype(float)) 
                c=ti.series[0].axes.lower().find('c')
                self.spectralIm.append(np.delete(ti.asarray()[:,:,:ch+1,:,:],-1,axis=c))
            if valerr==False:                     
                self.spectralIm.append(ti.asarray()[:,:,:ch,:,:])
        
    def asSpectralPhasor(self,har,Fileindex,ch):
        def specphasor(sp,si,co):
            s=np.sum(si*sp,0)
            s=s/(np.sum(sp,0)+np.finfo(float).eps)
            g=np.sum(co*sp,0)
            g=g/(np.sum(sp,0)+np.finfo(float).eps)
            return s,g
        # ch=9
        i=self.spectralIm[Fileindex]
        order=self.order[Fileindex]
        t=np.linspace(0,1,ch)
        si=np.sin(har*2*np.pi*t)
        co=np.cos(har*2*np.pi*t)
        sh=np.shape(self.spectralIm[Fileindex])
        isize=sh[order.find('x')]
        jsize=sh[order.find('y')]
        bigsi=np.tile(si,(isize,jsize))
        bigco=np.tile(co,(isize,jsize))
        bigsi=bigsi.swapaxes(0,1).reshape((ch,isize,jsize),order='F')
        bigco=bigco.swapaxes(0,1).reshape((ch,isize,jsize),order='F')
        lsAx=['t','z']
        meanIm=meanAlong(lsAx,i,order)
        s,g=specphasor(meanIm,bigsi,bigco)
        # H=np.histogram2d(s.flatten(),g.flatten(),bins=256,range=[[-1,1],[-1,1]])
        # plt.pcolormesh(H[1], H[2], H[0],zorder=0)
        # plt.title(os.path.basename(self.SpecfileName[Fileindex]))
        # circ=plt.Circle((00,0),radius=1,color='white',fill=False,zorder=10)
        # plt.gcf().gca().add_artist(circ)
        # plt.show()
        # plt.hist2d(s.flatten(),g.flatten(),range=[[-1, 1],[-1,1]],bins=1000)
        return s,g
    def meanSpectrum(self,fileIndex):
        i=self.spectralIm[fileIndex]
        waves=self.spectralRange[fileIndex]
        order=self.order[fileIndex]
        lsAx=['t','z','x','y']
        meanIm=meanAlong(lsAx,i,order)
        spectrum=meanIm;
        plt.plot(waves,spectrum)
        plt.title(os.path.basename(self.SpecfileName[fileIndex]))
        plt.show()
        return waves,spectrum
    def spectralMeanSG(self,fileIndex,har,ch):
        # returns average s and g coordinates as s1,g1
        s,g=self.asSpectralPhasor(har,fileIndex,ch)
        return np.mean(s),np.mean(g)
    
    def showSpecImage(self,fileIndex):
            i=self.spectralIm[fileIndex]
            order=self.order[fileIndex]
            # if order.find('t')>=0:
            #     meanIm=i.mean(order.find('t'))
            # if order.find('c')>=0:
            #     np.mean=meanIm.mean(order.find('c'));
            # if order.find('z')>=0:
            # meanIm=meanIm.mean(order.find('z'));  
            lsAx=['t','c','z']
            meanIm=meanAlong(lsAx,i,order)
            plt.imshow(meanIm)
            plt.title(os.path.basename(self.SpecfileName[fileIndex]))
            plt.show()

    def add_refFile(self):
         # reads in r64, separates into DC (intensity image) and Phasor Components 
        RefFile=tkinter.filedialog.askopenfilename(title='Select a .r64 file',filetypes=[("R64 files", "*.r64")])
        if os.path.splitext(RefFile)[1].lower() =='.r64': 
            self.lifetimefileName.append(RefFile)
            with lfdfiles.SimfcsR64(RefFile) as f:
                self.lifetimeIm.append(f.asarray()[0,:,:])
                s1=f.asarray()[2,:,:]*np.sin(np.radians(f.asarray()[1,:,:]))
                g1=f.asarray()[2,:,:]*np.cos(np.radians(f.asarray()[1,:,:]))
                s2=f.asarray()[4,:,:]*np.sin(np.radians(f.asarray()[3,:,:]))
                g2=f.asarray()[4,:,:]*np.cos(np.radians(f.asarray()[3,:,:]))
                self.lifetimePhasor.append([s1,g1,s2,g2])
            
    def add_refFileauto(self,RefFile):
     # reads in r64, separates into DC (intensity image) and Phasor Components 
    # RefFile=tkinter.filedialog.askopenfilename(title='Select a .r64 file',filetypes=[("R64 files", "*.r64")])
        if os.path.splitext(RefFile)[1].lower() =='.r64': 
            self.lifetimefileName.append(RefFile)
            with lfdfiles.SimfcsR64(RefFile) as f:
                self.lifetimeIm.append(f.asarray()[0,:,:])
                s1=f.asarray()[2,:,:]*np.sin(np.radians(f.asarray()[1,:,:]))
                g1=f.asarray()[2,:,:]*np.cos(np.radians(f.asarray()[1,:,:]))
                s2=f.asarray()[4,:,:]*np.sin(np.radians(f.asarray()[3,:,:]))
                g2=f.asarray()[4,:,:]*np.cos(np.radians(f.asarray()[3,:,:]))
                self.lifetimePhasor.append([s1,g1,s2,g2])            
    def showLifeImage(self,fileIndex):
        i=self.lifetimeIm[fileIndex]
        plt.imshow(i)
        plt.title(os.path.basename(self.lifetimefileName[fileIndex]))
        plt.show()
    def showLifePhasor(self,fileIndex,har): # returns histogram
        #har=harmonic 1 or harmonic 2
        phasor=self.lifetimePhasor[fileIndex]
        if (har != 1) and (har!=2):
            #print('incorrect harmonic, default=1')
            har=1
        if har ==1:
            s=phasor[0]
            g=phasor[1]
        if har ==2:
            s=phasor[2]
            g=phasor[3]    
        H=np.histogram2d(s.flatten(),g.flatten(),bins=256,range=[[0,1],[0,1]])
        fig,ax=plt.subplots()
        plt.pcolormesh(H[1], H[2], np.log(H[0]),zorder=0)
        plt.title(os.path.basename(self.lifetimefileName[fileIndex])+'\n LogPhasor')
        circ=plt.Circle((0.5,0),radius=0.5,color='black',fill=False,zorder=10)
        plt.gcf().gca().add_artist(circ)
        plt.show()
        return H
    def lifeSG_Stats(self,fileIndex):
        # returns average s and g coordinates as list [means,variances]
        #self.showLifePhasor(fileIndex,1)
        #self.showLifePhasor(fileIndex,2)
        [s1,g1,s2,g2]=self.lifetimePhasor[fileIndex]
        means=[np.mean(s1),np.mean(g1),np.mean(s2),np.mean(g2)]
        variances=[np.var(s1),np.var(g1),np.var(s2),np.var(g2)]
        return means,variances

def LifetimeAsPureComponentSG(listOfFluorObjects,harmonics):
    species=np.zeros((2,harmonics,len(listOfFluorObjects)))
    i=0
    for fluorophore in listOfFluorObjects:
        average=[]
        for j in range(len(fluorophore.lifetimeIm)):
            measurement=fluorophore.lifeSG_Stats(j)[0]
            average.append(measurement)
        if harmonics==2:    
            trueSG =np.mean(average,0)
        elif harmonics==1:
            trueSG =np.mean(average,0)[:2]
        species[:,:,i]=np.reshape(trueSG,[2,harmonics],'F')
        i+=1
    return species     
def SpectrumAsPureComponentSG(listOfFluorObjects,harmonics,ch):
    species=np.zeros((2,harmonics,len(listOfFluorObjects)))
    i=0
    for fluorophore in listOfFluorObjects:
        for harmonic in range(harmonics):
            average=[]
            for j in range(len(fluorophore.spectralIm)):
                measurement=fluorophore.spectralMeanSG(j,harmonic+1,ch)
                average.append(measurement)
            if len(np.shape(average))>1:
                trueSG =np.mean(average,0)
            else:
                trueSG=average
            species[:,harmonic,i]=trueSG #np.reshape(trueSG,[2,2],'F')
        i+=1
    return species

    

# a=np.zeros((2,harmonics,tau.size))
# for i in range(1,harmonics+1):
#     a[0,i-1,:]=1/(1+((i*omega)**2)*(tau**2))
#     a[1,i-1,:]=(i*omega*tau)/(1+(((i*omega)**2)*(tau**2)));
      
# def findFractions(signal,species): 
#     #signal is in the form [(s,g),harmonics] ie shape = (2,har,256,256)
#     #species is in form [(s,g),Pure components,harmonics] ie shape = (2,nComponents,harmonics)
#     optimize=False
#     for a in (signal,species):
#         m, n = a.shape[-2:]
#         if m != n:
#             optimize=True# this is wrong
#     if optimize:
#         sz=np.shape(species)[2]
#         bnds = [(0,1) for x in range(sz)]
#         init= [0.1 for x in range(sz)]
#         #init=[0.08,0.12,0.3,0.15,0.22,0.13]
#         cons = ({'type': 'eq', 'fun': lambda x:  1-np.sum(x)})
#         res = minimize(funcFs,init , method='SLSQP',bounds=bnds,constraints = cons,tol=1e-10)
#         res=res.x  
#     elif not optimize:
#         har=signal.shape[1]
#         signal = signal.flatten('F')
#         comps=species.shape[2]
#         species=species.reshape((2*har),comps,order = 'F')
#         res=np.linalg.solve(species,signal)
#         res=res
#     return res  
    

def findFractionsMLE(signal,species,var): 
    #signal is in the form [(s,g),harmonics] ie shape = (2,har,256,256)
    #species is in form [(s,g),Pure components,harmonics] ie shape = (2,nComponents,harmonics)
    # def funcFs(x):
    #     chisq=np.sum(np.sum((signal-np.sum(x*species,2))**2))
    #     return chisq
    def funcFs(x):
        #mle regression: We want to find the vector f containing fractional contributions:
        #s = s1*f1+s2*f2+...
        #g= g1*f1+g2*f2+...
        sd=x[-4:] # unpack into vector containing standard deviations (scale)
        sd= np.reshape(sd,(2,2),'F')
        yhat=np.sum(x[:-4]*species,2) # these are the pure species..
        negLL=-np.sum(stats.norm.logpdf(signal, loc=yhat,scale=sd))        
        return negLL
    sz=np.shape(species)[2]
    bnds = [(0,1) for x in range(sz)]
    init= [0.1 for x in range(sz)]
    bnds.extend((v*0.1,v*3) for v in var)
    init.extend(var)
    #init=[0.08,0.12,0.3,0.15,0.22,0.13]
    cons = ({'type': 'eq', 'fun': lambda x:  1-np.sum(x[:-4])})
    res = minimize(funcFs,init , method='SLSQP',bounds=bnds,constraints = cons,tol=1e-10)
    # res = minimize(funcFs,init , method='Nelder-Mead')#,bounds=bnds)#,constraints = cons,tol=1e-10)
    res=res.x  
    return res
def findFractionsMLE1h(signal,species,var): 
    #signal is in the form [(s,g),harmonics] ie shape = (2,har,256,256)
    #species is in form [(s,g),Pure components,harmonics] ie shape = (2,nComponents,harmonics)
    # def funcFs(x):
    #     chisq=np.sum(np.sum((signal-np.sum(x*species,2))**2))
    #     return chisq
    def funcFs(x):
        #mle regression: We want to find the vector f containing fractional contributions:
        #s = s1*f1+s2*f2+...
        #g= g1*f1+g2*f2+...
        sd=x[-4:] # unpack into vector containing standard deviations (scale)
        sd= np.reshape(sd,(2,2),'F')
        yhat=np.sum(x[:-4]*species,2) # these are the pure species..
        negLL=-np.sum(stats.norm.logpdf(signal, loc=yhat,scale=sd))        
        return negLL
    sz=np.shape(species)[2]
    bnds = [(0,1) for x in range(sz)]
    init= [0.2 for x in range(sz)]
    bnds.extend((v*0.1,v*3) for v in var)
    init.extend(var)
    #init=[0.08,0.12,0.3,0.15,0.22,0.13]
    cons = ({'type': 'eq', 'fun': lambda x:  1-np.sum(x[:-4])})
    res = minimize(funcFs,init , method='SLSQP',bounds=bnds,constraints = cons,tol=1e-10)
    # res = minimize(funcFs,init , method='Nelder-Mead')#,bounds=bnds)#,constraints = cons,tol=1e-10)
    res=res.x  
    return res

def findFractions(signal,species): 
    #signal is in the form [(s,g),harmonics] ie shape = (2,har,256,256)
    #species is in form [(s,g),Pure components,harmonics] ie shape = (2,nComponents,harmonics)
    def funcFs(x):
        # chisq=np.sum(np.sum(np.square((np.squeeze(signal)-np.squeeze(np.sum(x* species,2))))))
        chisq=np.sum(np.sum((signal[0]-np.sum(x*species[0],1))**2))+np.sum(np.sum((signal[1]-np.sum(x*species[1],1))**2))
        return chisq
    sz=np.size(species,2)
    bnds = [(0,1) for x in range(sz)]
    init= [0.2 for x in range(sz)]
    #init=[0.08,0.12,0.3,0.15,0.22,0.13]
    linearconstraint=LinearConstraint([1 for x in range(sz)],[0.999],[1.001])
    cons = ({'type': 'eq', 'fun': lambda x:  1-np.sum(x)})#, 'jac': lambda x:np.array(1.0 for x in range(sz))})
    # res = minimize(funcFs,init , method='SLSQP',bounds=bnds,constraints = cons,tol=1e-10)
    # res = minimize(funcFs,init , method='Nelder-Mead')#,bounds=bnds)#,constraints = cons,tol=1e-10)
    res1=0#optimize.differential_evolution(funcFs,bounds=bnds,constraints=linearconstraint)
    res=optimize.shgo(funcFs,bounds=bnds,constraints=cons,iters=1)
    res=res.x  
    return res1,res

def findFractionsinit(signal,species,init): 
    #signal is in the form [(s,g),harmonics] ie shape = (2,har,256,256)
    #species is in form [(s,g),Pure components,harmonics] ie shape = (2,nComponents,harmonics)
    def funcFs(x):
        chisq=np.sum(np.sum((np.squeeze(signal)-np.squeeze(np.sum(x*species,2)))**2))
        return chisq
    sz=np.size(species,2)
    bnds = [(0,1) for x in range(sz)]
    # init= [0.2 for x in range(sz)]
    #init=[0.08,0.12,0.3,0.15,0.22,0.13]
    cons = ({'type': 'eq', 'fun': lambda x:  1-np.sum(x), 'jac': lambda x:np.array(1.0)})
    res = minimize(funcFs,init , method='SLSQP',bounds=bnds,constraints = cons,tol=1e-7)
    #res = minimize(funcFs,init , method='anneal')#,bounds=bnds)#,constraints = cons,tol=1e-10)
    res=optimize.shgo(funcFs,bounds=bnds,constraints=cons)
    res=res.x  
    return res
def findFractionsComplete(specsignal,specspecies,lifesignal,lifespecies): 
    #signal is in the form [(s,g),harmonics] ie shape = (2,har)
    #species is in form [(s,g),Pure components,harmonics] ie shape = (2,nComponents,harmonics)
    def funcFs(x):
        chisq=np.sum(np.sum((np.squeeze(lifesignal)-np.squeeze(np.sum(x*lifespecies,2)))**2)) + np.sum(np.sum((np.squeeze(specsignal)-np.squeeze(np.sum(x*specspecies,2)))**2))   #spectrum
        return chisq
    sz=np.size(lifespecies,2)
    bnds = [(0,1) for x in range(sz)]
    # init= [0.1 for x in range(sz)]
    #init=[0.08,0.12,0.3,0.15,0.22,0.13]
    cons = ({'type': 'eq', 'fun': lambda x:  1-np.sum(x)})
    # res = minimize(funcFs,init , method='SLSQP',bounds=bnds,constraints = cons,tol=1e-10)
    res=optimize.shgo(funcFs,bounds=bnds,constraints=cons)
    #res = minimize(funcFs,init , method='Nelder-Mead')#,bounds=bnds)#,constraints = cons,tol=1e-10)
    res=res.x  
    return res

def findFractionsCompleteinit(specsignal,specspecies,lifesignal,lifespecies,init): 
    #signal is in the form [(s,g),harmonics] ie shape = (2,har)
    #species is in form [(s,g),Pure components,harmonics] ie shape = (2,nComponents,harmonics)
    def funcFs(x):
        chisq=np.sum(np.sum((np.squeeze(lifesignal)-np.squeeze(np.sum(x*lifespecies,2)))**2)) + np.sum(np.sum((np.squeeze(specsignal)-np.squeeze(np.sum(x*specspecies,2)))**2))   #spectrum
        return chisq
    sz=np.size(lifespecies,2)
    bnds = [(0,1) for x in range(sz)]
    # init= [0.1 for x in range(sz)]
    #init=[0.08,0.12,0.3,0.15,0.22,0.13]
    cons = ({'type': 'eq', 'fun': lambda x:  1-np.sum(x)})
    res = minimize(funcFs,init , method='SLSQP',bounds=bnds,constraints = cons,tol=1e-7)
    res=optimize.shgo(funcFs,bounds=bnds,constraints=cons)

    #res = minimize(funcFs,init , method='Nelder-Mead')#,bounds=bnds)#,constraints = cons,tol=1e-10)
    res=res.x  
    return res

def phasorit(tau,omega,harmonics):
    a=np.zeros((2,harmonics,tau.size))
#    for j in range(0,tau.size)
    for i in range(1,harmonics+1):
        a[1,i-1,:]=1/(1+((i*omega)**2)*(tau**2))
        a[0,i-1,:]=(i*omega*tau)/(1+(((i*omega)**2)*(tau**2)));
    return a
#
#%%

from tifffile import *
def imDecomposition(NADH,PYO,PVD,boundNadhsignal,NADH_bound,myob):
    # tau=np.zeros(1)
    # tau[0]=0.37e-9
    # NADHPerf=phasorit(tau, 2*pi*80e6,1)
    LifeSpecies=LifetimeAsPureComponentSG([NADH,PYO,PVD],1)
    LifeSpecies=np.concatenate((LifeSpecies,NADH_bound),axis=2)
    # LifeSpecies=np.concatenate((LifeSpecies,NADHPerf),axis=2)
    
    har=2
    SpecSpecies=SpectrumAsPureComponentSG([NADH,PYO,PVD],har,ch)
    SpecSpecies=np.concatenate((SpecSpecies,boundNadhsignal),axis=2)
    # signal is in the form [(s,g),harmonics,1] ie shape = (2,har,1)
    SpecSignal=np.zeros((2,har,256,256)) 
    SpecSignal[:,0,:,:] =myob.asSpectralPhasor(har=1,Fileindex=0)[:]
    SpecSignal[:,1,:,:] =myob.asSpectralPhasor(har=2,Fileindex=0)[:]
    
    # LifeSpecies=LifetimeAsPureComponentSG([NADH,PYO,PVD],1) #perfectnadh
    # LifeSpecies=np.concatenate((LifeSpecies,NADH_bound),axis=2)#perfectnadh
    
    LifeSignal=np.zeros((2,1,256,256))
    LifeSignal[0,0,:,:]=myob.lifetimePhasor[0][0]
    LifeSignal[1,0,:,:]=myob.lifetimePhasor[0][1]
    # LifeSignal[0,1,:,:]=myob.lifetimePhasor[0][2]
    # LifeSignal[1,1,:,:]=myob.lifetimePhasor[0][3]
    @jit(parallel=True)
    def fun(SpecSignal,SpecSpecies,LifeSignal,LifeSpecies,myob):
        # from scipy.optimize import minimize 
        # import numpy as np
        SpectralResde=np.zeros((4,256,256))
        LifeTimeResde=np.zeros((4,256,256))
        SpectralRes=np.zeros((4,256,256))
        LifeTimeRes=np.zeros((4,256,256))
        SpecLifeRes=np.zeros((4,256,256))
        sz=np.size(LifeSpecies,2)
        init= [0.2 for x in range(sz)]
        for i in numba.prange(256):
            for j in numba.prange(256):
                # SpectralRes[:,i,j] = findFractions(SpecSignal[:,:,i,j].reshape(2,har,1), SpecSpecies)
                if myob.specthreshold<(np.squeeze(np.mean(myob.spectralIm[0],2)))[i,j]: #Don't want to find fractions if there is nothing there!
                    SpectralResde[:,i,j],SpectralRes[:,i,j] = findFractions(SpecSignal[:,:,i,j].reshape(2,har,1), SpecSpecies)
                else: SpectralResde[:,i,j],SpectralRes[:,i,j] =np.nan,np.nan
                if i>2 and i<253:
                    if j>2 and j<253:
                        if myob.lifetimethreshold<myob.lifetimeIm[0][i,j]:
                            LifeTimeResde[:,i,j],LifeTimeRes[:,i,j]= findFractions(LifeSignal[:,:,i,j].reshape(2,1,1), LifeSpecies)
                        else: LifeTimeResde[:,i,j],LifeTimeRes[:,i,j]=np.nan,np.nan
                    else: LifeTimeResde[:,i,j],LifeTimeRes[:,i,j]=np.nan,np.nan
                else: LifeTimeResde[:,i,j],LifeTimeRes[:,i,j]=np.nan,np.nan
                # SpecLifeRes[:,i,j]= findFractionsComplete(SpecSignal[:,:,i,j].reshape(2,har,1),SpecSpecies,LifeSignal[:,:,i,j].reshape(2,har,1),LifeSpecies)
                # init=np.mean([SpectralRes[:,i,j],LifeTimeRes[:,i,j],SpecLifeRes[:,i,j]],0)
            print('{} lines left'.format(255-i))
           #perfect nadh print('LIFETIME: \n Mean fractions in line {}: \n NADH={}, \n PYO={}, \n PVD={}, \n NADH_bound={}'.format(i, np.nanmean(LifeTimeRes[0,i,:]),np.nanmean(LifeTimeRes[1,i,:]),np.nanmean(LifeTimeRes[2,i,:]),np.nanmean(LifeTimeRes[3,i,:])))
            print('LIFETIME: \n Mean fractions in line {}: \n NADH={}, \n PYO={}, \n PVD={}, \n NADH_bound={}'.format(i, np.nanmean(LifeTimeRes[0,i,:]),np.nanmean(LifeTimeRes[1,i,:]),np.nanmean(LifeTimeRes[2,i,:]),np.nanmean(LifeTimeRes[3,i,:])))
 
           # print('LIFETIME: \n Mean fractions in line {} from differential evolution: \n NADH={}, \n PYO={},\n PVD={}, \n NADH_bound={}'.format(i,np.nanmean(LifeTimeResde[0,i,:]),np.nanmean(LifeTimeResde[1,i,:]),np.nanmean(LifeTimeResde[2,i,:]),np.nanmean(LifeTimeResde[3,i,:])))
            print('SPECTRUM: \n Mean fractions in line {}: \n NADH={}, \n PYO={}, \n PVD={}, \n NADH_bound={}'.format(i, np.nanmean(SpectralRes[0,i,:]),np.nanmean(SpectralRes[1,i,:]),np.nanmean(SpectralRes[2,i,:]),np.nanmean(SpectralRes[3,i,:])))
            # print('SPECTRUM: \n Mean fractions in line {} from differential evolution: \n NADH={}, \n PYO={},\n PVD={}, \n NADH_bound={}'.format(i,np.nanmean(SpectralResde[0,i,:]),np.nanmean(SpectralResde[1,i,:]),np.nanmean(SpectralResde[2,i,:]),np.nanmean(SpectralResde[3,i,:])))
        return SpectralResde,LifeTimeResde,SpectralRes,LifeTimeRes,SpecLifeRes
    
    SpectralResde,LifeTimeResde,SpectralRes,LifeTimeRes,SpecLifeRes=fun(SpecSignal,SpecSpecies,LifeSignal,LifeSpecies,myob)
    
    
    
    
    
    s=np.zeros((256,256))
    g=np.zeros((256,256))
    for i in range(256):
        for j in range(256):
            s[i,j] = np.sum(LifeTimeRes[:,i,j]*LifeSpecies[0,0,:])
            g[i,j] = np.sum(LifeTimeRes[:,i,j]*LifeSpecies[1,0,:])
    
    H=np.histogram2d(s.flatten(),g.flatten(),bins=256,range=[[0,1],[0,1]])
    fig,ax=plt.subplots()
    plt.pcolormesh(H[1], H[2], np.log(H[0]),zorder=0)
    plt.title('Result')
    #plt.title(os.path.basename(self.lifetimefileName[fileIndex])+'\n LogPhasor')
    circ=plt.Circle((0.5,0),radius=0.5,color='black',fill=False,zorder=10)
    plt.gcf().gca().add_artist(circ)
    
    ax=plt.scatter(LifeSpecies[1,0,:],LifeSpecies[0,0,:])
    circ=plt.Circle((0.5,0),radius=0.5,color='black',fill=False,zorder=10)
    plt.gcf().gca().add_artist(circ)
    axes = plt.gca()
    axes.set_xlim([0,1])
    axes.set_ylim([0,1])
    lifeSignal=LifetimeAsPureComponentSG([myob],har)
    plt.scatter(lifeSignal[1,0,:],lifeSignal[0,0,:],marker='x')
    
    plt.show()
    
    
    
    H=np.histogram2d(myob.lifetimePhasor[0][0].flatten()*(myob.lifetimeIm[0].flatten()>myob.lifetimethreshold),myob.lifetimePhasor[0][1].flatten(),bins=256,range=[[0,1],[0,1]])
    fig,ax=plt.subplots()
    plt.pcolormesh(H[1], H[2], np.log(H[0]),zorder=0)
    plt.title('Data')
    #plt.title(os.path.basename(self.lifetimefileName[fileIndex])+'\n LogPhasor')
    circ=plt.Circle((0.5,0),radius=0.5,color='black',fill=False,zorder=10)
    plt.gcf().gca().add_artist(circ)
    plt.show()
    
    # imwrite(os.path.join(os.path.dirname(myob.lifetimefileName[0]),'SpectralResults_NADH_PYO_PVD_NADHb_firstHarmonic.tif'), np.float32(SpectralRes))
    # imwrite(os.path.join(os.path.dirname(myob.SpecfileName[0]),'LifetimeResults_NADH_PYO_PVD_NADHb_firstHarmonic.tif'), np.float32(LifeTimeRes))  
    # imwrite(os.path.join(os.path.dirname(myob.SpecfileName[0]),'SpectralLifeResults_NADH_PYO_PVD_NADHb_firstHarmonic.tif'), np.uint16(1000*SpecLifeRes), photometric='minisblack')
    return SpectralRes,LifeTimeRes


def GiveMeAHyperSignal(wavearray,signal,lambdaStart,lambdaEnd,channels):
    i=np.argmin(np.abs(wavearray-lambdaStart))
    j=np.argmin(np.abs(wavearray-lambdaEnd))
    print('Requested start={}, end={}'.format(wavearray[i],wavearray[j]))
    chans=abs(j-i)
    if chans<channels:
        print('You\'re asking for too many channels ({}), defaulting to max possible ({})'.format(channels,chans))
        return wavearray[i:j],signal[i:j]
    else: # data is oversampled. good. lets decimate like a hyperspectral detector
        sigout=np.zeros(channels)
        windows=np.int(np.floor(chans/channels))
        waveout=np.linspace(lambdaStart,lambdaEnd,channels)
        l=0
        for k in range(channels):
            sigout[k]=np.mean(signal[i+(k)*windows:i+(k+1)*windows])
            l+=1
        if windows<channels:
            sigout[-1]=np.mean(signal[i+(l)*channels:i+(l+2)*windows]) ##Not sure 
        print('returned start={}, end={}'.format(waveout[0],waveout[-1]))
        return waveout,sigout
def specphasor(sp,si,co):
    s=np.sum(si*sp,0)
    s=s/(np.sum(sp,0)+np.finfo(float).eps)
    g=np.sum(co*sp,0)
    g=g/(np.sum(sp,0)+np.finfo(float).eps)
    return s,g
def SGSignalfromLeosSpectrum(signal):
    ch=np.size(signal[0])
    har=1
    t=np.linspace(0,1,ch)
    si=np.sin(har*2*np.pi*t)
    co=np.cos(har*2*np.pi*t)
    ns,ng=specphasor(signal[1],si,co)
    har=2
    t=np.linspace(0,1,ch)
    si=np.sin(har*2*np.pi*t)
    co=np.cos(har*2*np.pi*t)
    ns2,ng2=specphasor(signal[1],si,co)
    boundNadhsignal=np.zeros((2,2,1))
    boundNadhsignal[0,0,0]=ns
    boundNadhsignal[1,0,0]=ng
    boundNadhsignal[0,1,0]=ns2
    boundNadhsignal[1,1,0]=ng2
    return boundNadhsignal




def imDecomposition_spec(NADH,PYO,PVD,boundNadhsignal,myob,ch):
    # tau=np.zeros(1)
    # tau[0]=0.37e-9
    # NADHPerf=phasorit(tau, 2*pi*80e6,1)
    # LifeSpecies=LifetimeAsPureComponentSG([NADH,PYO,PVD],1)
    # LifeSpecies=np.concatenate((LifeSpecies,NADH_bound),axis=2)
    # LifeSpecies=np.concatenate((LifeSpecies,NADHPerf),axis=2)
    
    har=2
    SpecSpecies=SpectrumAsPureComponentSG([NADH,PYO,PVD],har,ch)
    SpecSpecies=np.concatenate((SpecSpecies,boundNadhsignal),axis=2)
    # signal is in the form [(s,g),harmonics,1] ie shape = (2,har,1)
    SpecSignal=np.zeros((2,har,256,256)) 
    SpecSignal[:,0,:,:] =myob.asSpectralPhasor(har=1,Fileindex=0,ch=ch)[:]
    SpecSignal[:,1,:,:] =myob.asSpectralPhasor(har=2,Fileindex=0,ch=ch)[:]
    
    # LifeSpecies=LifetimeAsPureComponentSG([NADH,PYO,PVD],1) #perfectnadh
    # LifeSpecies=np.concatenate((LifeSpecies,NADH_bound),axis=2)#perfectnadh
    
    # LifeSignal=np.zeros((2,1,256,256))
    # LifeSignal[0,0,:,:]=myob.lifetimePhasor[0][0]
    # LifeSignal[1,0,:,:]=myob.lifetimePhasor[0][1]
    # LifeSignal[0,1,:,:]=myob.lifetimePhasor[0][2]
    # LifeSignal[1,1,:,:]=myob.lifetimePhasor[0][3]
    @jit(parallel=True)
    def fun(SpecSignal,SpecSpecies,myob):
        # from scipy.optimize import minimize 
        # import numpy as np
        SpectralResde=np.zeros((4,256,256))
        # LifeTimeResde=np.zeros((4,256,256))
        SpectralRes=np.zeros((4,256,256))
        # LifeTimeRes=np.zeros((4,256,256))
        SpecLifeRes=np.zeros((4,256,256))
        sz=np.size(SpecSpecies,2)
        init= [0.2 for x in range(sz)]
        for i in numba.prange(256):
            for j in numba.prange(256):
                # SpectralRes[:,i,j] = findFractions(SpecSignal[:,:,i,j].reshape(2,har,1), SpecSpecies)
                if myob.specthreshold<(np.squeeze(np.mean(myob.spectralIm[0],2)))[i,j]: #Don't want to find fractions if there is nothing there!
                    SpectralResde[:,i,j],SpectralRes[:,i,j] = findFractions(SpecSignal[:,:,i,j].reshape(2,har,1), SpecSpecies)
                else: SpectralResde[:,i,j],SpectralRes[:,i,j] =np.nan,np.nan
                # LifeTimeResde[:,i,j],LifeTimeRes[:,i,j]=np.nan,np.nan
                # SpecLifeRes[:,i,j]= findFractionsComplete(SpecSignal[:,:,i,j].reshape(2,har,1),SpecSpecies,LifeSignal[:,:,i,j].reshape(2,har,1),LifeSpecies)
                # init=np.mean([SpectralRes[:,i,j],LifeTimeRes[:,i,j],SpecLifeRes[:,i,j]],0)
            print('{} lines left'.format(255-i))
           #perfect nadh print('LIFETIME: \n Mean fractions in line {}: \n NADH={}, \n PYO={}, \n PVD={}, \n NADH_bound={}'.format(i, np.nanmean(LifeTimeRes[0,i,:]),np.nanmean(LifeTimeRes[1,i,:]),np.nanmean(LifeTimeRes[2,i,:]),np.nanmean(LifeTimeRes[3,i,:]))) 
           # print('LIFETIME: \n Mean fractions in line {} from differential evolution: \n NADH={}, \n PYO={},\n PVD={}, \n NADH_bound={}'.format(i,np.nanmean(LifeTimeResde[0,i,:]),np.nanmean(LifeTimeResde[1,i,:]),np.nanmean(LifeTimeResde[2,i,:]),np.nanmean(LifeTimeResde[3,i,:])))
            print('SPECTRUM: \n Mean fractions in line {}: \n NADH={}, \n PYO={}, \n PVD={}, \n NADH_bound={}'.format(i, np.nanmean(SpectralRes[0,i,:]),np.nanmean(SpectralRes[1,i,:]),np.nanmean(SpectralRes[2,i,:]),np.nanmean(SpectralRes[3,i,:])))
            # print('SPECTRUM: \n Mean fractions in line {} from differential evolution: \n NADH={}, \n PYO={},\n PVD={}, \n NADH_bound={}'.format(i,np.nanmean(SpectralResde[0,i,:]),np.nanmean(SpectralResde[1,i,:]),np.nanmean(SpectralResde[2,i,:]),np.nanmean(SpectralResde[3,i,:])))
        return SpectralResde,SpectralRes,SpecLifeRes
    
    SpectralResde,SpectralRes,SpecLifeRes=fun(SpecSignal,SpecSpecies,myob)
    
    
    
    return SpectralRes


def imDecomposition_life(NADH,PYO,PVD,NADH_bound,myob):
    # tau=np.zeros(1)
    # tau[0]=0.37e-9
    # NADHPerf=phasorit(tau, 2*pi*80e6,1)
    LifeSpecies=LifetimeAsPureComponentSG([NADH,PYO,PVD],1)
    LifeSpecies=np.concatenate((LifeSpecies,NADH_bound),axis=2)
    # LifeSpecies=np.concatenate((LifeSpecies,NADHPerf),axis=2)
    
        # LifeSpecies=LifetimeAsPureComponentSG([NADH,PYO,PVD],1) #perfectnadh
    # LifeSpecies=np.concatenate((LifeSpecies,NADH_bound),axis=2)#perfectnadh
    
    LifeSignal=np.zeros((2,1,256,256))
    LifeSignal[0,0,:,:]=myob.lifetimePhasor[0][0]
    LifeSignal[1,0,:,:]=myob.lifetimePhasor[0][1]
    # LifeSignal[0,1,:,:]=myob.lifetimePhasor[0][2]
    # LifeSignal[1,1,:,:]=myob.lifetimePhasor[0][3]
    @jit(parallel=True)
    def fun(LifeSignal,LifeSpecies,myob):
        # from scipy.optimize import minimize 
        # import numpy as np
        # SpectralResde=np.zeros((4,256,256))
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
                            LifeTimeResde[:,i,j],LifeTimeRes[:,i,j]= findFractions(LifeSignal[:,:,i,j].reshape(2,1,1), LifeSpecies)
                        else: LifeTimeResde[:,i,j],LifeTimeRes[:,i,j]=np.nan,np.nan
                    else: LifeTimeResde[:,i,j],LifeTimeRes[:,i,j]=np.nan,np.nan
                else: LifeTimeResde[:,i,j],LifeTimeRes[:,i,j]=np.nan,np.nan
                # SpecLifeRes[:,i,j]= findFractionsComplete(SpecSignal[:,:,i,j].reshape(2,har,1),SpecSpecies,LifeSignal[:,:,i,j].reshape(2,har,1),LifeSpecies)
                # init=np.mean([SpectralRes[:,i,j],LifeTimeRes[:,i,j],SpecLifeRes[:,i,j]],0)
            print('{} lines left'.format(255-i))
           #perfect nadh print('LIFETIME: \n Mean fractions in line {}: \n NADH={}, \n PYO={}, \n PVD={}, \n NADH_bound={}'.format(i, np.nanmean(LifeTimeRes[0,i,:]),np.nanmean(LifeTimeRes[1,i,:]),np.nanmean(LifeTimeRes[2,i,:]),np.nanmean(LifeTimeRes[3,i,:])))
            print('LIFETIME: \n Mean fractions in line {}: \n NADH={}, \n PYO={}, \n PVD={}, \n NADH_bound={}'.format(i, np.nanmean(LifeTimeRes[0,i,:]),np.nanmean(LifeTimeRes[1,i,:]),np.nanmean(LifeTimeRes[2,i,:]),np.nanmean(LifeTimeRes[3,i,:])))
 
           # print('LIFETIME: \n Mean fractions in line {} from differential evolution: \n NADH={}, \n PYO={},\n PVD={}, \n NADH_bound={}'.format(i,np.nanmean(LifeTimeResde[0,i,:]),np.nanmean(LifeTimeResde[1,i,:]),np.nanmean(LifeTimeResde[2,i,:]),np.nanmean(LifeTimeResde[3,i,:])))
            # print('SPECTRUM: \n Mean fractions in line {}: \n NADH={}, \n PYO={}, \n PVD={}, \n NADH_bound={}'.format(i, np.nanmean(SpectralRes[0,i,:]),np.nanmean(SpectralRes[1,i,:]),np.nanmean(SpectralRes[2,i,:]),np.nanmean(SpectralRes[3,i,:])))
            # print('SPECTRUM: \n Mean fractions in line {} from differential evolution: \n NADH={}, \n PYO={},\n PVD={}, \n NADH_bound={}'.format(i,np.nanmean(SpectralResde[0,i,:]),np.nanmean(SpectralResde[1,i,:]),np.nanmean(SpectralResde[2,i,:]),np.nanmean(SpectralResde[3,i,:])))
        return LifeTimeResde,LifeTimeRes
    
    LifeTimeResde,LifeTimeRes=fun(LifeSignal,LifeSpecies,myob)
    
    
    # imwrite(os.path.join(os.path.dirname(myob.lifetimefileName[0]),'SpectralResults_NADH_PYO_PVD_NADHb_firstHarmonic.tif'), np.float32(SpectralRes))
    # imwrite(os.path.join(os.path.dirname(myob.SpecfileName[0]),'LifetimeResults_NADH_PYO_PVD_NADHb_firstHarmonic.tif'), np.float32(LifeTimeRes))  
    # imwrite(os.path.join(os.path.dirname(myob.SpecfileName[0]),'SpectralLifeResults_NADH_PYO_PVD_NADHb_firstHarmonic.tif'), np.uint16(1000*SpecLifeRes), photometric='minisblack')
    return LifeTimeRes
