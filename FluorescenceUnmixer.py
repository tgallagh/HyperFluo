# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:14:59 2020

@author: Simon Leemans
"""
from fluorObjects import *
import numba
from numba import jit

tkinter.messagebox.showinfo(
    title="FluorObject",
    message="To make a FluorObject, select corresponding LSM and R64 files",
)

NADH = FluorObject("NADH")
NADH.addLSM()
NADH.add_refFile()

fullNADH = FluorObject("NADH Bound")
fullNADH.addLSM()
xx, mm = fullNADH.meanSpectrum(0)
# a=np.abs(ft.ifft(ft.fft(scipy.signal.medfilt(mm,3))*ft.fftshift(np.exp(1j*(np.linspace(0,i,32))))))
x = xx[:8]
aa = a[:8]


def specphasor(sp, si, co):
    s = np.sum(si * sp, 0)
    s = s / (np.sum(sp, 0) + np.finfo(float).eps)
    g = np.sum(co * sp, 0)
    g = g / (np.sum(sp, 0) + np.finfo(float).eps)
    return s, g


har = 1
t = np.linspace(0, 1, 8)
si = np.sin(har * 2 * np.pi * t)
co = np.cos(har * 2 * np.pi * t)
ns, ng = specphasor(aa, si, co)
har = 2
t = np.linspace(0, 1, 8)
si = np.sin(har * 2 * np.pi * t)
co = np.cos(har * 2 * np.pi * t)
ns2, ng2 = specphasor(aa, si, co)
boundNadhsignal = np.zeros((2, 2, 1))
boundNadhsignal[0, 0, 0] = ns
boundNadhsignal[1, 0, 0] = ng
boundNadhsignal[0, 1, 0] = ns2
boundNadhsignal[1, 1, 0] = ng2


# NADH_bound=FluorObject('NADH_bound')
tau = np.zeros(1)
tau[0] = 3.2e-9
NADH_bound = phasorit(tau, 2 * pi * 80e6, 2)


PYO = FluorObject("PYO")
PYO.addLSM()
PYO.add_refFile()

PVD = FluorObject("PVD")
PVD.addLSM()
PVD.add_refFile()

myob = FluorObject("sample")
myob.add_refFile()
myob.addLSM()
myob.specthreshold = 0
myob.lifetimethreshold = 0

if len(myob.spectralIm) > 0:
    ans = "n"
    myob.showSpecImage(0)
    while ans != "y":
        plt.hist(
            np.mean(myob.spectralIm[0], 2).flatten()
            * (np.mean(myob.spectralIm[0], 2).flatten() > myob.specthreshold),
            bins=100,
            range=[2, np.max(myob.spectralIm[0])],
        )
        plt.show()
        myob.specthreshold = np.round(
            float(input("Where (on x axis) do you want to set threshold"))
        )
        plt.imshow(
            (np.squeeze(np.mean(myob.spectralIm[0], 2)) > myob.specthreshold)
            * np.squeeze(np.mean(myob.spectralIm[0], 2))
        )
        plt.show()
        ans = input("Satisfied? answer y or n")

if len(myob.lifetimeIm) > 0:
    ans = "n"
    myob.showSpecImage(0)
    while ans != "y":
        plt.hist(
            myob.lifetimeIm[0].flatten()
            * (myob.lifetimeIm[0].flatten() > myob.lifetimethreshold),
            bins=100,
            range=[2, np.max(myob.lifetimeIm[0])],
        )
        plt.show()
        myob.lifetimethreshold = np.round(
            float(input("Where (on x axis) do you want to set threshold"))
        )
        plt.imshow(myob.lifetimeIm[0] * (myob.lifetimeIm[0] > myob.lifetimethreshold))
        plt.show()
        ans = input("Satisfied? answer y or n")


ASMPa14coverslip = FluorObject("tara")
ASMPa14coverslip.addLSM()
ASMPa14coverslip.add_refFile()
ASMPa14coverslip.specthreshold = 0
ASMPa14coverslip.lifetimethreshold = 0

ASMPa14Nocoverslip = FluorObject("tara")
ASMPa14Nocoverslip.addLSM()
ASMPa14Nocoverslip.add_refFile()
ASMPa14Nocoverslip.specthreshold = 0
ASMPa14Nocoverslip.lifetimethreshold = 0
ASMDphzcoverslip = FluorObject("tara")
ASMDphzcoverslip.addLSM()
ASMDphzcoverslip.add_refFile()
ASMDphzcoverslip.specthreshold = 0
ASMDphzcoverslip.lifetimethreshold = 0
ASMDphzNocoverslip = FluorObject("tara")
ASMDphzNocoverslip.addLSM()
ASMDphzNocoverslip.add_refFile()
ASMDphzNocoverslip.specthreshold = 0
ASMDphzNocoverslip.lifetimethreshold = 0


# Taras plots:
s, g = (
    ASMPa14coverslip.asSpectralPhasor(1, 0)[0].flatten(),
    ASMPa14coverslip.asSpectralPhasor(1, 0)[1].flatten(),
)
s, g = (
    np.concatenate((s, ASMPa14Nocoverslip.asSpectralPhasor(1, 0)[0].flatten())),
    np.concatenate((g, ASMPa14Nocoverslip.asSpectralPhasor(1, 0)[1].flatten())),
)
s, g = (
    np.concatenate((s, ASMDphzcoverslip.asSpectralPhasor(1, 0)[0].flatten())),
    np.concatenate((g, ASMDphzcoverslip.asSpectralPhasor(1, 0)[1].flatten())),
)
s, g = (
    np.concatenate((s, ASMDphzNocoverslip.asSpectralPhasor(1, 0)[0].flatten())),
    np.concatenate((g, ASMDphzNocoverslip.asSpectralPhasor(1, 0)[1].flatten())),
)

har = 2

H = np.histogram2d(s.flatten(), g.flatten(), bins=256, range=[[-1, 1], [-1, 1]])
fig, ax = plt.subplots()
plt.pcolormesh(H[1], H[2], np.log(H[0]), zorder=0)
plt.title(
    "All files \n 8 channels: {} nm - {} nm".format(
        ASMPa14coverslip.spectralRange[0][0], ASMPa14coverslip.spectralRange[0][-1]
    )
)
# plt.title(os.path.basename(self.lifetimefileName[fileIndex])+'\n LogPhasor')
circ = plt.Circle((0.0, 0), radius=1, color="black", fill=False, zorder=10)
plt.gcf().gca().add_artist(circ)

SpecSpecies = SpectrumAsPureComponentSG([NADH, PYO, PVD], har)
SpecSpecies = np.concatenate((SpecSpecies, boundNadhsignal), axis=2)

ax = plt.scatter(SpecSpecies[1, 0, 0], SpecSpecies[0, 0, 0], color="r")
ax = plt.scatter(SpecSpecies[1, 0, 1], SpecSpecies[0, 0, 1], color="g")
ax = plt.scatter(SpecSpecies[1, 0, 2], SpecSpecies[0, 0, 2], color="k")
ax = plt.scatter(SpecSpecies[1, 0, 3], SpecSpecies[0, 0, 3], color="b")

circ = plt.Circle((0.0, 0), radius=1, color="black", fill=False, zorder=10)
plt.gcf().gca().add_artist(circ)
axes = plt.gca()
axes.set_xlim([-1, 1])
axes.set_ylim([-1, 1])
# plt.scatter(SpecSignal[1,0,:],SpecSignal[0,0,:],marker='x')


x1, mean1 = NADH.meanSpectrum(0)[0], NADH.meanSpectrum(0)[1] / (
    np.max(NADH.meanSpectrum(0)[1])
)
x2, mean2 = PYO.meanSpectrum(0)[0], PYO.meanSpectrum(0)[1] / (
    np.max(PYO.meanSpectrum(0)[1])
)
x3, mean3 = PVD.meanSpectrum(0)[0], PVD.meanSpectrum(0)[1] / (
    np.max(PVD.meanSpectrum(0)[1])
)
x4, mean4 = x, aa / np.max(aa)
fig, ax = plt.subplots()
ax.plot(x1, mean1, label="NADH")
ax.plot(x2, mean2, label="PYO")
ax.plot(x3, mean3, label="PVD")
ax.plot(x4, mean4, label="BoundNADH")
plt.title("Pure Species")
ax.legend()

x1, mean1 = ASMPa14coverslip.meanSpectrum(0)[0], ASMPa14coverslip.meanSpectrum(0)[1] / (
    np.max(ASMPa14coverslip.meanSpectrum(0)[1])
)
x2, mean2 = (
    ASMPa14Nocoverslip.meanSpectrum(0)[0],
    ASMPa14Nocoverslip.meanSpectrum(0)[1]
    / (np.max(ASMPa14Nocoverslip.meanSpectrum(0)[1])),
)
x3, mean3 = ASMDphzcoverslip.meanSpectrum(0)[0], ASMDphzcoverslip.meanSpectrum(0)[1] / (
    np.max(ASMDphzcoverslip.meanSpectrum(0)[1])
)
x4, mean4 = (
    ASMDphzNocoverslip.meanSpectrum(0)[0],
    ASMDphzNocoverslip.meanSpectrum(0)[1]
    / (np.max(ASMDphzNocoverslip.meanSpectrum(0)[1])),
)

fig, ax = plt.subplots()

# ax.plot(x1,mean1,label='ASMPa14coverslip')
ax.plot(x1, mean1, label="ASMPa14coverslip")
ax.plot(x2, mean2, label="ASMPa14Nocoverslip")
ax.plot(x3, mean3, label="ASMDphzCoverslip")
ax.plot(x4, mean4, label="ASMDphzNocoverslip")
plt.title("Samples")
ax.legend()


allsamples = [
    ASMPa14coverslip,
    ASMPa14Nocoverslip,
    ASMDphzcoverslip,
    ASMDphzNocoverslip,
]
resultsSpec = []
resultsLife = []
for ob in allsamples:
    SpectralRes, LifeTimeRes = imDecomposition(NADH, PYO, PVD, boundNadhsignal, ob)
    resultsSpec.append(SpectralRes)
    resultsLife.append(LifeTimeRes)

i = 0
for ob in allsamples:
    resultsim = 1000 * resultsSpec[i]
    meanspecim = np.mean(np.squeeze(ob.spectralIm[0]), axis=0)
    normspecim = 1000 * meanspecim / np.max(meanspecim)
    normspecim = np.expand_dims(normspecim, axis=0)
    a = np.concatenate((resultsim, normspecim), axis=0)
    a = np.uint16(a)
    imwrite(
        os.path.join(
            os.path.dirname(ob.lifetimefileName[0]),
            "SpectralResults_NADH_PYO_PVD_NADHb_firstHarmonic.tif",
        ),
        a,
        photometric="minisblack",
    )
    resultsim = 1000 * resultsLife[i]
    meanspecim = ob.lifetimeIm[0]
    normspecim = 1000 * meanspecim / np.max(meanspecim)
    normspecim = np.expand_dims(normspecim, axis=0)
    a = np.concatenate((resultsim, normspecim))
    a = np.uint16(a)
    imwrite(
        os.path.join(
            os.path.dirname(ob.SpecfileName[0]),
            "LifetimeResults_NADH_PYO_PVD_NADHb_firstHarmonic.tif",
        ),
        a,
        photometric="minisblack",
    )
    i += 1

i = 0
for SpectralRes in resultsSpec:
    # histogram
    plt.figure()
    plt.hist(
        [
            SpectralRes[i].flatten()[~np.isnan(SpectralRes[i].flatten())]
            for i in range(4)
        ],
        bins=15,
        range=[0, 1],
        label=["NADH", "PYO", "PVD", "NADH bound"],
    )
    plt.title(
        os.path.basename(allsamples[i].SpecfileName[0])[:25] + "\n Spectral results"
    )
    plt.legend()
    plt.figure()
    allsamples[i].showSpecImage(0)

    H = np.histogram2d(
        allsamples[i].asSpectralPhasor(1, 0)[0].flatten(),
        allsamples[i].asSpectralPhasor(1, 0)[1].flatten(),
        bins=256,
        range=[[-1, 1], [-1, 1]],
    )
    plt.pcolormesh(H[1], H[2], np.log(H[0]), zorder=0)
    plt.title(
        os.path.basename(allsamples[i].SpecfileName[0])[:25] + "\n Spectral results"
    )
    circ = plt.Circle((00, 0), radius=1, color="black", fill=False, zorder=10)
    plt.gcf().gca().add_artist(circ)
    plt.show()
    # plt.hist2d(s.flatten(),g.flatten(),range=[[-1, 1],[-1,1]],bins=1000)
    i += 1

k = 0
for Liferes in resultsLife:
    # histogram
    plt.figure()
    plt.hist(
        [Liferes[i].flatten()[~np.isnan(Liferes[i].flatten())] for i in range(4)],
        bins=15,
        range=[0, 1],
        label=["NADH", "PYO", "PVD", "NADH bound"],
    )
    plt.title(
        os.path.basename(allsamples[k].lifetimefileName[0])[:20] + "\n Lifetime results"
    )
    plt.legend()
    plt.figure()
    allsamples[k].showLifeImage(0)
    allsamples[k].showLifePhasor(0, 0)
    s = np.zeros((256, 256))
    g = np.zeros((256, 256))
    for i in range(256):
        for j in range(256):
            s[i, j] = np.sum(Liferes[:, i, j] * LifeSpecies[0, 0, :])
            g[i, j] = np.sum(Liferes[:, i, j] * LifeSpecies[1, 0, :])
    H = np.histogram2d(s.flatten(), g.flatten(), bins=256, range=[[0, 1], [0, 1]])
    fig, ax = plt.subplots()
    plt.pcolormesh(H[1], H[2], np.log(H[0]), zorder=0)
    plt.title("Fitting Result")
    # plt.title(os.path.basename(self.lifetimefileName[fileIndex])+'\n LogPhasor')
    circ = plt.Circle((0.5, 0), radius=0.5, color="black", fill=False, zorder=10)
    plt.gcf().gca().add_artist(circ)

    ax = plt.scatter(LifeSpecies[1, 0, :], LifeSpecies[0, 0, :])
    circ = plt.Circle((0.5, 0), radius=0.5, color="black", fill=False, zorder=10)
    plt.gcf().gca().add_artist(circ)
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    lifeSignal = LifetimeAsPureComponentSG([myob], 1)
    plt.scatter(lifeSignal[1, 0, :], lifeSignal[0, 0, :], marker="x")
    plt.show()
    k += 1

plt.hist(
    SpectralRes[0].flatten()[~np.isnan(SpectralRes[0].flatten())],
    bins=50,
    range=[0, 1],
    label="spectralNADH",
)
plt.title("Fractional Contribution - Spectral \n NADH")
plt.legend()

plt.figure()
plt.hist(
    LifeTimeRes[0].flatten()[~np.isnan(LifeTimeRes[0].flatten())],
    bins=50,
    range=[0, 1],
    label="lifetimeNADH",
)
plt.title("Fractional Contribution - LifeTime \n NADH")
plt.legend()

plt.figure()
plt.hist(
    SpectralRes[1].flatten()[~np.isnan(SpectralRes[1].flatten())],
    bins=50,
    range=[0, 1],
    label="spectralPYO",
)
plt.title("Fractional Contribution - Spectral \n PYO")
plt.legend()

plt.figure()
plt.hist(
    LifeTimeRes[1].flatten()[~np.isnan(LifeTimeRes[1].flatten())],
    bins=50,
    range=[0, 1],
    label="lifetimePYO",
)
plt.title("Fractional Contribution - LifeTime \n PYO")
plt.legend()

plt.figure()
plt.hist(
    SpectralRes[2].flatten()[~np.isnan(SpectralRes[2].flatten())],
    bins=50,
    range=[0, 1],
    label="spectralPVD",
)
plt.title("Fractional Contribution - Spectral \n PVD")
plt.legend()

plt.figure()
plt.hist(
    LifeTimeRes[2].flatten()[~np.isnan(LifeTimeRes[2].flatten())],
    bins=50,
    range=[0, 1],
    label="lifetimePVD",
)
plt.title("Fractional Contribution - LifeTime \n PVD")
plt.legend()

plt.figure()
plt.hist(
    SpectralRes[3].flatten()[~np.isnan(SpectralRes[3].flatten())],
    bins=50,
    range=[0, 1],
    label="spectralPVD",
)
plt.title("Fractional Contribution - Spectral \n PVD")
plt.legend()

plt.figure()
plt.hist(
    LifeTimeRes[3].flatten()[~np.isnan(LifeTimeRes[3].flatten())],
    bins=50,
    range=[0, 1],
    label="lifetimePVD",
)
plt.title("Fractional Contribution - LifeTime \n PVD")
plt.legend()


s = np.zeros((256, 256))
g = np.zeros((256, 256))
for i in range(256):
    for j in range(256):
        s[i, j] = np.sum(LifeTimeRes[:, i, j] * LifeSpecies[0, 0, :])
        g[i, j] = np.sum(LifeTimeRes[:, i, j] * LifeSpecies[1, 0, :])

H = np.histogram2d(s.flatten(), g.flatten(), bins=256, range=[[0, 1], [0, 1]])
fig, ax = plt.subplots()
plt.pcolormesh(H[1], H[2], np.log(H[0]), zorder=0)
plt.title("Result")
# plt.title(os.path.basename(self.lifetimefileName[fileIndex])+'\n LogPhasor')
circ = plt.Circle((0.5, 0), radius=0.5, color="black", fill=False, zorder=10)
plt.gcf().gca().add_artist(circ)

ax = plt.scatter(LifeSpecies[1, 0, :], LifeSpecies[0, 0, :])
circ = plt.Circle((0.5, 0), radius=0.5, color="black", fill=False, zorder=10)
plt.gcf().gca().add_artist(circ)
axes = plt.gca()
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
lifeSignal = LifetimeAsPureComponentSG([myob], har)
plt.scatter(lifeSignal[1, 0, :], lifeSignal[0, 0, :], marker="x")

plt.show()


H = np.histogram2d(
    myob.lifetimePhasor[0][0].flatten()
    * (myob.lifetimeIm[0].flatten() > myob.lifetimethreshold),
    myob.lifetimePhasor[0][1].flatten(),
    bins=256,
    range=[[0, 1], [0, 1]],
)
fig, ax = plt.subplots()
plt.pcolormesh(H[1], H[2], np.log(H[0]), zorder=0)
plt.title("Data")
# plt.title(os.path.basename(self.lifetimefileName[fileIndex])+'\n LogPhasor')
circ = plt.Circle((0.5, 0), radius=0.5, color="black", fill=False, zorder=10)
plt.gcf().gca().add_artist(circ)
plt.show()


#%%
har = 2
NADH_bound = phasorit(tau, 2 * pi * 80e6, har)
LifeSpecies = LifetimeAsPureComponentSG([NADH, PYO, PVD], har)
LifeSpecies = np.concatenate((LifeSpecies, NADH_bound), axis=2)
ax = plt.scatter(LifeSpecies[1, 0, :], LifeSpecies[0, 0, :])
circ = plt.Circle((0.5, 0), radius=0.5, color="black", fill=False, zorder=10)
plt.gcf().gca().add_artist(circ)
axes = plt.gca()
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])

lifeSignal = LifetimeAsPureComponentSG([myob], har)
plt.scatter(lifeSignal[1, 0, :], lifeSignal[0, 0, :], marker="x")

# MLE optimization
# var=myob.lifeSG_Stats(0)[1]
# res=findFractionsMLE(lifeSignal,LifeSpecies,var)
# comboPredict = np.sum(res[:-4]*LifeSpecies,2)
# plt.scatter(comboPredict[1,0],comboPredict[0,0],marker='+',edgecolors='g')

# LLS Optimization
res_lifetimede, res_lifetime = findFractions(lifeSignal, LifeSpecies)
comboPredict = np.sum(res_lifetime * LifeSpecies, 2)
plt.scatter(comboPredict[1, 0], comboPredict[0, 0], marker=5, edgecolors="r")
comboPredict = np.sum(res_lifetimede * LifeSpecies, 2)
plt.scatter(comboPredict[1, 0], comboPredict[0, 0], marker=6, edgecolors="r")
#%%

har = 2
plt.figure()
SpecSpecies = SpectrumAsPureComponentSG([NADH, PYO, PVD], har)
SpecSignal = SpectrumAsPureComponentSG([myob], har)

ax = plt.scatter(SpecSpecies[1, 0, :], SpecSpecies[0, 0, :])
circ = plt.Circle((0.0, 0), radius=1, color="black", fill=False, zorder=10)
plt.gcf().gca().add_artist(circ)
axes = plt.gca()
axes.set_xlim([-1, 1])
axes.set_ylim([-1, 1])
plt.scatter(SpecSignal[1, 0, :], SpecSignal[0, 0, :], marker="x")

plt.figure()
ax = plt.scatter(SpecSpecies[1, 0, :], SpecSpecies[0, 0, :])
circ = plt.Circle((0.0, 0), radius=1, color="black", fill=False, zorder=10)
plt.gcf().gca().add_artist(circ)
axes = plt.gca()
axes.set_xlim([-1, 1])
axes.set_ylim([-1, 1])
plt.scatter(SpecSignal[1, 0, :], SpecSignal[0, 0, :], marker="x")

res, res_spectral = findFractions(SpecSignal, SpecSpecies)
comboPredict = np.sum(res_spectral * SpecSpecies, 2)
plt.scatter(comboPredict[1, 0], comboPredict[0, 0], marker=5, edgecolors="r")

#%%
# Image Decomposition

# print('To start up the ole parallel computer you need to \n open an anaconda prompt and run:\n ')
# print('ipcluster start -n 4')

# from ipyparallel import Client
# rc = Client()
# if np.size(rc.ids)<1:
#     print('sorry couldn''t find clusters...')
# else: print(rc.ids)
# dview = rc[:] # use all engines

# from multiprocessing import Pool
# import multiprocessing as mp
# # pool = Pool(4)
from tifffile import *


def imDecomposition(NADH, PYO, PVD, boundNadhsignal, myob):
    tau = np.zeros(1)
    tau[0] = 3.2e-9
    NADH_bound = phasorit(tau, 2 * pi * 80e6, 1)

    har = 2
    SpecSpecies = SpectrumAsPureComponentSG([NADH, PYO, PVD], har)
    SpecSpecies = np.concatenate((SpecSpecies, boundNadhsignal), axis=2)
    # signal is in the form [(s,g),harmonics,1] ie shape = (2,har,1)
    SpecSignal = np.zeros((2, har, 256, 256))
    SpecSignal[:, 0, :, :] = myob.asSpectralPhasor(har=1, Fileindex=0)[:]
    SpecSignal[:, 1, :, :] = myob.asSpectralPhasor(har=2, Fileindex=0)[:]

    LifeSpecies = LifetimeAsPureComponentSG([NADH, PYO, PVD], 1)
    LifeSpecies = np.concatenate((LifeSpecies, NADH_bound), axis=2)

    LifeSignal = np.zeros((2, 1, 256, 256))
    LifeSignal[0, 0, :, :] = myob.lifetimePhasor[0][0]
    LifeSignal[1, 0, :, :] = myob.lifetimePhasor[0][1]
    # LifeSignal[0,1,:,:]=myob.lifetimePhasor[0][2]
    # LifeSignal[1,1,:,:]=myob.lifetimePhasor[0][3]
    @jit(parallel=True)
    def fun(SpecSignal, SpecSpecies, LifeSignal, LifeSpecies, myob):
        # from scipy.optimize import minimize
        # import numpy as np
        SpectralResde = np.zeros((4, 256, 256))
        LifeTimeResde = np.zeros((4, 256, 256))
        SpectralRes = np.zeros((4, 256, 256))
        LifeTimeRes = np.zeros((4, 256, 256))
        SpecLifeRes = np.zeros((4, 256, 256))
        sz = np.size(LifeSpecies, 2)
        init = [0.2 for x in range(sz)]
        for i in numba.prange(256):
            for j in numba.prange(256):
                # SpectralRes[:,i,j] = findFractions(SpecSignal[:,:,i,j].reshape(2,har,1), SpecSpecies)
                if (
                    myob.specthreshold
                    < (np.squeeze(np.mean(myob.spectralIm[0], 2)))[i, j]
                ):  # Don't want to find fractions if there is nothing there!
                    SpectralResde[:, i, j], SpectralRes[:, i, j] = findFractions(
                        SpecSignal[:, :, i, j].reshape(2, har, 1), SpecSpecies
                    )
                else:
                    SpectralResde[:, i, j], SpectralRes[:, i, j] = np.nan, np.nan
                if i > 2 and i < 253:
                    if j > 2 and j < 253:
                        if myob.lifetimethreshold < myob.lifetimeIm[0][i, j]:
                            (
                                LifeTimeResde[:, i, j],
                                LifeTimeRes[:, i, j],
                            ) = findFractions(
                                LifeSignal[:, :, i, j].reshape(2, 1, 1), LifeSpecies
                            )
                        else:
                            LifeTimeResde[:, i, j], LifeTimeRes[:, i, j] = (
                                np.nan,
                                np.nan,
                            )
                    else:
                        LifeTimeResde[:, i, j], LifeTimeRes[:, i, j] = np.nan, np.nan
                else:
                    LifeTimeResde[:, i, j], LifeTimeRes[:, i, j] = np.nan, np.nan
                # SpecLifeRes[:,i,j]= findFractionsComplete(SpecSignal[:,:,i,j].reshape(2,har,1),SpecSpecies,LifeSignal[:,:,i,j].reshape(2,har,1),LifeSpecies)
                # init=np.mean([SpectralRes[:,i,j],LifeTimeRes[:,i,j],SpecLifeRes[:,i,j]],0)
            print("{} lines left".format(255 - i))
            print(
                "LIFETIME: \n Mean fractions in line {}: \n NADH={}, \n PYO={}, \n PVD={}, \n NADH_bound={}".format(
                    i,
                    np.nanmean(LifeTimeRes[0, i, :]),
                    np.nanmean(LifeTimeRes[1, i, :]),
                    np.nanmean(LifeTimeRes[2, i, :]),
                    np.nanmean(LifeTimeRes[3, i, :]),
                )
            )
            # print('LIFETIME: \n Mean fractions in line {} from differential evolution: \n NADH={}, \n PYO={},\n PVD={}, \n NADH_bound={}'.format(i,np.nanmean(LifeTimeResde[0,i,:]),np.nanmean(LifeTimeResde[1,i,:]),np.nanmean(LifeTimeResde[2,i,:]),np.nanmean(LifeTimeResde[3,i,:])))
            print(
                "SPECTRUM: \n Mean fractions in line {}: \n NADH={}, \n PYO={}, \n PVD={}, \n NADH_bound={}".format(
                    i,
                    np.nanmean(SpectralRes[0, i, :]),
                    np.nanmean(SpectralRes[1, i, :]),
                    np.nanmean(SpectralRes[2, i, :]),
                    np.nanmean(SpectralRes[3, i, :]),
                )
            )
            # print('SPECTRUM: \n Mean fractions in line {} from differential evolution: \n NADH={}, \n PYO={},\n PVD={}, \n NADH_bound={}'.format(i,np.nanmean(SpectralResde[0,i,:]),np.nanmean(SpectralResde[1,i,:]),np.nanmean(SpectralResde[2,i,:]),np.nanmean(SpectralResde[3,i,:])))
        return SpectralResde, LifeTimeResde, SpectralRes, LifeTimeRes, SpecLifeRes

    SpectralResde, LifeTimeResde, SpectralRes, LifeTimeRes, SpecLifeRes = fun(
        SpecSignal, SpecSpecies, LifeSignal, LifeSpecies, myob
    )

    s = np.zeros((256, 256))
    g = np.zeros((256, 256))
    for i in range(256):
        for j in range(256):
            s[i, j] = np.sum(LifeTimeRes[:, i, j] * LifeSpecies[0, 0, :])
            g[i, j] = np.sum(LifeTimeRes[:, i, j] * LifeSpecies[1, 0, :])

    H = np.histogram2d(s.flatten(), g.flatten(), bins=256, range=[[0, 1], [0, 1]])
    fig, ax = plt.subplots()
    plt.pcolormesh(H[1], H[2], np.log(H[0]), zorder=0)
    plt.title("Result")
    # plt.title(os.path.basename(self.lifetimefileName[fileIndex])+'\n LogPhasor')
    circ = plt.Circle((0.5, 0), radius=0.5, color="black", fill=False, zorder=10)
    plt.gcf().gca().add_artist(circ)

    ax = plt.scatter(LifeSpecies[1, 0, :], LifeSpecies[0, 0, :])
    circ = plt.Circle((0.5, 0), radius=0.5, color="black", fill=False, zorder=10)
    plt.gcf().gca().add_artist(circ)
    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])
    lifeSignal = LifetimeAsPureComponentSG([myob], har)
    plt.scatter(lifeSignal[1, 0, :], lifeSignal[0, 0, :], marker="x")

    plt.show()

    H = np.histogram2d(
        myob.lifetimePhasor[0][0].flatten()
        * (myob.lifetimeIm[0].flatten() > myob.lifetimethreshold),
        myob.lifetimePhasor[0][1].flatten(),
        bins=256,
        range=[[0, 1], [0, 1]],
    )
    fig, ax = plt.subplots()
    plt.pcolormesh(H[1], H[2], np.log(H[0]), zorder=0)
    plt.title("Data")
    # plt.title(os.path.basename(self.lifetimefileName[fileIndex])+'\n LogPhasor')
    circ = plt.Circle((0.5, 0), radius=0.5, color="black", fill=False, zorder=10)
    plt.gcf().gca().add_artist(circ)
    plt.show()

    imwrite(
        os.path.join(
            os.path.dirname(myob.lifetimefileName[0]),
            "SpectralResults_NADH_PYO_PVD_NADHb_firstHarmonic.tif",
        ),
        np.uint16(1000 * SpectralRes),
    )
    imwrite(
        os.path.join(
            os.path.dirname(myob.SpecfileName[0]),
            "LifetimeResults_NADH_PYO_PVD_NADHb_firstHarmonic.tif",
        ),
        np.uint16(1000 * LifeTimeRes),
    )
    # imwrite(os.path.join(os.path.dirname(myob.SpecfileName[0]),'SpectralLifeResults_NADH_PYO_PVD_NADHb_firstHarmonic.tif'), np.uint16(1000*SpecLifeRes), photometric='minisblack')
    return SpectralRes, LifeTimeRes
