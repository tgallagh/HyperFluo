#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:44:13 2020

@author: SimonL
"""


#%% baby t
import numpy as np
import lfdfiles as lfd
import matplotlib.pyplot as plt
import os


def phasorit(tau, omega, harmonics):
    a = np.zeros((2, harmonics, tau.size))
    #    for j in range(0,tau.size)
    for i in range(1, harmonics + 1):
        a[0, i - 1, :] = 1 / (1 + ((i * omega) ** 2) * (tau ** 2))
        a[1, i - 1, :] = (i * omega * tau) / (1 + (((i * omega) ** 2) * (tau ** 2)))
    return a


#


def makeandplotphasor(ax):
    i = 0
    sz = 3000
    tau = np.linspace(1e-10, 5e-7, sz)
    species = np.zeros((2, sz))
    for t in tau:
        species[:, i] = np.squeeze(phasorit(t, omega, 1))
        i += 1
    ax.plot(
        species[0, :],
        species[1, :],
        "k-",
    )


omega = 2 * np.pi * 80e6
binning = 256
xedges = np.linspace(0, 1, binning)
yedges = np.linspace(0, 1, binning)
fig, ax = plt.subplots()

d = np.ndarray((binning - 1, binning - 1))
# myfile='031320_PA14chunk_nocoverslip_fov55_p7_780nm_20xA_CH1_CH2000$CC0S_ch1_h1_h2.R64';
mydir = os.listdir()
for file in mydir:
    if os.path.splitext(file)[1].lower() == ".r64":
        with lfd.SimfcsR64(file) as f:
            x = np.reshape(
                f.asarray()[2, :, :] * np.sin(np.deg2rad(f.asarray()[1, :, :])),
                256 * 256,
            )
            y = np.reshape(
                f.asarray()[2, :, :] * np.cos(np.deg2rad(f.asarray()[1, :, :])),
                256 * 256,
            )
            H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
            H = H.T
            d += H
            #            fig, ax = plt.subplots()
            ax.scatter(np.mean(x), np.mean(y))
            #            ax.imshow(H, interpolation='nearest', origin='low',
            #                       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            ax.set_title(os.path.splitext(file)[0][:35])
            makeandplotphasor(ax)


#
# isize=256
# im=np.zeros((256,256))
# with lfd.SimfcsR64(myfile) as f:
#    for i in range(isize-1):
#        for j in range(isize-1):
#            im[255*int(np.rint(f.asarray()[2,i,j]*np.sin(np.deg2rad(f.asarray()[1,i,j])))), 255*int(np.rint(f.asarray()[2,i,j]*np.cos(np.deg2rad(f.asarray()[1,i,j]))))]+=1
#        print(i)


plt.show()
