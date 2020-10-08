#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:34:57 2020

@author: Tara
"""

# make plots of zeiss LSM files containing spectral 32-channel images

# libraries
import os
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import numpy.fft as ft
import scipy.signal as sc

#%%

wavelen = np.linspace(410, 695, 32)  # make list of wavelengths for channels 1-32
os.chdir(
    "/Volumes/GoogleDrive/My Drive/PseudomonasImaging/Data/Spectral/TCEP_gradients"
)
image_stack = tiff.imread(os.listdir()[0])  # read in first file in directory
image_stack.shape  # shape of file
spec = np.sum(image_stack[0, 0, :, :, :], 2)
spec1 = np.sum(spec, 1)  # contains sum intensity for each channel (1-32)
# plt.plot(wavelen,spec1[0:32])
# plt.show()

# for loop

spec_df = pd.DataFrame()

spec_df = pd.DataFrame()
for file in os.listdir():
    image_stack = tiff.imread(file)
    spec = np.sum(image_stack[0, 0, :, :, :], 2)
    spec1 = np.sum(spec, 1)[0:32]  # 32 channels (last channel is Transmission)
    spec1 = pd.DataFrame(spec1)
    spec1.insert(0, "FileName", file)
    spec_df = spec1.append(spec_df)

# spec_df.FileName.astype("category")
spec_df.insert(0, "Sample", spec_df.FileName)
spec_df.Sample = spec_df.Sample.replace("TCEP.*", "", regex=True)
spec_df.insert(1, "TCEP_conc", spec_df.Sample)
spec_df.TCEP_conc = spec_df.TCEP_conc.replace(".*_", "", regex=True)
spec_df = spec_df.rename(columns={0: "Intensity"})

spec_df.insert(4, "Wavelength", pd.Series(wavelen))


fig = sns.relplot(
    x="Wavelength", y="Intensity", kind="line", hue="TCEP_conc", data=spec_df
)
plt.ylabel("Fluorescence emission")
plt.show(fig)
fig.savefig("tcep_gradients.pdf", dpi=300)


#%% ## look at other spectral files
os.chdir("D:/FLIM/PseudomonasSpectral/Pyocyanin-10-28-19/Tara_Experiments10_28_2019")

spec_df = pd.DataFrame()

spec_df = pd.DataFrame()
for file in os.listdir():
    if file.split(".")[1] == "lsm":
        image_stack = tiff.imread(file)
        spec = np.sum(image_stack[0, 0, :, :, :], 2)
        spec1 = np.sum(spec, 1)[0:32]  # 32 channels (last channel is Transmission)
        spec1 = pd.DataFrame(spec1)
        spec1.insert(0, "FileName", file)
        spec_df = spec1.append(spec_df)
spec_df.insert(0, "Sample", spec_df.FileName)
spec_df.Sample = spec_df.Sample.replace("_*.", regex=True)
spec_df = spec_df.rename(columns={0: "Intensity"})
spec_df.insert(3, "Wavelength", pd.Series(wavelen))

# remove eveyrthing after 2nd underscore
sample_names_df = spec_df.Sample.str.split(pat="_", n=2, expand=True)
sample_names_df = sample_names_df.rename(
    columns={0: "entry1", 1: "entry2", 2: "entry3"}
)
sample_names_df.insert(
    0, "NewSample", sample_names_df["entry1"] + sample_names_df["entry2"]
)
spec_df2 = pd.concat([spec_df, sample_names_df], axis=1)
spec_df_tcep = spec_df2[
    spec_df2.Sample.str.contains(pat="TCEP") & spec_df.Sample.str.contains(pat="PYO")
]
spec_df_tcep.insert(0, "TCEP", spec_df_tcep.entry2.str.split(pat="mm", expand=True)[0])
spec_df_tcep.TCEP.replace("umTCEP", "", regex=True)


spec_df_tcep = spec_df_tcep.loc[
    (spec_df_tcep["TCEP"] == "100umTCEP")
    | (spec_df_tcep["TCEP"] == "400umTCEP")
    | (spec_df_tcep["TCEP"] == "500umTCEP")
    | (spec_df_tcep["TCEP"] == "125")
    | (spec_df_tcep["TCEP"] == "2")
]

spec_df_tcep.insert(
    0, "containsmicromolar", spec_df_tcep["TCEP"].str.contains("um", regex=True)
)
spec_df_tcep["test"] = np.where(
    spec_df_tcep.containsmicromolar == True,
    spec_df_tcep.TCEP.replace("umTCEP", "", regex=True).astype("int") / 1000,
    spec_df_tcep.TCEP,
)
spec_df_tcep["TCEP_conc"] = spec_df_tcep["test"].astype("str")
spec_df_tcep["TCEP_conc"] = spec_df_tcep["TCEP_conc"].astype("category")

fig, ax = plt.subplots()
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e"]
g = sns.lineplot(
    x="Wavelength",
    y="Intensity",
    hue="TCEP_conc",
    hue_order=["125", "2", "0.5", "0.4", "0.1"],
    style="TCEP_conc",
    ci=None,
    data=spec_df_tcep,
    palette=flatui,
    legend=False,
)
plt.legend(loc="best", title="TCEP(mM)", labels=["125", "2", "0.5", "0.4", "0.1"])
plt.show(g)


fig.savefig("C:/Users/Pseudomonas/Desktop/reduced_pyo.pdf", dpi=300)

#%% ## EC reduced pyocyanin
file = "D:/FLIM/PseudomonasSpectral/03-13-20_DIVER_880_ASMplates/TaraSpectral_3_13_2020/electrochemicallyred_PYO821uM_10p_52fov_spectral_20xA_740nm_410-695_9nmstep_4usdwell_bidi16bitlinesum.lsm"
image_stack = tiff.imread(file)
spec = np.sum(image_stack[0, 0, :, :, :], 2)
spec1 = np.sum(spec, 1)[0:32]  # 32 channels (last channel is Transmission)
spec1 = pd.DataFrame(spec1)
spec1.insert(0, "FileName", file)
spec_ec = spec1
spec_ec.insert(0, "Sample", spec_ec.FileName)
spec_ec.Sample = ["PVD" for i in range(32)]
spec_ec.insert(3, "Wavelength", pd.Series(wavelen))
spec_ec = spec_ec.rename(columns={0: "Intensity"})


fig, ax = plt.subplots()
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e"]
g = sns.lineplot(x="Wavelength", y="Intensity", ci=None, data=spec_ec, legend=False)


#%% ## pyoverdine
file = "D:/FLIM/PseudomonasSpectral/FILES_FOR_ANALYSIS/pure_species_spectral/060719_pyoverdine_LSM880_40xW_740nm_12p_410nm_695nm_9nmres_fov39um_PVD40uM.lsm"
image_stack = tiff.imread(file)
spec = np.sum(image_stack[0, 0, :, :, :], 2)
spec1 = np.sum(spec, 1)[0:32]  # 32 channels (last channel is Transmission)
spec1 = pd.DataFrame(spec1)
spec1.insert(0, "FileName", file)
spec_pvd = spec1
spec_pvd.insert(0, "Sample", spec_pvd.FileName)
spec_pvd.Sample = ["PVD" for i in range(32)]
spec_pvd.insert(3, "Wavelength", pd.Series(wavelen))
spec_pvd = spec_pvd.rename(columns={0: "Intensity"})


fig, ax = plt.subplots()
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e"]
g = sns.lineplot(x="Wavelength", y="Intensity", ci=None, data=spec_pvd, legend=False)


### hydroxy phenazine


fig, ax = plt.subplots()
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e"]
g = sns.lineplot(x="Wavelength", y="Intensity", ci=None, data=spec_pvd, legend=False)


#%%
# sp=np.sum(spec,1)
har = 1
t = np.linspace(0, 1, 32)
si = np.sin(har * 2 * np.pi * t)
co = np.cos(har * 2 * np.pi * t)

bigsi = np.tile(si, (256, 256))
bigco = np.tile(co, (256, 256))
bigsi = bigsi.swapaxes(0, 1).reshape((32, 256, 256), order="F")
bigco = bigco.swapaxes(0, 1).reshape((32, 256, 256), order="F")

# stack=np.sum(image_stack[0,:,:,:,:],0);

# stack.ravel('F')

# si=np.repeat()
# s,g=specphasor(stack,bigsi,bigco)


def specphasor(sp, si, co):
    s = np.sum(si * sp, 0)
    s = s / (np.sum(sp, 0) + np.finfo(float).eps)
    g = np.sum(co * sp, 0)
    g = g / (np.sum(sp, 0) + np.finfo(float).eps)
    return s, g


def PhModu(s, g):
    modu = s ** 2 + g ** 2
    ph = np.arctan2(s, g)
    return ph, modu


def MeanVar(ph, mod, lambdamin, lambdamax):
    mean = lambdamin + (ph) / (2 * np.pi) * (lambdamax - lambdamin)
    var = 0 + (mod - 1) / (-1) * (lambdamax - lambdamin)
    return mean, var


wavemin = 400
wavemax = 700
har = 1
num = 3000
t = np.linspace(0, 1, num)
waveX = np.linspace(400, 700, num)
si = np.sin(har * 2 * np.pi * t)
co = np.cos(har * 2 * np.pi * t)
sig = 1
mean = 300
imax = 50
meanwave = np.zeros(imax)
mn = np.zeros(imax)
var = np.zeros(imax)
truevar = np.zeros(imax)
for j in range(imax):
    mean = 300
    for i in range(imax):
        mn[i] = mean
        sp = np.exp(-((waveX - mean) ** 2) / ((sig) ** 2))
        plt.plot(waveX, sp)
        s, g = specphasor(sp, si, co)
        ph, mod = PhModu(s, g)
        meanwave[i], var[i] = MeanVar(ph, mod, wavemin, wavemax)
        print("true mean = {}, calculated mean = {}".format(mean, meanwave[i]))
        print("true var = {}, calculated variance = {}".format(sig, var[i]))
        mean += 10
    sig += 10
    truevar[j] = sig
    plt.figure()
    plt.plot(mn, meanwave)
plt.plot(truevar, var)


print(meanwave, var)

# plt.ion()
import matplotlib.cm as cm

c = cm.rainbow(np.linspace(0, 2, 400))
fig, ax = plt.subplots(2, 1)

for j in range(1, 20, 2):
    plt.gcf().gca()
    sp = sc.gaussian(32, j)
    ct = 0
    for i in range(0, 400, 2):
        ax[1].cla()
        a = np.abs(
            ft.ifft(ft.fft(sp) * ft.fftshift(np.exp(1j * (np.linspace(0, i, 32)))))
        )
        s, g = specphasor(a, si, co)
        ax[1].plot(a)
        ax[0].scatter(s, g, marker=".", color=c[ct])
        ct += 1
        # fig.show()
