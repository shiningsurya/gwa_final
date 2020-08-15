# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ssg
import glob
plt.ion ()
############ PARAMETERS
fs=2048
T=1
t = np.arange (0, int(T*fs)) / fs
tsize = t.size
############ ccsn
print ("ccsn")
FF = glob.glob ("ccsn_signals/*.dat")
FL = len(FF)
NITER = 15
NSAMP = FL * NITER
### splitter
def spliter (i):
    fi = FF[i]
    t,x = np.genfromtxt (fi, unpack=True)
    mm = x.argmin()
    II = mm - 410
    JJ = mm + 1638
    if JJ >= t.size:
        print ("JJ>size")
        ij = JJ - t.size
        II = II - ij
    if II < 0:
        print ("II<0")
    return t[II:JJ],x[II:JJ]
### amps in 0.5, 3.0
amps  = 2.5 * np.random.rand (FL, NITER)
amps += 0.5
### output array
### workloop
fs=2048
spdict = {'fs':fs, 'nperseg':fs//9, 'noverlap':fs//10, 'nfft':fs//9}
MS = np.zeros ((NSAMP, 114, 80))
I = 0
for i in range (FL):
    t,x = spliter (i)
    x -= x.mean()
    x /= x.std()
    print (x.shape)
    for j in range (NITER):
        ix = x * amps[i,j]
        ix += np.random.randn (*x.shape)
        sf,st,sxx = ssg.spectrogram (ix, **spdict)
        MS[I] = sxx
        I = I + 1
print (I, MS.shape)
### end
np.save ("ccsn_sxx_noise.npy", MS)


