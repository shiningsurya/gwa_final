# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ssg
plt.ion ()
############ PARAMETERS
fs=2048
T=1
t = np.arange (0, int(T*fs)) / fs
tsize = t.size
############ Mixed sine
print ("mixed sine")
NCOMP = 100
NSAMP = 500
### freqs
### In 300 to 600
freqs = 300 * np.random.rand (NSAMP, NCOMP)
freqs += 300
### amps in 0.5, 3.0
amps  = 2.5 * np.random.rand (NSAMP, NCOMP)
amps += 0.5
### output array
MS = np.zeros ((NSAMP, tsize))
### workloop
for i in range (NSAMP):
    for f in range(NCOMP):
        MS[i] += ( amps[i,f] * np.sin (2 * np.pi * freqs[i,f] * t) )
        MS[i] += np.random.randn (tsize,)
### end
### spectogram
spdict = {'fs':fs, 'nperseg':fs//9, 'noverlap':fs//10, 'nfft':fs//9}
###### size from ipy shell
MSP = np.zeros ((NSAMP, 114, 80))
for i in range (NSAMP):
    sf,st,sxx = ssg.spectrogram (MS[i], **spdict)
    MSP[i] = sxx
### end
np.save ("mixed_sine_timeseries_noise.npy", MS)
np.save ("mixed_sine_sxx_noise.npy", MSP)


