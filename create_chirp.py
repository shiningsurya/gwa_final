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
print ("chirp")
NSAMP = 500
F0 = 200 + ( 50*np.random.rand (NSAMP,) )
F1 = 900 + ( 100*np.random.rand (NSAMP,) )
### amps in 0.5, 3.0
amps  = 2.5 * np.random.rand (NSAMP,)
amps += 0.5
###
meth = ['linear', 'quadratic', 'logarithmic', 'hyperbolic']
imeth = np.random.randint (0, 4, size=(NSAMP,))
### output array
MS = np.zeros ((NSAMP, tsize))
### workloop
for i in range (NSAMP):
    MS[i] = amps[i] * ssg.chirp (t, f0=F0[i], t1=T, f1=F1[i], method=meth[imeth[i]])
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
if True:
    np.save ("chirp_timeseries_noise.npy", MS)
    np.save ("chirp_sxx_noise.npy", MSP)


