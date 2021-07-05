#%%
# This is an example for a simple echo linebroadening of a Bruker FID, without further processing
import matplotlib.pyplot as plt
from nmrglue import process
from numpy.core.fromnumeric import size
from cpmg_tools import processing
import numpy as np
import os
from pathlib import Path
plt.rcParams['figure.dpi'] = 200

# Set path to dataset
# datapath = r'example_data\195Pt_PtMix_MAS_WCPMG_stepped\2999\pdata\1'
datapath = r'example_data/195Pt_PtMix_MAS_WCPMG_stepped/2999/pdata/1'

# Read Bruker FID
data, _ = processing.read_brukerfid(datapath)
# data, _ = processing.read_brukerproc(datapath)

# Apply linebroadening
data_lb, window = processing.linebroadening(data,
                                            # lb_variant='scipy_exponential',
                                            # lb_variant='scipy_chebwin',
                                            # lb_variant='scipy_taylor',
                                            # lb_variant='scipy_parzen',
                                            # lb_variant='scipy_nuttall',
                                    lb_variant='scipy_general_hamming',
                                            # lb_variant='scipy_blackmanharris',
                                            # lb_variant='scipy_kaiser',
                                            # lb_variant='scipy_dpss',
                                            # lb_variant='compressed_wurst',
                                            # lb_variant='shifted_wurst',
                                            # lb_variant='gaussian',
                                            # lb_variant='scipy',
                                            # lb_const=0.24,
                                            # lb_n=2,
                                            # **{'nbar':2, 'sll':50}
                                            # **{'NW':1.0}
                                            # **{'beta':4}
                                            **{'alpha':0.62}
                                            # **{'at':50}
                                            # **{'tau':600}
                                            )

# Plotting
plt.figure()

plt.plot(processing.interleave_complex(data.real, data.imag), c='k')
plt.plot(processing.interleave_complex(data_lb.real, data_lb.imag), c='grey')
plt.plot(np.linspace(0, len(data)*2, num=size(data)), window*max(processing.interleave_complex(data.real, data.imag)), c='r', label='1')
plt.yticks([])

plt.show()

#%%
# This is an example of a full linebroadening processing pipeline, comparing two different window functions and calculating the S/N
import matplotlib.pyplot as plt
from cpmg_tools import processing, proc_base
import numpy as np
import matplotlib.gridspec as gridspec

plt.rcParams['figure.dpi'] = 200

# Read magnitude data
data_mc, ppm_scale_mc, hz_scale_mc = processing.read_brukerproc(r'example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/1')

# Read bruker FID
data, timescale, dic = processing.read_brukerfid(r'example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/1', dict=True)
data2, timescale2, dic2 = processing.read_brukerfid(r'example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/1', dict=True)

# Save data for comaprison
data_before_lb = data
# Apply linebroadening
data, window = processing.linebroadening(data,
                                        lb_variant='shifted_wurst',
                                        lb_const=0.24,
                                        lb_n=2
                                        )


fid_before = data_before_lb
fid_after = data
# Fouriertransform, zerofilling and phasing
data = processing.fft(data, si=32768, mc=False, phase=[847, -268718])

# Generate new scales
ppm_scale, hz_scale = processing.get_scale(data, dic)

# Apply linebroadening
data2, window2 = processing.linebroadening(data2,
                                        lb_variant='scipy_general_hamming',
                                        **{'alpha':0.8}
                                        )

# Fouriertransform, zerofilling and phasing
data2 = processing.fft(data2, si=32768, mc=False, phase=[847, -268718])

# Generate new scales
ppm_scale2, hz_scale2 = processing.get_scale(data2, dic2)

# Process comparison data equally
data_before_lb = processing.fft(data_before_lb, si=32768, mc=False, phase=[847, -268718])

# Calculation of S/N
sino_before_lb = processing.signaltonoise_region(data_before_lb.real, noisepts=(1000, 15000))
sino_after_method1 = processing.signaltonoise_region(data.real, noisepts=(1000, 15000))
sino_after_method2 = processing.signaltonoise_region(data2.real, noisepts=(1000, 15000))

# Plotting
fig = plt.figure(figsize=(7, 7), facecolor='#f4f4f4')
spec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
spec.update(wspace=0.0, hspace=0.0)
f1_ax1 = fig.add_subplot(spec[1:, 0])
f1_ax2 = fig.add_subplot(spec[0, 0])

f1_ax2.plot(processing.interleave_complex(fid_before.real, fid_before.imag), c='k')
f1_ax2.plot(processing.interleave_complex(fid_after.real, fid_after.imag), c='grey')
f1_ax2.plot(np.linspace(0, len(fid_before)*2, num=len(fid_before)), window*max(abs(processing.interleave_complex(fid_before.real, fid_before.imag))), c='r')
f1_ax2.plot(np.linspace(0, len(fid_before)*2, num=len(fid_before)), window2*max(abs(processing.interleave_complex(fid_before.real, fid_before.imag))), c='b')
f1_ax2.set_yticks([])
f1_ax2.set_xticks([])

f1_ax1.plot(ppm_scale, data_before_lb.real/max(abs(data_before_lb.real)), c='dimgrey', lw=1, label='No LB - {:.4f}'.format(sino_before_lb))
f1_ax1.plot(ppm_scale, data.real/max(abs(data.real)), c='r', lw=1, label='shifted WURST - {:.4f}'.format(sino_after_method1))
f1_ax1.plot(ppm_scale2, data2.real/max(abs(data2.real)), c='b', lw=1, label='Hamming - {:.4f}'.format(sino_after_method2))

f1_ax1.set_xlim(1000, -1000)
f1_ax1.set_xlabel('$^{207}$Pb / ppm')
f1_ax1.legend()
f1_ax1.set_yticks([])

plt.show()

#%%
# This is an example for the experimental multiwindow linebroadening.
import matplotlib.pyplot as plt
from cpmg_tools import processing, proc_base
import numpy as np
import matplotlib.gridspec as gridspec

plt.rcParams['figure.dpi'] = 200

# Read magnitude data
data_mc, ppm_scale_mc, hz_scale_mc = processing.read_brukerproc(r'example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/11')

# Read bruker FID
data, timescale, dic = processing.split_echotrain(r'example_data/207Pb_PbZrO3_MAS_WCPMG/2/pdata/1',
                                                  dw=0.2, echolength=560, blankinglength=80, numecho=48)


# Save data for comparison
data_before_lb = data
# Apply linebroadening
data, window = processing.linebroadening(data,
                                        lb_variant='compressed_wurst',
                                        lb_const=0.01,
                                        num_windows=7,
                                        lb_n=10,
                                        )


fid_before = data_before_lb
fid_after = data

# Fouriertransform, zerofilling and phasing
data = processing.fft(data, si=32768, mc=True)

# Generate new scales
ppm_scale, hz_scale = processing.get_scale(data, dic)

# Process comparison data equally
data_before_lb = processing.fft(data_before_lb, si=32768, mc=True)

# Calculation of S/N
sino_before_lb = processing.signaltonoise_region(data_before_lb.real, noisepts=(1000, 15000))
sino_after_method1 = processing.signaltonoise_region(data.real, noisepts=(1000, 15000))

# Plotting
fig = plt.figure(figsize=(7, 7), facecolor='#f4f4f4')
spec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
spec.update(wspace=0.0, hspace=0.0)
f1_ax1 = fig.add_subplot(spec[1:, 0])
f1_ax2 = fig.add_subplot(spec[0, 0])

f1_ax2.plot(processing.interleave_complex(fid_before.real, fid_before.imag), c='k')
f1_ax2.plot(processing.interleave_complex(fid_after.real, fid_after.imag), c='grey')
f1_ax2.plot(np.linspace(0, len(fid_before)*2, num=len(fid_before)), window*max(abs(processing.interleave_complex(fid_before.real, fid_before.imag))), c='r')
f1_ax2.set_yticks([])
f1_ax2.set_xticks([])

f1_ax1.plot(ppm_scale_mc, data_mc.real/max(abs(data_mc.real)), c='dimgrey', lw=1, label='No LB - {:.4f}'.format(sino_before_lb))
f1_ax1.plot(ppm_scale, abs(data)/max(abs(data.real)), c='r', lw=1, label='shifted WURST - {:.4f}'.format(sino_after_method1))

f1_ax1.set_xlim(1000, -1000)
f1_ax1.set_xlabel('$^{207}$Pb / ppm')
f1_ax1.legend()
f1_ax1.set_yticks([])

plt.show()
