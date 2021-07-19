#%%
# This is an example for the use of SVD denoising as described in DOI: 10.1080/05704928.2018.1523183
import matplotlib.pyplot as plt
from cpmg_tools import processing, proc_base
import numpy as np
import matplotlib.gridspec as gridspec

plt.rcParams['figure.dpi'] = 160

# Read Bruker FID
data, timescale, dic = processing.read_brukerfid(r'example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/1', dict=True)
# fid_before = data
fid_before, _ = processing.linebroadening(data, lb_variant='scipy_general_hamming', **{'alpha':0.6})

# Apply SVD denoising
data = processing.denoise(data, k_thres=0, max_err=10)
fid_after_sg = data

# Apply linebroadening
data, window = processing.linebroadening(data, lb_variant='scipy_general_hamming', **{'alpha':0.6})
fid_after_sglb = data

# Fouriertransform, zerofilling and phasing
data_before = processing.fft(fid_before, si=32768, mc=False, phase=[847, -268718])

# Fouriertransform, zerofilling and phasing
data_after_sg = processing.fft(fid_after_sg, si=32768, mc=False, phase=[847, -268718])

# Fouriertransform, zerofilling and phasing
data_after_sglb = processing.fft(fid_after_sglb, si=32768, mc=False, phase=[847, -268718])

# Generate new scales
ppm_scale, hz_scale = processing.get_scale(data_before, dic)

# Calculate signal to noise
# print('Data before LB:')
snr_fft = processing.signaltonoise_region(data_before.real, noisepts=(1000, 15000))
# print('Data after SG:')
snr_svd = processing.signaltonoise_region(data_after_sg.real, noisepts=(1000, 15000))
# print('Data after SG+LB:')
snr_svdlb = processing.signaltonoise_region(data_after_sglb.real, noisepts=(1000, 15000))

# Plotting
fig = plt.figure(figsize=(7, 7), facecolor='#f4f4f4')
spec = gridspec.GridSpec(ncols=1, nrows=3, figure=fig)
spec.update(wspace=0.0, hspace=0.0)
f1_ax1 = fig.add_subplot(spec[1:, 0])
f1_ax2 = fig.add_subplot(spec[0, 0])

f1_ax2.plot(processing.interleave_complex(fid_before.real, fid_before.imag), c='k')
f1_ax2.plot(processing.interleave_complex(fid_after_sg.real, fid_after_sg.imag), c='grey')
f1_ax2.plot(processing.interleave_complex(fid_after_sglb.real, fid_after_sglb.imag), c='firebrick')
f1_ax2.plot(np.linspace(0, len(fid_before)*2, num=len(fid_before)), window*max(abs(processing.interleave_complex(fid_before.real, fid_before.imag))), c='firebrick')
f1_ax2.set_yticks([])
f1_ax2.set_xticks([])

f1_ax1.plot(ppm_scale, abs(data_before)/max(abs(data_before)), c='k', lw=1, label='FFT+LB - SNR = {:.0f}'.format(snr_fft))
f1_ax1.plot(ppm_scale, abs(data_after_sg)/max(abs(data_after_sg)), c='grey', lw=1, label='After SVD - SNR = {:.0f}'.format(snr_svd))
f1_ax1.plot(ppm_scale, abs(data_after_sglb)/max(abs(data_after_sglb)), c='firebrick', lw=1, label='After SVD+LB - SNR = {:.0f}'.format(snr_svdlb))
f1_ax1.set_xlim(1000, -1000)
f1_ax1.legend()
f1_ax1.set_yticks([])
f1_ax1.set_xlabel('$^{207}$Pb / ppm')

plt.show()
