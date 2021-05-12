#%%
import matplotlib.pyplot as plt
from nmr_tools import processing, proc_base
import numpy as np
import matplotlib.gridspec as gridspec

plt.rcParams['figure.dpi'] = 160

# Read magnitude data
data_mc, ppm_scale_mc, hz_scale_mc = processing.read_brukerproc('/home/m_buss13/ownCloud/git/nmr_tools/examples/example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/11')

# Read bruker FID
data, timescale, dic = processing.read_brukerfid('/home/m_buss13/ownCloud/git/nmr_tools/examples/example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/1', dict=True)
fid_before = data

# Apply SVD denoising
data = processing.denoise(data, k_thres=0, max_err=10)
fid_after_sg = data

# Apply linebroadening
data, window = processing.linebroadening(data, lb_variant='scipy_general_hamming', **{'alpha':0.62})
fid_after_sglb = data

# Fouriertransform, zerofilling and phasing
data_before = proc_base.zf_size(fid_before, 32768)    # zero fill to 32768 points
data_before = proc_base.fft(proc_base.rev(data_before))               # Fourier transform
data_before = proc_base.ps(data_before, p0=847, p1=-268718)
# Fouriertransform, zerofilling and phasing
data_after_sg = proc_base.zf_size(fid_after_sg, 32768)    # zero fill to 32768 points
data_after_sg = proc_base.fft(proc_base.rev(data_after_sg))               # Fourier transform
data_after_sg = proc_base.ps(data_after_sg, p0=847, p1=-268718)
# Fouriertransform, zerofilling and phasing
data_after_sglb = proc_base.zf_size(fid_after_sglb, 32768)    # zero fill to 32768 points
data_after_sglb = proc_base.fft(proc_base.rev(data_after_sglb))               # Fourier transform
data_after_sglb = proc_base.ps(data_after_sglb, p0=847, p1=-268718)
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

f1_ax1.plot(ppm_scale, data_before.real/max(abs(data_before.real)), c='k', lw=1, label='Only FFT+ZF - SNR = {:.0f}'.format(snr_fft))
f1_ax1.plot(ppm_scale, data_after_sg.real/max(abs(data_after_sg.real)), c='grey', lw=1, label='After SVD - SNR = {:.0f}'.format(snr_svd))
f1_ax1.plot(ppm_scale, data_after_sglb.real/max(abs(data_after_sglb.real)), c='firebrick', lw=1, label='After SVD+LB - SNR = {:.0f}'.format(snr_svdlb))
f1_ax1.set_xlim(-1500, 1500)
f1_ax1.legend()
f1_ax1.set_yticks([])

plt.savefig('/home/m_buss13/ownCloud/svd_test.png', dpi=600)
plt.show()
