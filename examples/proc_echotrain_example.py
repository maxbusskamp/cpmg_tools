#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nmr_tools import processing, proc_base


plt.rcParams['figure.dpi'] = 200

# Split FID echotrain and sum all echos
data, _, dic = processing.split_echotrain(datapath='/home/m_buss13/ownCloud/git/nmr_tools/examples/example_data/119Sn_SnO2_double_echo_cpmg/1100/pdata/1',
                                  dw=0.5, echolength=300, blankinglength=300, numecho=50, dict=True)

# Process the resulting echo FID
data_sum_proc = proc_base.zf_size(data, 32768)    # zero fill to 32768 points
data_sum_proc = proc_base.fft(proc_base.rev(data_sum_proc))               # Fourier transform
ppm_scale, hz_scale = processing.get_scale(data_sum_proc, dic)

# Read comparison spikelet spectrum
data_mc, ppm_scale_mc, hz_scale_mc = processing.read_brukerproc('/home/m_buss13/ownCloud/git/nmr_tools/examples/example_data/119Sn_SnO2_double_echo_cpmg/1101/pdata/11')

# Plotting
fig = plt.figure(figsize=(6.5, 4), facecolor='#f4f4f4')
spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
spec.update(wspace=0.0, hspace=0.0)
f1_ax1 = fig.add_subplot(spec[0, 0])

f1_ax1.plot(ppm_scale, abs(data_sum_proc), lw=1.5, c='k', label='Abs. Coadd Python')
f1_ax1.plot(ppm_scale_mc, data_mc, lw=1.0, ls='--', c='g', label='Abs. Coadd AU')

f1_ax1.set_xlim(600,-1300)
f1_ax1.set_yticks([])
f1_ax1.legend(fontsize=6, facecolor='#f4f4f4')

plt.show()
