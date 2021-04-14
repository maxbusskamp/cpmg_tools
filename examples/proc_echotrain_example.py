#%%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from plot_module import plot_module
from nmr_tools import processing, proc_base


plt.rcParams['figure.dpi'] = 200

ppm_scale, hz_scale, data = processing.split_echotrain(datapath='/home/m_buss13/ownCloud/nmr_data/neo500/2020_08_23_double_echo_119Sn_4mm_10kHz_cpmg/1100/pdata/1',
                                                       dw=0.5, echolength=300, blankinglength=300, numecho=50)

# process the spectrum
data_sum_proc = proc_base.zf_size(data, 32768)    # zero fill to 32768 points
data_sum_proc = proc_base.fft(data_sum_proc)               # Fourier transform
ppm_scale = np.linspace(ppm_scale[-1], ppm_scale[0], len(data_sum_proc))

# PLOTTING PART
bruker_proc_abs = plot_module.plot_1D("/home/m_buss13/ownCloud/nmr_data/neo500/2020_08_23_double_echo_119Sn_4mm_10kHz_cpmg/1101/pdata/11",
                            val_only=True,
                            )

fig = plt.figure(figsize=(6.5, 4), facecolor='#f4f4f4')
spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
spec.update(wspace=0.0, hspace=0.0)
f1_ax1 = fig.add_subplot(spec[0, 0])

f1_ax1.plot(ppm_scale, abs(data_sum_proc), lw=1.5, c='k', label='Abs. Coadd Python')
f1_ax1.plot(bruker_proc_abs[0], bruker_proc_abs[1], lw=1.0, c='g', label='Abs. Coadd AU')

f1_ax1.set_xlim(600,-1300)
f1_ax1.set_yticks([])
f1_ax1.legend(fontsize=6, facecolor='#f4f4f4')

plt.show()
