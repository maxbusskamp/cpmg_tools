#%%
# This is an example for the proeccing of an cpmg echotrain in python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cpmg_tools import processing
plt.rcParams['figure.dpi'] = 200


# Split FID echotrain and sum all echos
# Choose one of the possible options below by un/commenting one of the lines starting with dw=0.5,...
data, _, dic = processing.split_echotrain(datapath=r'example_data/119Sn_SnO2_double_echo_cpmg/1100/pdata/1',
                                  # dw=0.5, echolength=300, blankinglength=300, numecho=52, echotop=219)
                                #   dw=0.5, echolength=300, blankinglength=300, numecho=52, echotop=None)
                                #   dw=0.5, echolength=600, blankinglength=None, numecho=52, echotop=300)
                                #   dw=0.5, echolength=600, blankinglength=None, numecho=52, echotop=None)
                                #   dw=0.5, echolength=300, blankinglength=300, numecho=None, echotop=219)
                                #   dw=0.5, echolength=300, blankinglength=300, numecho=None, echotop=None)
                                #   dw=0.5, echolength=600, blankinglength=None, numecho=None, echotop=300)
                                  dw=0.5, echolength=600, blankinglength=None, numecho=None, echotop=None)

# Process the resulting echo FID
data_ft, ppm_scale, hz_scale = processing.fft(data, si=65536, dic=dic, mc=True)

# Read comparison spikelet spectrum
data_mc, ppm_scale_mc, hz_scale_mc = processing.read_brukerproc(r'example_data/119Sn_SnO2_double_echo_cpmg/1101/pdata/11')

# Plotting
fig = plt.figure(figsize=(6.5, 4), facecolor='#f4f4f4')
spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
spec.update(wspace=0.0, hspace=0.0)
f1_ax1 = fig.add_subplot(spec[0, 0])

f1_ax1.plot(ppm_scale, data_ft, lw=1.5, c='k', label='Abs. Coadd Python')
f1_ax1.plot(ppm_scale_mc, data_mc, lw=1.0, ls='--', c='g', label='Abs. Coadd AU')

f1_ax1.set_xlim(600,-1300)
f1_ax1.set_yticks([])
f1_ax1.legend(fontsize=6, facecolor='#f4f4f4')

plt.show()
