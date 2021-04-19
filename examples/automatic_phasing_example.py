#%%
import matplotlib.pyplot as plt
from nmr_tools import processing, proc_base
import numpy as np

plt.rcParams['figure.dpi'] = 200

datapath = '/home/m_buss13/ownCloud/nmr_data/development/195Pt_Pt-Mix_WCPMG/1/pdata/1'
datapath_mc = '/home/m_buss13/ownCloud/nmr_data/development/195Pt_Pt-Mix_WCPMG/1/pdata/11'

ppm_scale_mc, hz_scale_mc, data_mc = processing.read_brukerproc(datapath_mc)
ppm_scale, hz_scale, data, dic = processing.read_brukerfid(datapath, dict=True)

data, phase = processing.automatic_phasecorrection2(data, bnds=((-360, 360), (-60000, -50000), (-15000, -10000)), Ns=10, verb=True, loss_func='mse')
# print(phase)


plt.figure()
plt.plot(ppm_scale_mc, data_mc/max(data_mc), c='k', lw=1.0, label='Magnitude')
plt.plot(ppm_scale, data.real/max(data.real), c='r', lw=1.0, label='p0={:.0f}°, p1={:.0f}, p2={:.0f}°'.format(phase[0],phase[1],phase[2]))
# plt.plot(ppm_scale, data.imag/max(data.imag), c='b', lw=1.0, label='p0={:.0f}°, p1={:.0f}, p2={:.0f}°'.format(phase[0],phase[1],phase[2]))
plt.yticks([])
plt.xlim(4000, -6000)
# plt.xlim(-4500, -5000)
plt.legend()


# p0 = 0.0 * np.pi / 180.
# p1 = 0.0 * np.pi / 180.
# p2 = 0.0 * np.pi / 180.
# size = data.shape[-1]
# # apod = p0 + (p1 * ((np.arange(size) / size)-0.5)) + (p2 * np.power((np.arange(size) / size)-0.5, 2))
# apod = np.exp(1.0j * (p0 + (p1 * ((np.arange(size) / size)-0.5)) + (p2 * np.power((np.arange(size) / size)-0.5, 2))))

# plt.figure()
# plt.plot(np.arange(size), apod.real, c='k', lw=1.0, label='Magnitude')
# plt.plot(np.arange(size), apod.imag, c='r', lw=1.0, label='Magnitude')
