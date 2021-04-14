#%%
import matplotlib.pyplot as plt
from nmr_tools import processing

plt.rcParams['figure.dpi'] = 200

datapath = '/home/m_buss13/ownCloud/nmr_data/development/207Pb_wcpmg_mas/1/pdata/1'

ppm_scale_mc, hz_scale_mc, data_mc = processing.read_brukerproc(datapath)
ppm_scale, hz_scale, data, dic = processing.read_brukerfid(datapath, dict=True)

data, phase = processing.automatic_phasecorrection(data, bnds=((-360, 360), (0, 200000)), SI=32768, Ns=50, verb=False, loss_func='mse')
print(phase)



ppm_scale, hz_scale = processing.get_scale(data, dic)

plt.figure()
plt.plot(ppm_scale_mc, data_mc, c='k', label='Magnitude')
plt.plot(ppm_scale, data, c='r', label='p0={:.0f}°, p1={:.0f}°'.format(phase[0],phase[1]))
plt.yticks([])
plt.xlim(-1000, 1000)
plt.legend()
