#%%
import matplotlib.pyplot as plt
from nmr_tools import processing

plt.rcParams['figure.dpi'] = 200

datapath = '/home/m_buss13/ownCloud/nmr_data/development/195Pt_PtMix_stepped/2999/pdata/1'

ppm_scale, hz_scale, data, dic = processing.read_brukerfid(datapath, dict=True)

data_lb, window = processing.linebroadening(data, lb_variant='hamming', lb_const=0.2, lb_n=2)


plt.figure()
plt.plot(data.real, c='k')
plt.plot(data.imag, c='k')
plt.plot(data_lb.real, c='grey')
plt.plot(data_lb.imag, c='grey')
plt.plot(window*data_lb.max(), c='r')
plt.yticks([])
