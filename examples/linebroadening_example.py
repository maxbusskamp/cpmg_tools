#%%
import matplotlib.pyplot as plt
from nmr_tools import processing

plt.rcParams['figure.dpi'] = 200

datapath = '/home/m_buss13/ownCloud/nmr_data/development/195Pt_PtMix_stepped/2999/pdata/1'

data, _ = processing.read_brukerfid(datapath)

data_lb, window = processing.linebroadening(data, lb_variant='hamming', lb_const=0.2, lb_n=2)


plt.figure()
plt.plot(data.real, c='k')
plt.plot(data.imag, c='k')
plt.plot(data_lb.real, c='grey')
plt.plot(data_lb.imag, c='grey')
plt.plot(window*data_lb.max(), c='r')
plt.yticks([])

#%%
import matplotlib.pyplot as plt
from nmr_tools import processing, proc_base
import time
plt.rcParams['figure.dpi'] = 200


data_mc, ppm_scale_mc, hz_scale_mc = processing.read_brukerproc('/home/m_buss13/ownCloud/git/nmr_tools/examples/example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/11')

data, timescale, dic = processing.read_brukerfid('/home/m_buss13/ownCloud/git/nmr_tools/examples/example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/1', dict=True)
data_before_lb = data
data, window = processing.linebroadening(data, lb_variant='hamming', lb_const=0.54, lb_n=2)

data = proc_base.zf_size(data, 32768)    # zero fill to 32768 points
data = proc_base.fft(proc_base.rev(data))               # Fourier transform
data = proc_base.ps(data, p0=847, p1=-268718)
ppm_scale, hz_scale = processing.get_scale(data, dic)

data_before_lb = proc_base.zf_size(data_before_lb, 32768)    # zero fill to 32768 points
data_before_lb = proc_base.fft(proc_base.rev(data_before_lb))               # Fourier transform
data_before_lb = proc_base.ps(data_before_lb, p0=847, p1=-268718)

plt.figure()
# plt.plot(ppm_scale_mc, data_mc/max(data_mc), c='r', lw=1.0, label='Magnitude')
plt.plot(ppm_scale, data_before_lb.real/max(abs(data_before_lb.real)), c='k', lw=1.0, label='No LB')
plt.plot(ppm_scale, data.real/max(abs(data.real)), c='r', lw=1.0, label='p0=847°, p1=-268718°')

# plt.xlim(1000, -1000)
# plt.xlim(1000, -5500)
# plt.xlim(3000, -5500)
# plt.xlim(-1000, -1100)
plt.xlim(1500, -1500)
plt.legend()
plt.yticks([])
# plt.savefig('automatic_phasecorrection_example.png', dpi=300)
plt.show()
# plt.close()
