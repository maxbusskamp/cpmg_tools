#%%
from nmr_tools import proc_base, processing
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.dpi'] = 120


# Or you can read in the unprocessed bruker files:
data1, ppm_scale1, hz_scale1, dic1 = processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/195Pt_Pt-Mix_WCPMG-MAS_23.02.21/2999/pdata/1', dict=True)
data2, ppm_scale2, hz_scale2, dic2 = processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/195Pt_Pt-Mix_WCPMG-MAS_23.02.21/3999/pdata/1', dict=True)

# Then apply some processing:
data1, null = processing.linebroadening(data1, lb_variant='hamming', lb_const=0.1)
data2, null = processing.linebroadening(data2, lb_variant='hamming', lb_const=0.1)


data1, phase1 = processing.autophase(data1, bnds=((0, 360), (-100000, -50000), (-10000, 0)),
                                   Ns=32, verb=True, loss_func='int_sum', workers=4, int_sum_cutoff=0.5,
                                   minimizer='Nelder-Mead', T=1000, niter=100, disp=False, stepsize=1000,
                                   tol=1e-25, options={'rhobeg':1000.0, 'maxiter':1000, 'maxfev':1000})

data1, ppm_scale1, hz_scale1, dic1 = processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/195Pt_Pt-Mix_WCPMG-MAS_23.02.21/2999/pdata/1', dict=True)
data1 = proc_base.ps2(proc_base.fft(proc_base.rev(proc_base.zf(data1, pad=4096*32))), p0=phase1[0], p1=phase1[1], p2=phase1[2])    # Fourier transform
ppm_scale1, hz_scale1 = processing.get_scale(data1, dic1)

data2, phase2 = processing.autophase(data2, bnds=((0, 360), (-100000, -50000), (-10000, 0)),
                                   Ns=32, verb=True, loss_func='int_sum', workers=4, int_sum_cutoff=0.5,
                                   minimizer='Nelder-Mead', T=1000, niter=100, disp=False, stepsize=1000,
                                   tol=1e-25, options={'rhobeg':1000.0, 'maxiter':1000, 'maxfev':1000})

data2, ppm_scale2, hz_scale2, dic2 = processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/195Pt_Pt-Mix_WCPMG-MAS_23.02.21/3999/pdata/1', dict=True)
data2 = proc_base.ps2(proc_base.fft(proc_base.rev(proc_base.zf(data2, pad=4096*32))), p0=phase2[0], p1=phase2[1], p2=phase2[2])    # Fourier transform
ppm_scale2, hz_scale2 = processing.get_scale(data2, dic2)


plt.figure()
plt.plot(ppm_scale1, data1.real, c='r', lw=1.0, label='p0={:.0f}°, p1={:.0f}°, p2={:.0f}°'.format(phase1[0],phase1[1],phase1[2]))
plt.plot(ppm_scale2, data2.real, c='b', lw=1.0, label='p0={:.0f}°, p1={:.0f}°, p2={:.0f}°'.format(phase2[0],phase2[1],phase2[2]))
plt.legend()

# and fourier transform the data:
datasets = [(data1, ppm_scale1, hz_scale1),
            (data2, ppm_scale2, hz_scale2),
            ]

output_path = '/home/m_buss13/'
output_name = 'combined'

# Combine Datsets
data = processing.combine_stepped_aq(datasets, set_sw=2000e3, precision_multi=8, verbose=True)
print('Finished combining Datasets')

# and fourier transform the data:
datasets_mc = ['/home/m_buss13/ownCloud/nmr_data/development/195Pt_Pt-Mix_WCPMG-MAS_23.02.21/2999/pdata/1',
               '/home/m_buss13/ownCloud/nmr_data/development/195Pt_Pt-Mix_WCPMG-MAS_23.02.21/3999/pdata/1',
              ]

output_path = '/home/m_buss13/'
output_name = 'combined'

# Combine Datsets
data_mc = processing.combine_stepped_aq(datasets_mc, set_sw=2000e3, precision_multi=8, verbose=True)
print('Finished combining Datasets')

# Just some plotting for the example
plt.figure()
plt.plot(data_mc[:,0], data_mc[:,1], lw=1.0, c='r', label='Combined Spectrum')
plt.plot(data[:,0], data[:,1], lw=1.0, c='k', label='Combined Spectrum')
plt.yticks([])
# plt.xlim(450000, -750000)
plt.xlim(-200000, -400000)
plt.savefig('PtMix_example_zoom.png', dpi=600)
plt.show()
