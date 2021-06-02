#%%
from cpmg_tools import proc_base, processing
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 120


# Or you can read in the unprocessed bruker files:
data1, timescale1, dic1 = processing.read_brukerfid('/home/m_buss13/ownCloud/git/cpmg_tools/examples/example_data/195Pt_PtMix_MAS_WCPMG_stepped/2999/pdata/1', dict=True)
data2, timescale2, dic2 = processing.read_brukerfid('/home/m_buss13/ownCloud/git/cpmg_tools/examples/example_data/195Pt_PtMix_MAS_WCPMG_stepped/3999/pdata/1', dict=True)


# Then apply some processing:
data1, null = processing.linebroadening(data1, lb_variant='scipy_hamming')
data2, null = processing.linebroadening(data2, lb_variant='scipy_hamming')

# Automatically phase spectrum
data1, phase1 = processing.autophase(data1, bnds=((0, 360), (-100000, -50000), (-10000, 0)),
                                   Ns=32, verb=True, loss_func='int_sum', workers=4, int_sum_cutoff=0.5,
                                   minimizer='Nelder-Mead', T=1000, niter=100, disp=False, stepsize=1000,
                                   tol=1e-25, options={'rhobeg':1000.0, 'maxiter':1000, 'maxfev':1000})

# Fouriertransform, zerofill and phase spectrum. Then calculate new scales
data1, timescale1, dic1 = processing.read_brukerfid('/home/m_buss13/ownCloud/git/cpmg_tools/examples/example_data/195Pt_PtMix_MAS_WCPMG_stepped/2999/pdata/1', dict=True)
data1 = proc_base.ps2(proc_base.fft(proc_base.rev(proc_base.zf(data1, pad=4096*32))), p0=phase1[0], p1=phase1[1], p2=phase1[2])    # Fourier transform
ppm_scale1, hz_scale1 = processing.get_scale(data1, dic1)

# Automatically phase spectrum
data2, phase2 = processing.autophase(data2, bnds=((0, 360), (-100000, -50000), (-10000, 0)),
                                   Ns=32, verb=True, loss_func='int_sum', workers=4, int_sum_cutoff=0.5,
                                   minimizer='Nelder-Mead', T=1000, niter=100, disp=False, stepsize=1000,
                                   tol=1e-25, options={'rhobeg':1000.0, 'maxiter':1000, 'maxfev':1000})

# Fouriertransform, zerofill and phase spectrum. Then calculate new scales
data2, timescale2, dic2 = processing.read_brukerfid('/home/m_buss13/ownCloud/git/cpmg_tools/examples/example_data/195Pt_PtMix_MAS_WCPMG_stepped/3999/pdata/1', dict=True)
data2 = proc_base.ps2(proc_base.fft(proc_base.rev(proc_base.zf(data2, pad=4096*32))), p0=phase2[0], p1=phase2[1], p2=phase2[2])    # Fourier transform
ppm_scale2, hz_scale2 = processing.get_scale(data2, dic2)

# Plotting
plt.figure()
plt.plot(ppm_scale1, data1.real, c='r', lw=1.0, label='p0={:.0f}°, p1={:.0f}°, p2={:.0f}°'.format(phase1[0],phase1[1],phase1[2]))
plt.plot(ppm_scale2, data2.real, c='b', lw=1.0, label='p0={:.0f}°, p1={:.0f}°, p2={:.0f}°'.format(phase2[0],phase2[1],phase2[2]))
plt.xlim(5000, -8000)
plt.legend()

# Combine processed datasets
datasets = [(data1, ppm_scale1, hz_scale1),
            (data2, ppm_scale2, hz_scale2),
            ]

output_path = '/home/m_buss13/'
output_name = 'combined'

# Combine Datsets
data, hz_scale = processing.combine_stepped_aq(datasets, set_sw=2000e3, precision_multi=8, verbose=True)
print('Finished combining Datasets')

# Combine magnitude datasets
datasets_mc = ['/home/m_buss13/ownCloud/git/cpmg_tools/examples/example_data/195Pt_PtMix_MAS_WCPMG_stepped/2999/pdata/1',
               '/home/m_buss13/ownCloud/git/cpmg_tools/examples/example_data/195Pt_PtMix_MAS_WCPMG_stepped/3999/pdata/1',
              ]

# Combine Datsets
data_mc, hz_scale_mc = processing.combine_stepped_aq(datasets_mc, set_sw=2000e3, precision_multi=8, verbose=True)
print('Finished combining Datasets')

# Just some plotting for the example
plt.figure()
plt.plot(hz_scale/1000, data, lw=1.0, c='r', label='Combined Spectrum')
plt.plot(hz_scale_mc/1000, data_mc, lw=1.0, c='k', label='Combined Spectrum')
plt.yticks([])
# plt.xlim(500, -700)
plt.xlim(-400, -600)
plt.xlabel('$^{195}$Pt / kHz')
plt.savefig('/home/m_buss13/ownCloud/plots/cpmg_tools/PtMix_example_zoom', dpi=600)
plt.show()
