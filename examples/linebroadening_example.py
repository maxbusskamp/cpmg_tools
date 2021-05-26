#%%
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
from cpmg_tools import processing
import numpy as np
import scipy.signal as signal

plt.rcParams['figure.dpi'] = 200

# Set path to dataset
datapath = '/home/m_buss13/ownCloud/cpmg_tools/development/195Pt_PtMix_stepped/2999/pdata/1'

# Read Bruker FID
data, _ = processing.read_brukerfid(datapath)

# Apply linebroadening
data_lb, window = processing.linebroadening(data,
                                            lb_variant='scipy_chebwin',
                                            # lb_variant='scipy_general_hamming',
                                            # lb_variant='compressed_wurst',
                                            # lb_variant='shifted_wurst',
                                            # lb_variant='gaussian',
                                            # lb_variant='scipy',
                                            lb_const=0.2,
                                            lb_n=2,
                                            # **{'alpha':0.5}
                                            **{'at':100}
                                            )

# Plotting
plt.figure()
plt.plot(processing.interleave_complex(data.real, data.imag), c='k')
plt.plot(processing.interleave_complex(data_lb.real, data_lb.imag), c='grey')
plt.plot(np.linspace(0, len(data)*2, num=size(data)), window*max(processing.interleave_complex(data.real, data.imag)), c='r', label='1')
# plt.plot(np.linspace(0, len(data)*2, num=size(data)), window_scipy*data_lb.max(), c='b', label='SciPy')
# plt.ylim((0,1))
# plt.yticks([])
# plt.legend()

#%%
import matplotlib.pyplot as plt
from cpmg_tools import processing, proc_base
import numpy as np
plt.rcParams['figure.dpi'] = 200

# Read magnitude data
data_mc, ppm_scale_mc, hz_scale_mc = processing.read_brukerproc('/home/m_buss13/ownCloud/git/cpmg_tools/examples/example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/1')

# Read bruker FID
data, timescale, dic = processing.read_brukerfid('/home/m_buss13/ownCloud/git/cpmg_tools/examples/example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/1', dict=True)
data2, timescale2, dic2 = processing.read_brukerfid('/home/m_buss13/ownCloud/git/cpmg_tools/examples/example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/1', dict=True)
# Save data for comaprison
data_before_lb = data
# Apply linebroadening
data, window = processing.linebroadening(data,
                                        # lb_variant='scipy_exponential',
                                        # lb_variant='scipy_chebwin',
                                        # lb_variant='scipy_taylor',
                                        # lb_variant='scipy_parzen',
                                        # lb_variant='scipy_nuttall',
                                        # lb_variant='scipy_general_hamming',
                                        # lb_variant='scipy_blackmanharris',
                                        # lb_variant='scipy_kaiser',
                                        # lb_variant='scipy_dpss',
                                        # lb_variant='compressed_wurst',
                                        lb_variant='shifted_wurst',
                                        # lb_variant='gaussian',
                                        # lb_variant='scipy',
                                        lb_const=0.24,
                                        lb_n=2, 
                                        # **{'nbar':2, 'sll':50}
                                        # **{'NW':1.0}
                                        # **{'beta':4}
                                        # **{'alpha':0.62}
                                        # **{'at':50}
                                        # **{'tau':600}
                                        )


fid_before = data_before_lb
fid_after = data
# Fouriertransform, zerofilling and phasing
data = proc_base.zf_size(data, 32768)    # zero fill to 32768 points
data = proc_base.fft(proc_base.rev(data))               # Fourier transform
data = proc_base.ps(data, p0=847, p1=-268718)

# Generate new scales
ppm_scale, hz_scale = processing.get_scale(data, dic)

# Apply linebroadening
data2, window2 = processing.linebroadening(data2,
                                        # lb_variant='scipy_exponential',
                                        # lb_variant='scipy_chebwin',
                                        lb_variant='scipy_general_hamming',
                                        # lb_variant='scipy_blackmanharris',
                                        # lb_variant='scipy_kaiser',
                                        # lb_variant='scipy_dpss',
                                        # lb_variant='compressed_wurst',
                                        # lb_variant='shifted_wurst',
                                        # lb_variant='gaussian',
                                        # lb_variant='scipy',
                                        # lb_const=0.1,
                                        # lb_n=5, 
                                        # **{'NW':1.5}
                                        # **{'beta':4}
                                        **{'alpha':0.62}
                                        # **{'at':50}
                                        # **{'tau':600}
                                        )

# Fouriertransform, zerofilling and phasing
data2 = proc_base.zf_size(data2, 32768)    # zero fill to 32768 points
data2 = proc_base.fft(proc_base.rev(data2))               # Fourier transform
data2 = proc_base.ps(data2, p0=847, p1=-268718)

# Generate new scales
ppm_scale2, hz_scale2 = processing.get_scale(data2, dic2)

# Process comparison data equally
data_before_lb = proc_base.zf_size(data_before_lb, 32768)    # zero fill to 32768 points
data_before_lb = proc_base.fft(proc_base.rev(data_before_lb))               # Fourier transform
data_before_lb = proc_base.ps(data_before_lb, p0=847, p1=-268718)


plt.figure()
plt.plot(processing.interleave_complex(fid_before.real, fid_before.imag), c='k')
plt.plot(processing.interleave_complex(fid_after.real, fid_after.imag), c='grey')
plt.plot(np.linspace(0, len(fid_before)*2, num=len(fid_before)), window*max(abs(processing.interleave_complex(fid_before.real, fid_before.imag))), c='r')
plt.plot(np.linspace(0, len(fid_before)*2, num=len(fid_before)), window2*max(abs(processing.interleave_complex(fid_before.real, fid_before.imag))), c='b')
plt.yticks([])

# Plotting
plt.figure()
# plt.plot(ppm_scale_mc, data_mc/max(data_mc), c='r', lw=1.0, label='Magnitude')
# plt.plot(data_before_lb.real/max(abs(data_before_lb.real)), c='dimgrey', lw=0.5, label='No LB')
plt.plot(ppm_scale, data_before_lb.real/max(abs(data_before_lb.real)), c='dimgrey', lw=0.5, label='No LB')
plt.plot(ppm_scale, data.real/max(abs(data.real)), c='r', lw=0.5, label='First Method')
plt.plot(ppm_scale2, data2.real/max(abs(data2.real)), c='b', lw=0.5, label='Second method')

# plt.xlim(-300, 0)
# plt.xlim(-800, 800)
plt.xlim(-1500, 1500)
plt.legend()
plt.yticks([])
plt.show()
# plt.savefig('automatic_phasecorrection_example.png', dpi=300)
# plt.close()

print('Data before LB:')
print(processing.signaltonoise_region(data_before_lb.real, noisepts=(1000, 15000)))
print('Data after LB Method 1:')
print(processing.signaltonoise_region(data.real, noisepts=(1000, 15000)))
print('Data after LB Method 2:')
print(processing.signaltonoise_region(data2.real, noisepts=(1000, 15000)))
