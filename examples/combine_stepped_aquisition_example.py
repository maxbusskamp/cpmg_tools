#%%
from nmr_tools import proc_base, processing
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 200

# You can use direct paths to the processed bruker files:
datasets = ['/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9510131/pdata/1',
            '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9510231/pdata/1',
            '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9510331/pdata/1',
            '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9510431/pdata/1',
            '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9510531/pdata/1',
            '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9510631/pdata/1',
            '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610131/pdata/1',
            '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610231/pdata/1',
            '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610331/pdata/1',
            '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610431/pdata/1',
            '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610531/pdata/1',
            '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610631/pdata/1'
            ]


# # Or you can read in the unprocessed bruker files:
# data1, _, dic1 = processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9510131/pdata/1', dict=True)
# data2, _, dic2 = processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9510331/pdata/1', dict=True)
# data3, _, dic3 = processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610531/pdata/1', dict=True)
# data4, _, dic4 = processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610631/pdata/1', dict=True)

# # Then apply some processing:
# data1, _ = processing.linebroadening(data1, lb_variant='hamming', lb_const=0.54)
# data2, _ = processing.linebroadening(data2, lb_variant='hamming', lb_const=0.54)
# data3, _ = processing.linebroadening(data3, lb_variant='hamming', lb_const=0.54)
# data4, _ = processing.linebroadening(data4, lb_variant='hamming', lb_const=0.54)

# # and fourier transform the data:
# datasets = [
#             (processing.fft(proc_base.rev(data1), dic1, si=4096*4)),
#             (processing.fft(proc_base.rev(data2), dic2, si=4096*4)),
#             (processing.fft(proc_base.rev(data3), dic3, si=4096*4)),
#             (processing.fft(proc_base.rev(data4), dic4, si=4096*4)),
#             ]

output_path = '/home/m_buss13/'
output_name = 'combined'

# Combine Datsets
data, ppm_scale, hz_scale = processing.combine_stepped_aq(datasets, set_sw=2e6, precision_multi=2, mode='skyline', verbose=True, larmor_freq=36.1597680)
print('Finished combining Datasets')

# Just some plotting for the example
plt.figure()
plt.plot(hz_scale, data.real, lw=.3, c='k', label='Combined Spectrum')
plt.yticks([])
plt.show()

# Save to XRI file
processing.save_xri(output_path, output_name, data, hz_scale)

# Save to .spe file
processing.save_spe(output_path, output_name, data, hz_scale)
