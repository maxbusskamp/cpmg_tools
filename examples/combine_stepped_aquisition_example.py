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
# data1, _, dic1 =   processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9510131/pdata/1', dict=True)
# data2, _, dic2 =   processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9510231/pdata/1', dict=True)
# data3, _, dic3 =   processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9510331/pdata/1', dict=True)
# data4, _, dic4 =   processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9510431/pdata/1', dict=True)
# data5, _, dic5 =   processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9510531/pdata/1', dict=True)
# data6, _, dic6 =   processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9510631/pdata/1', dict=True)
# data7, _, dic7 =   processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610131/pdata/1', dict=True)
# data8, _, dic8 =   processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610231/pdata/1', dict=True)
# data9, _, dic9 =   processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610331/pdata/1', dict=True)
# data10, _, dic10 = processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610431/pdata/1', dict=True)
# data11, _, dic11 = processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610531/pdata/1', dict=True)
# data12, _, dic12 = processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610631/pdata/1', dict=True)

# # Then apply some processing:
# data1, _ =  processing.linebroadening(data1, lb_variant='scipy_general_hamming',  **{'alpha':0.78})
# data2, _ =  processing.linebroadening(data2, lb_variant='scipy_general_hamming',  **{'alpha':0.78})
# data3, _ =  processing.linebroadening(data3, lb_variant='scipy_general_hamming',  **{'alpha':0.78})
# data4, _ =  processing.linebroadening(data4, lb_variant='scipy_general_hamming',  **{'alpha':0.78})
# data5, _ =  processing.linebroadening(data5, lb_variant='scipy_general_hamming',  **{'alpha':0.78})
# data6, _ =  processing.linebroadening(data6, lb_variant='scipy_general_hamming',  **{'alpha':0.78})
# data7, _ =  processing.linebroadening(data7, lb_variant='scipy_general_hamming',  **{'alpha':0.78})
# data8, _ =  processing.linebroadening(data8, lb_variant='scipy_general_hamming',  **{'alpha':0.78})
# data9, _ =  processing.linebroadening(data9, lb_variant='scipy_general_hamming',  **{'alpha':0.78})
# data10, _ = processing.linebroadening(data10, lb_variant='scipy_general_hamming', **{'alpha':0.78})
# data11, _ = processing.linebroadening(data11, lb_variant='scipy_general_hamming', **{'alpha':0.78})
# data12, _ = processing.linebroadening(data12, lb_variant='scipy_general_hamming', **{'alpha':0.78})

# # and fourier transform the data:
# datasets = [
#             (processing.fft(proc_base.rev(data1),  dic1,  si=4096*4)),
#             (processing.fft(proc_base.rev(data2),  dic2,  si=4096*4)),
#             (processing.fft(proc_base.rev(data3),  dic3,  si=4096*4)),
#             (processing.fft(proc_base.rev(data4),  dic4,  si=4096*4)),
#             (processing.fft(proc_base.rev(data5),  dic5,  si=4096*4)),
#             (processing.fft(proc_base.rev(data6),  dic6,  si=4096*4)),
#             (processing.fft(proc_base.rev(data7),  dic7,  si=4096*4)),
#             (processing.fft(proc_base.rev(data8),  dic8,  si=4096*4)),
#             (processing.fft(proc_base.rev(data9),  dic9,  si=4096*4)),
#             (processing.fft(proc_base.rev(data10), dic10, si=4096*4)),
#             (processing.fft(proc_base.rev(data11), dic11, si=4096*4)),
#             (processing.fft(proc_base.rev(data12), dic12, si=4096*4)),
#             ]

output_path = '/home/m_buss13/'
output_name = 'combined'

# Combine Datsets
data, ppm_scale, hz_scale = processing.combine_stepped_aq(datasets, set_sw=2e6, precision_multi=4, mode='skyline', verbose=True, larmor_freq=36.1597680)
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
