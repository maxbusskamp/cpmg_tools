#%%
from nmr_tools import processing
import matplotlib.pyplot as plt

# # You can use direct paths to the processed bruker files:
# datasets = ['/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9510131/pdata/1',
#             '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9510231/pdata/1',
#             '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9510331/pdata/1',
#             '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9510431/pdata/1',
#             '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9510531/pdata/1',
#             '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9510631/pdata/1',
#             '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610131/pdata/1',
#             '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610231/pdata/1',
#             '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610331/pdata/1',
#             '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610431/pdata/1',
#             '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610531/pdata/1',
#             '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610631/pdata/1',
#             ]


# Or you can read in the unprocessed bruker files:
ppm_scale1, hz_scale1, data1, dic1 = processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610531/pdata/1', dict=True)
ppm_scale2, hz_scale2, data2, dic2 = processing.read_brukerfid('/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610631/pdata/1', dict=True)

# Then apply some processing:
data1, null = processing.linebroadening(data1, lb_variant='hamming', lb_const=0.1)
data2, null = processing.linebroadening(data2, lb_variant='hamming', lb_const=0.1)

# and fourier transform the data:
datasets = [(processing.fft(data1, dic1, si=32168)),
            (processing.fft(data2, dic2, si=32168)),
            ]


output_path = '/home/m_buss13/'
output_name = 'combined'

# Combine Datsets
dataset_array = processing.combine_stepped_aq(datasets, set_sw=2000e3, precision_multi=2, verbose=True)
print('Finished combining Datasets')

# Just some plotting for the example
plt.plot(dataset_array[:,0], dataset_array[:,1], lw=.3, c='k', label='Combined Spectrum')
plt.yticks([])
plt.show()

# # Save to XRI file
# processing.save_xri(output_path, output_name, dataset_array)
# print('Saved XRI')

# # Save to .spe file
# processing.save_spe(output_path, output_name, dataset_array)
# print('Saved SPE')
