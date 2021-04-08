#%%
from nmr_tools import processing
import matplotlib.pyplot as plt
import nmrglue as ng
import matplotlib.gridspec as gridspec
import numpy as np

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
            '/home/m_buss13/ownCloud/nmr_data/development/glycine_wcpmg/9610631/pdata/1',
            ]

output_path = '/home/m_buss13/'
output_name = 'combined'

# Combine Datsets
dataset_array = processing.combine_stepped_aq(datasets, set_sw=2000e3, precision_multi=10, verbose=True)
print('Finished combining Datasets')

fig = plt.figure(figsize=(6.5, 4), facecolor='#f4f4f4')
spec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
spec.update(wspace=0.0, hspace=0.0)
f1_ax1 = fig.add_subplot(spec[0, 0])

f1_ax1.plot(dataset_array[:,0], dataset_array[:,1], lw=.3, c='k', label='Combined Spectrum')
# f1_ax1.set_xlim(-800,300)
f1_ax1.set_yticks([])
f1_ax1.invert_xaxis()
plt.show()

# Save to XRI file
processing.save_xri(output_path, output_name, dataset_array)
print('Saved XRI')

# Save to .spe file
processing.save_spe(output_path, output_name, dataset_array)
print('Saved SPE')
