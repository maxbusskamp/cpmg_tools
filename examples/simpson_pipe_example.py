#%%
# Example for multiple species
from nmr_tools import simpson, processing, proc_base
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.dpi'] = 200

# Specify working directory and filenames
output_path = '/home/m_buss13/testfiles/'
output_name = 'simpson_input'
ascii_file = 'simpson_input.xy'
spe_file = 'simpson_input.spe'



# Simulate species 1
# Create a dictionary containing the custom parameter of species 1
input_dict = {'species_1': {
                            "nuclei":'207Pb',
                            "cs_iso":1930,
                            "csa":-876,
                            "csa_eta":0.204,
                            "spin_rate":25000.0,
                            "proton_frequency":500.0e6,
                            "scaling_factor":1.0
                            },
              'species_2': {
                            "nuclei":'207Pb',
                            "cs_iso":1598,
                            "csa":-489,
                            "csa_eta":0.241,
                            "spin_rate":25000.0,
                            "proton_frequency":500.0e6,
                            "scaling_factor":1.0
                            }}


# Create simpson inputfile
timescale, data = simpson.create_simpson(output_path, output_name, input_dict=input_dict)


# Fourier transform summation FID
ppm_scale, hz_scale , data_fft = processing.asciifft(data, timescale, si=8192*2, larmor_freq=104.609)

# Read-in comparison file
ppm_scale_pb, hz_scale_pb, data_pb = processing.read_ascii(output_path+'PbZrO3_mas_scaled_combined.xy', larmor_freq=104.609)

# Plotting
plt.figure()
plt.plot(ppm_scale, data_fft.real, c='k', label='Python Pipeline')
plt.plot(ppm_scale_pb, data_pb[:,0], c='r', ls='--', label='Manual Simulation')
plt.legend()
plt.xlim(0, 3500)
