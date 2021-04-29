
#%%
# Example for the different loss function calculations
from nmr_tools import simpson
from nmr_tools import processing
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.dpi'] = 200

output_path = '/home/m_buss13/testfiles/'
output_name = 'simpson_input.tcl'
ascii_file = 'simpson_input.tcl.xy'
spe_file = 'simpson_input.tcl.spe'

# Simulate species 1
# Create a dictionary containing the custom parameter of species 1
input_dict = {
            "nuclei":'207Pb',
            "cs_iso":1930,
            "csa":-876,
            "csa_eta":0.204,
            "spin_rate":25000.0,
            "proton_frequency":500.0e6
             }
# Create simpson inputfile
simpson.create_simpson(output_path, output_name, input_dict=input_dict)
# Run previously created simpson inputfile
simpson.run_simpson(output_name, output_path)
# Read FID from ASCII file. FFT can also be performed in Simpson, then read_ascii() can be used.
timescale, data_org = processing.read_ascii_fid(output_path+ascii_file)

# Simulate species 2
# Create a dictionary containing the custom parameter of species 2
input_dict = {
            "nuclei":'207Pb',
            "cs_iso":1598,
            "csa":-489,
            "csa_eta":0.241,
            "spin_rate":25000.0,
            "proton_frequency":500.0e6
             }
# Create simpson inputfile
simpson.create_simpson(output_path, output_name, input_dict=input_dict)
# Run previously created simpson inputfile
simpson.run_simpson(output_name, output_path)
# Read FID from ASCII file. FFT can also be performed in Simpson, then read_ascii() can be used.
timescale2, data_org2 = processing.read_ascii_fid(output_path+ascii_file)

# Summation of both species, regarding their respective scaling factor. Should be performed on FIDs!
data_sum = data_org + data_org2*1.0

# Fourier transform summation FID
ppm_scale, hz_scale , data_fft = processing.asciifft(data_sum, timescale, si=8192*2, larmor_freq=104.609)

# Read-in comparison file
ppm_scale_pb, hz_scale_pb, data_pb = processing.read_ascii(output_path+'PbZrO3_mas_scaled_combined.xy', larmor_freq=104.609)





# Plotting
plt.figure()
plt.plot(ppm_scale, data_fft.real, c='k', label='Python Pipeline')
plt.plot(ppm_scale_pb, data_pb[:,0], c='r', ls='--', label='Manual Simulation')
plt.legend()
plt.xlim(0, 3500)
