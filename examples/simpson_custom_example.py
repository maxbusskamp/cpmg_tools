#%%
# Example for the different loss function calculations
from nmr_tools import simpson
from nmr_tools import processing
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.dpi'] = 200

# Specify working directory and filenames
output_path = '/home/m_buss13/testfiles/'
output_name = 'simpson_input'
ascii_file = 'simpson_input.xy'

# Create a dictionary containing the custom parameter of species 1
input_dict = {'cs_iso': 0.0,
              'csa': 100.0,
              'start_operator': 'Inz',
              'pulse': 'pulse 2.5 100000 90',
              'spin_rate': 12500}

# Create an empty dictionary for custom parts of the scripts. possible keys are: 'spinsys', 'par', 'pulseq', 'main'
# These will override any defaults in that section, but can use the default .format() placeholder e.g. {alpha}
proc_dict = {}
proc_dict['spinsys'] = """
spinsys {{
    channels 1H
    nuclei 1H 1H
    shift 1 {cs_iso}p {csa}p {csa_eta} {alpha} {beta} {gamma}
    shift 2 50p 50p 0.5 {alpha} {beta} {gamma}
    dipole 1 2 -10000 0 0 0
}}
"""

# Create simpson inputfile
simpson.create_simpson(output_path, output_name, input_dict=input_dict, proc_dict=proc_dict)
# Run previously created simpson inputfile
simpson.run_simpson(output_name, output_path)

# Read FID from ASCII file. FFT can also be performed in Simpson, then read_ascii() can be used.
data, timescale = processing.read_ascii_fid(output_path+ascii_file)

# Fourier transform FID
data, ppm_scale, hz_scale = processing.asciifft(data, timescale, si=8192*2, larmor_freq=500.0)

# Plotting
plt.figure()
plt.plot(ppm_scale, data, c='r')
plt.xlim(-250, 250)
