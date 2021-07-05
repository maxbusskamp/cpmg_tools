#%%
# Example for multiple species
from cpmg_tools import simpson, processing, proc_base
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 200

# Specify working directory and filenames
output_path = '/home/m_buss13/testfiles/'
output_name = 'simpson_input'
ascii_file = 'simpson_input.xy'

# Simulate two species
# Create a dictionary containing the custom parameter of both species
input_dict = {'1': {
                    "nuclei":'207Pb',
                    "cs_iso":-218.07138,
                    "csa":-500.52431,
                    "csa_eta":0.000,
                    "spin_rate":12500.0,
                    "crystal_file":'rep256',
                    "gamma_angles":17,
                    "proton_frequency":500.0e6,
                    "sw":2.5e6,
                    "scaling_factor":1.0,
                    "lb": 2153.78219,
                    },
              '2': {
                    "nuclei":'207Pb',
                    "cs_iso":128.80839,
                    "csa":-836.86949,
                    "csa_eta":0.27827,
                    "spin_rate":12500.0,
                    "crystal_file":'rep256',
                    "gamma_angles":17,
                    "proton_frequency":500.0e6,
                    "sw":2.5e6,
                    "scaling_factor":1.0,
                    "lb": 2166.78764,
                    }}


# # Create a dictionary containing the custom parameter of both species
# input_dict = {
#             "nuclei":'207Pb 207Pb',
#             "channels":'207Pb',
#             "cs_iso1":-246,
#             "cs_iso2":142,
#             "csa1":-452,
#             "csa2":-717,
#             "csa_eta1":0.48,
#             "csa_eta2":0.38,
#             "spin_rate":12500.0,
#             "crystal_file":'rep678',
#             "gamma_angles":33,
#             "proton_frequency":500.0e6,
#             "sw":2.5e6,
#             "lb": 2462,
#             }

# # Create an empty dictionary for custom parts of the scripts. possible keys are: 'spinsys', 'par', 'pulseq', 'main'
# # These will override any defaults in that section, but can use the default .format() placeholder e.g. {alpha}
# proc_dict = {}
# proc_dict['spinsys'] = """
#     spinsys {{
#         channels {channels}
#         nuclei {nuclei}
#         shift 1 {cs_iso1}p {csa1}p {csa_eta1} {alpha} {beta} {gamma}
#         shift 2 {cs_iso2}p {csa2}p {csa_eta2} {alpha} {beta} {gamma}
#     }}
#     """


# Create simpson inputfile
data, timescale = simpson.create_simpson(output_path, output_name, input_dict=input_dict, proc_dict=None)

# Fourier transform summation FID
data, ppm_scale, hz_scale = processing.asciifft(data, timescale, si=8192*4, larmor_freq=104.609)

# For read-in of comparison file, uncomment on of the following options:
# Either a already processed spectrum:
data_pb, ppm_scale_pb, _ = processing.read_brukerproc('/home/m_buss13/ownCloud/git/cpmg_tools/examples/example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/12')

# # Or a Bruker FID processed directly in python:
# data_pb, timescale, dic_pb = processing.read_brukerfid('/home/m_buss13/ownCloud/git/cpmg_tools/examples/example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/1', dict=True)
# # If FID is read-in, apply linebroadening
# data_pb, window = processing.linebroadening(data_pb, lb_variant='scipy_general_hamming', **{'alpha': 0.7})
# # Fouriertransform and zerofill the FID
# data_pb, ppm_scale_pb, _ = processing.fft(data_pb, si=32768, dic=dic_pb, mc=True)


# Scale both spectra to maximum 1.0
data_pb = data_pb/max(data_pb.real)
data = data/max(data.real)

# Calculate residual
mse = processing.calc_logcosh(data.real, data_pb.real)

# Plotting
plt.figure()
plt.plot(ppm_scale, data.real, c='k', lw=1.0, label='Python Pipeline')
plt.plot(ppm_scale_pb, data_pb.real, c='r', lw=1.0, label='\nExperiment\nMSE = {:.4f}'.format(mse))
plt.xlim(1000, -1000)
plt.yticks([])
plt.xlabel('$^{207}$Pb / ppm')
plt.legend()

plt.show()
