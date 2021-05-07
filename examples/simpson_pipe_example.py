#%%
# Example for multiple species
from nmr_tools import simpson, processing
import matplotlib.pyplot as plt


plt.rcParams['figure.dpi'] = 200

# Specify working directory and filenames
output_path = '/home/m_buss13/testfiles/'
output_name = 'simpson_input'
ascii_file = 'simpson_input.xy'

# Simulate species 1
# Create a dictionary containing the custom parameter of species 1
input_dict = {'1': {
                    "nuclei":'207Pb',
                    "cs_iso":1930-1800,
                    "csa":-800,
                    "csa_eta":0.304,
                    "spin_rate":12500.0,
                    "crystal_file":'rep100',
                    "gamma_angles":11,
                    "proton_frequency":500.0e6,
                    "scaling_factor":1.0,
                    "lb": 2000,
                    },
              '2': {
                    "nuclei":'207Pb',
                    "cs_iso":1598-1820,
                    "csa":-450,
                    "csa_eta":0.021,
                    "spin_rate":12500.0,
                    "crystal_file":'rep100',
                    "gamma_angles":11,
                    "proton_frequency":500.0e6,
                    "scaling_factor":1.0,
                    "lb": 2000,
                    }}

# Create simpson inputfile
data, timescale = simpson.create_simpson(output_path, output_name, input_dict=input_dict)

# Fourier transform summation FID
data, ppm_scale, _ = processing.asciifft(data, timescale, si=8192*4, larmor_freq=104.609)

# Read-in comparison file
# data_pb, ppm_scale_pb, _ = processing.read_ascii(output_path+'PbZrO3_mas_scaled_combined.xy', larmor_freq=104.609)
# data_pb, ppm_scale_pb, _ = processing.read_brukerproc('/home/m_buss13/ownCloud/git/nmr_tools/examples/example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/11')
data_pb, ppm_scale_pb, _ = processing.read_brukerproc('/home/m_buss13/ownCloud/git/nmr_tools/examples/example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/1')

data_pb = data_pb/max(data_pb.real)
data = data/max(data.real)

mse = processing.calc_mse(data.real, data_pb.real)
print(mse)

# Plotting
plt.figure()
plt.plot(ppm_scale, data.real, c='k', label='Python Pipeline')
plt.plot(ppm_scale_pb, data_pb.real, c='r', ls='--', label='Manual Simulation')
# plt.plot(data.real, c='k', label='Python Pipeline')
# plt.plot(data_pb.real, c='r', ls='--', label='Manual Simulation')
plt.legend()
plt.xlim(-1500, 1500)
# plt.savefig('simpson_pipeline.png', dpi=600)
plt.show()
# plt.close()
