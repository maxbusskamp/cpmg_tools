# Example for multiple species
from nmr_tools import simpson, processing, proc_base
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 200

# Specify working directory and filenames
output_path = '/home/m_buss13/testfiles/'
output_name = 'simpson_input'
ascii_file = 'simpson_input.xy'

# Simulate species 1
# Create a dictionary containing the custom parameter of species 1
input_dict = {'1': {
                    "nuclei":'207Pb',
                    "cs_iso":1598-1816,
                    "csa":-524.01547,
                    "csa_eta":0.00146,
                    "spin_rate":12500.0,
                    "crystal_file":'rep2000',
                    "gamma_angles":45,
                    "proton_frequency":500.0e6,
                    "sw":2.5e6,
                    "scaling_factor":1.0,
                    "lb": 2303.67112,
                    },
              '2': {
                    "nuclei":'207Pb',
                    "cs_iso":1930-1800,
                    "csa":-891.54853,
                    "csa_eta":0.27245,
                    "spin_rate":12500.0,
                    "crystal_file":'rep2000',
                    "gamma_angles":45,
                    "proton_frequency":500.0e6,
                    "sw":2.5e6,
                    "scaling_factor":1.0,
                    "lb": 2159.36892,
                    }}

# Create simpson inputfile
data, timescale = simpson.create_simpson(output_path, output_name, input_dict=input_dict)

# Fourier transform summation FID
data, ppm_scale, hz_scale = processing.asciifft(data, timescale, si=8192*4, larmor_freq=104.609)

# Read-in comparison file
# data_pb, ppm_scale_pb, _ = processing.read_ascii(output_path+'PbZrO3_mas_scaled_combined.xy', larmor_freq=104.609)
# data_pb, ppm_scale_pb, _ = processing.read_brukerproc('/home/m_buss13/ownCloud/git/nmr_tools/examples/example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/10')
# data_pb, ppm_scale_pb, _ = processing.read_brukerproc('/home/m_buss13/ownCloud/git/nmr_tools/examples/example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/11')
# data_pb, ppm_scale_pb, _ = processing.read_brukerproc('/home/m_buss13/ownCloud/git/nmr_tools/examples/example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/12')

data_pb, timescale, dic_pb = processing.read_brukerfid('/home/m_buss13/ownCloud/git/nmr_tools/examples/example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/1', dict=True)
data_pb, window = processing.linebroadening(data_pb, lb_variant='hamming', lb_const=0.54, lb_n=2)

data_pb = proc_base.zf_size(data_pb, 32768)    # zero fill to 32768 points
data_pb = proc_base.fft(proc_base.rev(data_pb))               # Fourier transform
data_pb = proc_base.ps(data_pb, p0=847, p1=-268718)
ppm_scale_pb, hz_scale_pb = processing.get_scale(data_pb, dic_pb)

data_pb = data_pb/max(data_pb.real)
data = data/max(data.real)

# data = np.insert(data, 0, 0)
# data = np.flip(data)
# mse = data.real - data_pb.real
mse = processing.calc_logcosh(data.real, data_pb.real)
print(mse)

# Plotting
plt.figure()
# plt.plot(data.real, c='k', lw=1.0, label='Python Pipeline')
# plt.plot(data_pb.real, c='r', lw=1.0, label='Experiment')
# plt.plot(mse, c='grey', lw=1.0, label='Python Pipeline')
# plt.xlim(18000, 15000)

plt.plot(ppm_scale, data.real, c='k', lw=1.0, label='Python Pipeline')
plt.plot(ppm_scale_pb, data_pb.real, c='r', lw=1.0, label='Experiment')
# # plt.plot(ppm_scale, mse, c='grey', lw=1.0, label='Python Pipeline')
plt.xlim(1000, -1000)
plt.yticks([])

plt.legend()
plt.savefig('simpson_pipeline_rep2000.png', dpi=600)
# plt.show()
plt.close()
