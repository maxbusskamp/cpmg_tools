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

simpson.create_simpson(output_path, output_name, sw=1e6, np=8192, spin_rate=25000, proton_frequency=500e6, crystal_file='rep10', gamma_angles=5,
                        cs_iso=10.0, csa=50.0, csa_eta=1.0, alpha=0.0, beta=0.0, gamma=0.0)
simpson.run_simpson(output_name, output_path)
ppm_scale_org, hz_scale_org, data_org = processing.read_ascii(output_path+ascii_file, larmor_freq=500.0)

mse = []
mae = []
logcosh = []
for i in range(51):
    simpson.create_simpson(output_path, output_name, sw=1e6, np=8192, spin_rate=25000, proton_frequency=500e6, crystal_file='rep10', gamma_angles=5,
                            cs_iso=10.0, csa=i, csa_eta=1.0, alpha=0.0, beta=0.0, gamma=0.0)
    simpson.run_simpson(output_name, output_path)

    ppm_scale_fit, hz_scale_fit, data_fit = processing.read_ascii(output_path+ascii_file, larmor_freq=500.0)

    mse = np.append(mse, processing.calc_mse(data_org, data_fit))
    mae = np.append(mae, processing.calc_mae(data_org, data_fit))
    logcosh = np.append(logcosh, processing.calc_logcosh(data_org, data_fit))

plt.figure()
plt.plot(mse, c='k')
plt.title('MSE')

plt.figure()
plt.plot(mae, c='k')
plt.title('MAE')

plt.figure()
plt.plot(logcosh, c='k')
plt.title('LogCosh')

#%%
# Example for multiple species
from nmr_tools import simpson, processing, proc_base
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.dpi'] = 200

output_path = '/home/m_buss13/testfiles/'
output_name = 'simpson_input.tcl'
ascii_file = 'simpson_input.tcl.xy'
spe_file = 'simpson_input.tcl.spe'

# Simulate species 1
simpson.create_simpson(output_path, output_name, nuclei='207Pb', sw=2e6, np=8192*2, spin_rate=25000, proton_frequency=500e6, crystal_file='rep100', gamma_angles=10,
                        cs_iso=1930, csa=-876, csa_eta=0.204, alpha=0.0, beta=0.0, gamma=0.0)
simpson.run_simpson(output_name, output_path)
timescale, data_org = processing.read_ascii_fid(output_path+ascii_file)

# Simulate species 2
simpson.create_simpson(output_path, output_name, nuclei='207Pb', sw=2e6, np=8192*2, spin_rate=25000, proton_frequency=500e6, crystal_file='rep100', gamma_angles=10,
                        cs_iso=1598, csa=-489, csa_eta=0.241, alpha=0.0, beta=0.0, gamma=0.0)
simpson.run_simpson(output_name, output_path)
timescale2, data_org2 = processing.read_ascii_fid(output_path+ascii_file)

# Summation of both species, regarding their respective scaling factor
data_sum = data_org + data_org2*1.0

# Fourier transform summation FID
ppm_scale, hz_scale , data_fft = processing.sfft(data_sum, timescale, si=8192*2, larmor_freq=104.609)

# Read-in comparison file
ppm_scale_pb, hz_scale_pb, data_pb = processing.read_ascii(output_path+'PbZrO3_mas_scaled_combined.xy', larmor_freq=104.609)

# Plotting
plt.figure()
plt.plot(ppm_scale_pb, data_pb[:,0], c='r')
plt.plot(ppm_scale, data_fft.real, c='k')
plt.xlim(0, 3500)
