#%%
from nmr_tools import simpson
from nmr_tools import processing
import numpy as np
import matplotlib.pyplot as plt

output_path = '/home/m_buss13/testfiles/'
output_name = 'simpson_input.tcl'
ascii_file = 'simpson_input.tcl.xy'
spe_file = 'simpson_input.tcl.spe'

mse = []
mae = []
logcosh = []

for i in range(11):
    simpson.create_simpson(output_path, output_name, sw=1e6, np=8192, spin_rate=25000, proton_frequency=500e6, crystal_file='rep10', gamma_angles=5,
                           cs_iso=10.0, csa=100.0, csa_eta=1.0, alpha=0.0, beta=0.0, gamma=0.0)
    simpson.run_simpson(output_name, output_path)

    ppm_scale_spe, hz_scale_spe, data_spe = processing.read_spe(output_path+spe_file, larmor_freq=500.0)
    ppm_scale_ascii, hz_scale_ascii, data_ascii = processing.read_ascii(output_path+ascii_file, larmor_freq=500.0)

    data_ascii = data_ascii*(0.75+i/20.0)


    mse = np.append(mse, processing.calc_mse(data_spe, data_ascii))
    mae = np.append(mae, processing.calc_mae(data_spe, data_ascii))
    logcosh = np.append(logcosh, processing.calc_logcosh(data_spe, data_ascii))

plt.figure()
plt.plot(mse, c='k')
plt.title('mse')
plt.figure()
plt.plot(mae, c='k')
plt.title('mae')
plt.figure()
plt.plot(logcosh, c='k')
plt.title('logcosh')
