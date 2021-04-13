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

simpson.create_simpson(output_path, output_name, sw=1e6, np=8192, spin_rate=25000, proton_frequency=500e6, crystal_file='rep10', gamma_angles=5,
                        cs_iso=10.0, csa=50.0, csa_eta=1.0, alpha=0.0, beta=0.0, gamma=0.0)
simpson.run_simpson(output_name, output_path)
ppm_scale_org, hz_scale_org, data_org = processing.read_ascii(output_path+ascii_file, larmor_freq=500.0)

for i in range(101):
    simpson.create_simpson(output_path, output_name, sw=1e6, np=8192, spin_rate=25000, proton_frequency=500e6, crystal_file='rep10', gamma_angles=5,
                            cs_iso=10.0, csa=i, csa_eta=1.0, alpha=0.0, beta=0.0, gamma=0.0)
    simpson.run_simpson(output_name, output_path)

    ppm_scale_fit, hz_scale_fit, data_fit = processing.read_ascii(output_path+ascii_file, larmor_freq=500.0)

    mse = np.append(mse, processing.calc_mse(data_org, data_fit))
    mae = np.append(mae, processing.calc_mae(data_org, data_fit))
    logcosh = np.append(logcosh, processing.calc_logcosh(data_org, data_fit))

plt.figure()
plt.plot(mse, c='k')
plt.title('mse')
plt.savefig('mse.png', )
plt.figure()
plt.plot(mae, c='k')
plt.title('mae')

plt.figure()
plt.plot(logcosh, c='k')
plt.title('logcosh')
