#%%
# This is an example for the automatic phasecorrection up to 2nd order in python
import matplotlib.pyplot as plt
from cpmg_tools import processing
import time
plt.rcParams['figure.dpi'] = 200


start_time = time.time()


# # PtMix Example:
datapath = r'example_data/195Pt_PtMix_WCPMG/1/pdata/1'
datapath_mc = r'example_data/195Pt_PtMix_WCPMG/1/pdata/11'
# # bnds=((3900, 4100), (-55000, -54000), (-15000, -14000))

# Reiset Example NMR300 (Jonas):
# datapath = r'example_data/195Pt_Reiset_WCPMG_300er/7/pdata/1'
# datapath_mc = r'example_data/195Pt_Reiset_WCPMG_300er/7/pdata/11'
# bnds=((0, 360), (-65000, -55000), (-15000, -5000))

# # Reiset Example Neo500:
# datapath = r'example_data/195Pt_Reiset_WCPMG_500er/1/pdata/1'
# datapath_mc = r'example_data/195Pt_Reiset_WCPMG_500er/1/pdata/11'
# # bnds=((0, 360), (-100000, -50000), (-10000, 10000))

# # PbZrO3 Example Neo500:
# datapath = r'example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/1'
# datapath_mc = r'example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/11'
# # bnds=((0, 1000), (-270000, -260000))

data_mc, ppm_scale_mc, hz_scale_mc = processing.read_brukerproc(datapath_mc)
data, _, dic = processing.read_brukerfid(datapath, dict=True)

# Note! The starting variables are highly optimized to enable the use of very short optimizations for testing.
# Original Data should be optimized much longer, or in multiple steps
data, phase = processing.autophase(data, bnds=((3900, 4100), (-55000, -54000), (-15000, -14000)),
                                   Ns=4, verb=True, loss_func='phaseloss', workers=4, int_sum_cutoff=0.5,
                                   minimizer='Nelder-Mead', T=1000, niter=100, disp=False, stepsize=1000,
                                   tol=1e-25, options={'rhobeg':1000.0, 'maxiter':100, 'maxfev':100},
                                   zf=4096*32, prominence=0.05)

# Get scales for spectrum
ppm_scale, hz_scale = processing.get_scale(data, dic)

# Plotting
plt.figure()
plt.plot(ppm_scale_mc, data_mc/max(data_mc), c='k', lw=2.0, label='Magnitude')
if(len(phase)==2):
    plt.plot(ppm_scale, data.real/max(abs(data.real)), c='r', lw=1.0, label='p0={:.0f}°, p1={:.0f}°'.format(phase[0],phase[1]))
elif(len(phase)==3):
    plt.plot(ppm_scale, data.real/max(abs(data.real)), c='r', lw=1.0, label='p0={:.0f}°, p1={:.0f}°, p2={:.0f}°'.format(phase[0],phase[1],phase[2]))
else:
    ('Wrong number of boundary conditions! Please set only 2 or 3 conditions')

peaks, _ = processing.find_peaks(data.real, prominence=max(abs(data))*0.05)
plt.plot(ppm_scale[peaks], data[peaks].real/max(abs(data.real)), 'gs', ms=1.0)

plt.xlim(4000, -6500)
plt.xlabel('$^{195}$Pt / ppm')
plt.legend()
plt.yticks([])

plt.show()

print("-------------------------------")
print("---%s seconds ---" % (time.time() - start_time))
print("-------------------------------")
