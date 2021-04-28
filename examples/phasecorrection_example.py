#%%
import matplotlib.pyplot as plt
from nmr_tools import processing, proc_base
import numpy as np
import time
from scipy.integrate import simps
plt.rcParams['figure.dpi'] = 200


start_time = time.time()

# # PtMix Example:
datapath = '/home/m_buss13/ownCloud/nmr_data/development/195Pt_Pt-Mix_WCPMG/1/pdata/1'
datapath_mc = '/home/m_buss13/ownCloud/nmr_data/development/195Pt_Pt-Mix_WCPMG/1/pdata/11'
# # bnds=((3900, 4100), (-55000, -54000), (-15000, -14000))

# Reiset Example NMR300 (Jonas):
# datapath = '/home/m_buss13/ownCloud/nmr_data/development/195Pt_[Pt(NH3)4]Cl2_WCPMG_14.06.16/7/pdata/1'
# datapath_mc = '/home/m_buss13/ownCloud/nmr_data/development/195Pt_[Pt(NH3)4]Cl2_WCPMG_14.06.16/7/pdata/11'
# bnds=((0, 360), (-65000, -55000), (-15000, -5000))

# # Reiset Example Neo500:
# datapath = '/home/m_buss13/ownCloud/nmr_data/development/195Pt_Reiset_WCPMG/1/pdata/1'
# datapath_mc = '/home/m_buss13/ownCloud/nmr_data/development/195Pt_Reiset_WCPMG/1/pdata/11'
# # bnds=((0, 360), (-100000, -50000), (-10000, 10000))

# # PbZrO3 Example Neo500:
# datapath = '/home/m_buss13/ownCloud/nmr_data/development/207Pb_wcpmg_mas/1/pdata/1'
# datapath_mc = '/home/m_buss13/ownCloud/nmr_data/development/207Pb_wcpmg_mas/1/pdata/11'
# # bnds=((0, 1000), (-270000, -260000))

ppm_scale_mc, hz_scale_mc, data_mc = processing.read_brukerproc(datapath_mc)
ppm_scale, hz_scale, data, dic = processing.read_brukerfid(datapath, dict=True)

data, phase = processing.autophase(data, bnds=((3900, 4100), (-55000, -54000), (-15000, -14000)),
                                   Ns=8, verb=True, loss_func='int_sum', workers=4, int_sum_cutoff=0.5,
                                   minimizer='Nelder-Mead', T=1000, niter=100, disp=False, stepsize=1000,
                                   tol=1e-25, options={'rhobeg':1000.0, 'maxiter':1000, 'maxfev':1000},
                                   zf=4096*32)

ppm_scale, hz_scale = processing.get_scale(data, dic)

plt.figure()
plt.plot(ppm_scale_mc, data_mc/max(data_mc), c='k', lw=1.0, label='Magnitude')
if(len(phase)==2):
    plt.plot(ppm_scale, data.real/max(abs(data.real)), c='r', lw=1.0, label='p0={:.0f}°, p1={:.0f}°'.format(phase[0],phase[1]))
elif(len(phase)==3):
    plt.plot(ppm_scale, data.real/max(abs(data.real)), c='r', lw=1.0, label='p0={:.0f}°, p1={:.0f}°, p2={:.0f}°'.format(phase[0],phase[1],phase[2]))
else:
    ('Wrong number of boundary conditions! Please set only 2 or 3 conditions')

# plt.xlim(1000, -1000)
# plt.xlim(1000, -5500)
plt.xlim(3000, -5500)
# plt.xlim(-1000, -1100)
# plt.xlim(-1500, 1500)
plt.legend()
plt.yticks([])
plt.savefig('automatic_phasecorrection_example.png', dpi=300)
plt.show()
plt.close()

print("-------------------------------")
print("---%s seconds ---" % (time.time() - start_time))
print("-------------------------------")
