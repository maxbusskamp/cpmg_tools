#%%
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cpmg_tools import processing, proc_base, bruker


plt.rcParams['figure.dpi'] = 200

savepath = '/home/m_buss13/ownCloud/git/cpmg_tools/examples/example_data/writetest/1'
datapath='/home/m_buss13/ownCloud/git/cpmg_tools/examples/example_data/119Sn_SnO2_double_echo_cpmg/1100/pdata/1'
# Split FID echotrain and sum all echos
data, _, dic = processing.split_echotrain(datapath=datapath, dw=0.5, echolength=300, blankinglength=300, numecho=50)

data, window = processing.linebroadening(data,
                                        lb_variant='scipy_hamming'
                                        )

processing.write_fid(savepath, data, dic, overwrite=True)
