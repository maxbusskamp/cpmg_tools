#%%
# Example for the different loss function calculations
from nmr_tools import simpson, processing
import matplotlib.pyplot as plt
from lmfit import fit_report
import time
plt.rcParams['figure.dpi'] = 200


start_time = time.time()

output_path = '/home/m_buss13/testfiles/'
output_name = 'simpson_input'
ascii_file = 'simpson_input.xy'

# Create a dictionary containing the custom parameter of both species
input_dict = {'1': {
                    "nuclei":'207Pb',
                    "cs_iso":1598-1816,
                    "csa":-489,
                    "csa_eta":0.021,
                    "spin_rate":12500.0,
                    "crystal_file":'rep100',
                    "gamma_angles":11,
                    "proton_frequency":500.0e6,
                    "sw":2.5e6,
                    "scaling_factor":1.0,
                    "lb": 2500,
                    },
              '2': {
                    "nuclei":'207Pb',
                    "cs_iso":1930-1800,
                    "csa":-876,
                    "csa_eta":0.304,
                    "spin_rate":12500.0,
                    "crystal_file":'rep100',
                    "gamma_angles":11,
                    "proton_frequency":500.0e6,
                    "sw":2.5e6,
                    "scaling_factor":1.0,
                    "lb": 2500,
                    }}

# Simulate only species 2
# Create a dictionary containing the custom parameter of species 2
# input_dict = {"nuclei":'207Pb',
#               "cs_iso":1930-1800,
#               "csa":-842,
#               "csa_eta":0.204,
#               "spin_rate":12500.0,
#               "crystal_file":'rep100',
#               "gamma_angles":11,
#               "proton_frequency":500.0e6,
#               "scaling_factor":1.0,
#               "lb": 2000}

# Create an empty dictionary for custom parts of the scripts. possible keys are: 'spinsys', 'par', 'pulseq', 'main'
# These will override any defaults in that section, but can use the default .format() placeholder e.g. {alpha}
# proc_dict = {}
# proc_dict['pulseq'] = """
#     proc pulseq {{}} {{
#         global par
#         {pulse}
#         acq_block {{
#             delay $par(tsw)
#         }}
#     }}
#     """

# Read-in comparison file
# data, _, _ = processing.read_ascii(output_path+'PbZrO3_mas_scaled_combined.xy', larmor_freq=104.609)
# data, _, _ = processing.read_ascii(output_path+'PbZrO3_mas_scaled_combined.xy', larmor_freq=104.609)
# data, _, _ = processing.read_brukerproc('/home/m_buss13/ownCloud/git/nmr_tools/examples/example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/11')
data, _, _ = processing.read_brukerproc('/home/m_buss13/ownCloud/git/nmr_tools/examples/example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/1')


# Define parameter to be optimized
# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
params_input = (
                ('csa_1', -489, True, -550, -450, None, None),
                ('csa_2', -876, True, -950, -850, None, None),
                ('lb_1', 2000, True, 2000, 3000, None, None),
                ('lb_2', 2000, True, 2000, 3000, None, None),
                ('csa_eta_1', 0.25, True, 0, 0.5, None, None),
                ('csa_eta_2', 0.25, True, 0, 0.5, None, None),
                )

results = simpson.fit_simpson(output_path=output_path,
                              output_name=output_name,
                              params_input=params_input,
                              data=data,
                              input_dict=input_dict,
                              si=8192*4,
                              # proc_dict=proc_dict,
                              verb=True,
                              method='powell',
                            #   **{'finish':'optimize.fmin'},
                            #   **{'xtol':1e-15,
                            #   'ftol':1e-15},
                            #   **{'options':{'xatol':1e-10}}
                              )

# print(fit_report(results))

print("-------------------------------")
print("---%s seconds ---" % (time.time() - start_time))
print("-------------------------------")
