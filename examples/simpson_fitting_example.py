#%%
# Example for the fitting of a 2-spin system
from cpmg_tools import simpson, processing
from lmfit import fit_report
import time


start_time = time.time()

output_path = '/home/m_buss13/testfiles/'
output_name = 'simpson_input'
ascii_file = 'simpson_input.xy'

# Create a dictionary containing the custom parameter of both species
input_dict = {'1': {
                    "nuclei":'207Pb',
                    "cs_iso":-218,
                    "csa":-489,
                    "csa_eta":0.021,
                    "spin_rate":12500.0,
                    "crystal_file":'rep30',
                    "gamma_angles":7,
                    "proton_frequency":500.0e6,
                    "sw":2.5e6,
                    "scaling_factor":1.0,
                    "lb": 2300,
                    },
              '2': {
                    "nuclei":'207Pb',
                    "cs_iso":128,
                    "csa":-876,
                    "csa_eta":0.304,
                    "spin_rate":12500.0,
                    "crystal_file":'rep30',
                    "gamma_angles":7,
                    "proton_frequency":500.0e6,
                    "sw":2.5e6,
                    "scaling_factor":1.0,
                    "lb": 2300,
                    }}

# Read-in comparison file
# data, _, _ = processing.read_ascii(output_path+'PbZrO3_mas_scaled_combined.xy', larmor_freq=104.609)
# data, _, _ = processing.read_ascii(output_path+'PbZrO3_mas_scaled_combined.xy', larmor_freq=104.609)
# data, _, _ = processing.read_brukerproc('/home/m_buss13/ownCloud/git/cpmg_tools/examples/example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/11')
data, _, _ = processing.read_brukerproc('/home/m_buss13/ownCloud/git/cpmg_tools/examples/example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/1')


# Define parameter to be optimized
# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
params_input = (
            #     ('cs_iso_1', -200, True, -250, -200, None, None),
            #     ('cs_iso_2', 120, True, 100, 150, None, None),
                ('csa_1', -200, True, -700, -100, None, None),
                ('csa_2', -1100, True, -1200, -700, None, None),
            #     ('lb_1', 2000, True, 2000, 2500, None, None),
            #     ('lb_2', 2000, True, 2000, 2500, None, None),
                ('csa_eta_1', 0.1, True, 0, 0.5, None, None),
                ('csa_eta_2', 0.25, True, 0, 0.5, None, None),
                )

# Specify fitting parameter
results = simpson.fit_simpson(output_path=output_path,
                              output_name=output_name,
                              params_input=params_input,
                              data=data,
                              input_dict=input_dict,
                              si=8192*4,
                              verb=False,
                              method='powell'
                              )

# Print fancy fit report, additionally to the default printed variables
# print(fit_report(results))

print("-------------------------------")
print("---%s seconds ---" % (time.time() - start_time))
print("-------------------------------")
