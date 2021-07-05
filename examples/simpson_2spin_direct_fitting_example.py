#%%
# Example for the fitting of a 2-spin system using the opt package. USE WITH CAUTION! NOT FINISHED!
from cpmg_tools import simpson, processing
from lmfit import fit_report
import time


start_time = time.time()

output_path = 'testfiles/'
output_name = 'simpson_input'
ascii_file = 'simpson_input.xy'

# Create a dictionary containing the custom parameter of both species
input_dict = {
            "nuclei":'207Pb 207Pb',
            "channels":'207Pb',
            "cs_iso1":-218,
            "cs_iso2":128,
            "csa1":-489,
            "csa2":-850,
            "csa_eta1":0.021,
            "csa_eta2":0.24,
            "spin_rate":12500.0,
            "crystal_file":'rep30',
            "gamma_angles":7,
            "proton_frequency":500.0e6,
            "sw":2.5e6,
            "lb": 2200,
            "comp_file":'207Pb_PbZrO3_MAS_WCPMG_1.spe'
            }

# Create an empty dictionary for custom parts of the scripts. possible keys are: 'spinsys', 'par', 'pulseq', 'main'
# These will override any defaults in that section, but can use the default .format() placeholder e.g. {alpha}

# Read-in comparison file
# data, _, _ = processing.read_ascii(output_path+'PbZrO3_mas_scaled_combined.xy', larmor_freq=104.609)
# data, _, _ = processing.read_ascii(output_path+'PbZrO3_mas_scaled_combined.xy', larmor_freq=104.609)
# data, _, _ = processing.read_brukerproc('/home/m_buss13/ownCloud/git/cpmg_tools/examples/example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/11')
data, _, _ = processing.read_brukerproc(r'example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata/1')

# /home/m_buss13/ownCloud/git/cpmg_tools/examples/example_data/207Pb_PbZrO3_MAS_WCPMG/1/pdata
# Define parameter to be optimized
# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
params_input = (
                ('csa1', -200, True, -700, -100, None, None),
                ('csa2', -1100, True, -1200, -700, None, None),
                ('csa_eta1', 0.1, True, 0, 0.5, None, None),
                ('csa_eta2', 0.25, True, 0, 0.5, None, None),
                )

# Specify fitting parameter
simpson.create_simpson_fit(output_path=output_path,
                            output_name=output_name,
                            input_dict=input_dict
                            )

# Print fancy fit report, additionally to the default printed variables
# print(fit_report(results))

print("-------------------------------")
print("---%s seconds ---" % (time.time() - start_time))
print("-------------------------------")
