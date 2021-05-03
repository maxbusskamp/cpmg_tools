
#%%
# Example for the different loss function calculations
from nmr_tools import simpson, processing
import matplotlib.pyplot as plt
from subprocess import run

plt.rcParams['figure.dpi'] = 200

output_path = '/home/m_buss13/testfiles/'
output_name = 'simpson_input'
ascii_file = 'simpson_input.xy'

# Simulate species 1
# Create a dictionary containing the custom parameter of species 1
input_dict = {'1': {
                            "nuclei":'207Pb',
                            "cs_iso":1930,
                            "csa":-876,
                            "csa_eta":0.204,
                            "spin_rate":25000.0,
                            "proton_frequency":500.0e6,
                            "crystal_file":'rep10',
                            "gamma_angles":5,
                            "scaling_factor":1.0
                            },
              '2': {
                            "nuclei":'207Pb',
                            "cs_iso":1598,
                            "csa":-489,
                            "csa_eta":0.241,
                            "spin_rate":25000.0,
                            "crystal_file":'rep10',
                            "gamma_angles":5,
                            "proton_frequency":500.0e6,
                            "scaling_factor":1.0
                            }}

# Create an empty dictionary for custom parts of the scripts. possible keys are: 'spinsys', 'par', 'pulseq', 'main'
# These will override any defaults in that section, but can use the default .format() placeholder e.g. {alpha}
proc_dict = {}
proc_dict['pulseq'] = """
    proc pulseq {{}} {{
        global par
        {pulse}
        acq_block {{
            delay $par(tsw)
        }}
    }}
    """

# Read-in comparison file
data, _, _ = processing.read_ascii(output_path+'PbZrO3_mas_scaled_combined.xy', larmor_freq=104.609)

# Define parameter to be optimized
# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
params_input = (('scaling_factor_1', 0.5, True, 0.0, 2.0),
                ('scaling_factor_2', 1.5, True, 0.0, 2.0))

simpson.fit_simpson(output_path=output_path,
                    output_name=output_name,
                    params_input=params_input,
                    data=data,
                    input_dict=input_dict,
                    proc_dict=proc_dict,
                    verb=False)
