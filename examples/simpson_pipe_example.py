#%%
from nmr_tools import simpson

output_path = '/home/m_buss13/'
output_name = 'simpson_input.tcl'

simpson.create_simpson(output_path, output_name, spin_rate=25000)
