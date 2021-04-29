import os
from subprocess import run
from typing import DefaultDict
from nmr_tools import processing
import numpy as np

def create_simpson(output_path, output_name, input_dict=None, proc_dict=None):  # Write simpson input files
    """This generates a custom Simpson inputfile, which can be used from the terminal with: 'simpson <output_name>'

    Args:
        output_path (str): Path to save the generated file
        output_name (str): Name of generated inputfile
        input_dict (dict): Used to specify simulation parameter.
            nuclei:'1H',
            cs_iso:0.0,
            csa:0.0,
            csa_eta:0.0,
            alpha:0.0,
            beta:0.0,
            gamma:0.0,
            sw (float): Spectral width in Hz
            np (float): Number of FID points
            spin_rate (float): MAS rate in Hz
            proton_frequency (float): Proton frequency in Hz
            crystal_file (str, optional): Crystal file chosen from e.g. rep20 rep2000 rep256 rep678 zcw232 zcw4180 zcw28656. Defaults to 'rep2000'.
            gamma_angles (int, optional): Gamma angles, should be at least sqrt(crystal_file orientations) Defaults to 45.
            lb (int, optional): Linebroadening in Hz. Defaults to 1000.
            lb_ratio (float, optional): Ratio between Gauss/Lorentz linebroadening. Defaults to 1.0.
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    simpson_variables = {}
    if input_dict is not None:
        if any(isinstance(i,dict) for i in input_dict.values()):
            for keys in input_dict:
                simpson_variables[keys] = {}

            for keys in input_dict:
                simpson_variables[keys] = {
                    "nuclei":'1H',
                    "cs_iso":0.0,
                    "csa":0.0,
                    "csa_eta":0.0,
                    "alpha":0.0,
                    "beta":0.0,
                    "gamma":0.0,
                    "pulse":'',
                    "spin_rate":0.0,
                    "proton_frequency":500.0e6,
                    "start_operator":'Inx',
                    "detect_operator":'Inp',
                    "crystal_file":'rep100',
                    "gamma_angles":10,
                    "method":'direct',
                    "sw":2e6,
                    "np":16384,
                    "lb":1000,
                    "lb_ratio":1.0,
                    "si":'np*2',
                    "scaling_factor":1.0,
                    "output_name":output_name}
            for keys in input_dict:
                for subkeys in input_dict[keys]:
                    simpson_variables[keys][subkeys] = input_dict[keys][subkeys]
        else:
            simpson_variables = {
                "nuclei":'1H',
                "cs_iso":0.0,
                "csa":0.0,
                "csa_eta":0.0,
                "alpha":0.0,
                "beta":0.0,
                "gamma":0.0,
                "pulse":'',
                "spin_rate":0.0,
                "proton_frequency":500.0e6,
                "start_operator":'Inx',
                "detect_operator":'Inp',
                "crystal_file":'rep100',
                "gamma_angles":10,
                "method":'direct',
                "sw":2e6,
                "np":16384,
                "lb":1000,
                "lb_ratio":1.0,
                "si":'np*2',
                "scaling_factor":1.0,
                "output_name":output_name}
            for keys in input_dict:
                simpson_variables[keys] = input_dict[keys]
    else:
        simpson_variables = {
                    "nuclei":'1H',
                    "cs_iso":0.0,
                    "csa":0.0,
                    "csa_eta":0.0,
                    "alpha":0.0,
                    "beta":0.0,
                    "gamma":0.0,
                    "pulse":'',
                    "spin_rate":0.0,
                    "proton_frequency":500.0e6,
                    "start_operator":'Inx',
                    "detect_operator":'Inp',
                    "crystal_file":'rep100',
                    "gamma_angles":10,
                    "method":'direct',
                    "sw":2e6,
                    "np":16384,
                    "lb":1000,
                    "lb_ratio":1.0,
                    "si":'np*2',
                    "scaling_factor":1.0,
                    "output_name":output_name}

    simpson = {'spinsys': '', 'par': '', 'pulseq': '', 'main': ''}

    simpson['spinsys'] = """
spinsys {{
    channels {nuclei}
    nuclei {nuclei}
    shift 1 {cs_iso}p {csa}p {csa_eta} {alpha} {beta} {gamma}
}}
"""

    simpson['par'] = """
par {{
    spin_rate        {spin_rate}
    proton_frequency {proton_frequency}
    start_operator   {start_operator}
    detect_operator  {detect_operator}
    method           {method}
    crystal_file     {crystal_file}
    gamma_angles     {gamma_angles}
    sw               {sw}
    variable tsw     1e6/sw
    verbose          0000
    np               {np}
    variable si      {si}
}}
"""

    simpson['pulseq'] = """
proc pulseq {{}} {{
    global par
    {pulse}
    acq_block {{
        delay $par(tsw)
    }}
}}
"""

    simpson['main'] = """
proc main {{}} {{
    global par

    set f [fsimpson]

    faddlb $f {lb} {lb_ratio}

    fsave $f {output_name}.xy -xreim
}}
"""

    if proc_dict is not None:
        for keys in proc_dict:
            simpson[keys] = proc_dict[keys]

    if any(isinstance(i,dict) for i in simpson_variables.values()):
        data_temp = np.empty(1)
        for keys in simpson_variables:
            with  open(output_path + output_name,'w') as myfile:
                myfile.write(''.join(simpson.values()).format(**simpson_variables[keys]))

            os.chdir(output_path)
            run(['simpson', output_name])
            timescale, data = processing.read_ascii_fid(output_path + output_name + '.xy')
            if data_temp.any():
                data = data*simpson_variables[keys]['scaling_factor'] + data_temp
            else:
                data = data*simpson_variables[keys]['scaling_factor']
            data_temp = data
    else:
        with  open(output_path + output_name,'w') as myfile:
                myfile.write(''.join(simpson.values()).format(**simpson_variables))
        os.chdir(output_path)
        run(['simpson', output_name])
        timescale, data = processing.read_ascii_fid(output_path + output_name + '.xy')
    
    return (timescale, data)


def run_simpson(input_file, working_dir, *args):

    os.chdir(working_dir)

    run(['simpson', input_file])
