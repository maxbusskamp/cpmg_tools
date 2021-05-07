import os
from subprocess import run
from nmr_tools import processing
import numpy as np
import lmfit
from collections import defaultdict
import sys

def constant_factory(value):

    return lambda: value



def create_simpson(output_path, output_name, input_dict=None, proc_dict=None, params_scaling={}):  # Write simpson input files
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
                    if keys in params_scaling:
                        if subkeys in params_scaling[keys]:
                            simpson_variables[keys][subkeys] = input_dict[keys][subkeys]*params_scaling[keys][subkeys]

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
                if keys in params_scaling:
                    simpson_variables[keys] = input_dict[keys]*params_scaling[keys]
                else:
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

    # TODO: REMOVE INDENTATION OF MULTILINE STRINGS IN FINAL VERSION
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
        counter = 0
        for keys in simpson_variables:
            with  open(output_path + output_name,'w') as myfile:
                myfile.write(''.join(simpson.values()).format(**simpson_variables[keys]))

            os.chdir(output_path)
            run(['simpson', output_name])
            data, timescale = processing.read_ascii_fid(output_path + output_name + '.xy')
            if counter == 0:
                data = data*simpson_variables[keys]['scaling_factor']
            else:
                data = data*simpson_variables[keys]['scaling_factor'] + data_temp
            data_temp = data
            counter += 1
    else:
        with  open(output_path + output_name,'w') as myfile:
                myfile.write(''.join(simpson.values()).format(**simpson_variables))
        os.chdir(output_path)
        run(['simpson', output_name])
        data, timescale = processing.read_ascii_fid(output_path + output_name + '.xy')
    
    return (data, timescale)


def run_simpson(input_file, working_dir, *args):

    os.chdir(working_dir)
    run(['simpson', input_file, *args])

    return()


def fit_helper(params, data, output_path, output_name, input_dict, proc_dict, params_scaling, si):


    for keys in params.valuesdict():
        input_dict[str(keys.rsplit('_', 1)[-1])][str(keys.rsplit('_', 1)[0])] = params.valuesdict()[keys]

    params_scaling_input = defaultdict(lambda: defaultdict(constant_factory(1.0)))
    for keys in params_scaling:
        params_scaling_input[str(keys.rsplit('_', 1)[-1])][str(keys.rsplit('_', 1)[0])] = params_scaling[keys]

    # print(input_dict)
    data_model, timescale_model  = create_simpson(output_path, output_name, input_dict=input_dict, proc_dict=proc_dict, params_scaling=params_scaling_input)
    if(len(data_model)>si):
        si = len(data_model)
        print('SI value to low. Value was corrected to: ' + str(si))
    data_model_fft, _ = processing.asciifft(data_model, timescale_model, si=si)
    
    # Scaling
    data_model_fft = data_model_fft/max(data_model_fft.real)
    data = data/max(data.real)

    # Check SI compliance
    if(len(data_model_fft)!=len(data)):
        sys.exit('Model and comparison data are not of same length. Check SI value. ' + str(len(data_model_fft)) + ' vs. ' + str(len(data)))

    # print(processing.calc_logcosh(data_model_fft.real, data.real))
    # print(processing.calc_mse(data_model_fft.real, data.real))
    # return(data_model_fft.real - data.real)
    return(processing.calc_logcosh(data_model_fft.real, data.real))


def fit_simpson(output_path, output_name, params_input, data, si, input_dict=None, proc_dict=None, verb=True, method='leastsq', **fit_kws):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.chdir(output_path)

    params = lmfit.Parameters()
    params_scaling = {}
    if any(isinstance(i,tuple) for i in params_input):
        for elem in params_input:
            name, value, vary, valmin, valmax, expr, brute_step = elem
            scalefactor = 1
            if value:
                if(abs(value)>scalefactor):
                    scalefactor = abs(value)
            if valmin != -np.inf and valmin != None:
                if(abs(valmin)>scalefactor):
                    scalefactor = abs(valmin)
            if valmax != np.inf and valmax != None:
                if(abs(valmax)>scalefactor):
                    scalefactor = abs(valmax)
            if value:
                value = value/scalefactor
            if valmin != -np.inf and valmin != None:
                valmin = valmin/scalefactor
            if valmax != np.inf and valmax != None:
                valmax = valmax/scalefactor
            if brute_step:
                brute_step = brute_step/scalefactor
            params.add(*(name, value, vary, valmin, valmax, expr, brute_step))
            params_scaling[name] = scalefactor
    else:
        name, value, vary, valmin, valmax, expr, brute_step = params_input
        # print(name, value, vary, valmin, valmax, expr, brute_step)
        scalefactor = 1
        if value:
            if(abs(value)>scalefactor):
                scalefactor = abs(value)
        if valmin != -np.inf and valmin != None:
            if(abs(valmin)>scalefactor):
                scalefactor = abs(valmin)
        if valmax != np.inf and valmax != None:
            if(abs(valmax)>scalefactor):
                scalefactor = abs(valmax)
        if value:
            value = value/scalefactor
        if valmin != -np.inf and valmin != None:
            valmin = valmin/scalefactor
        if valmax != np.inf and valmax != None:
            valmax = valmax/scalefactor
        if brute_step:
            brute_step = brute_step/scalefactor
        # print(name, value, vary, valmin, valmax, expr, brute_step)
        params.add(*(name, value, vary, valmin, valmax, expr, brute_step))
        params_scaling[name] = scalefactor
    if verb:
        print('This are the parameters you passed for optimization, after scaling:')
        params.pretty_print()
        print('Scale parameters are:')
        print(params_scaling)

    out = lmfit.minimize(fit_helper, params, args=(data, output_path, output_name, input_dict, proc_dict, params_scaling, si), method=method, **fit_kws)

    print('--------------------------------------')
    print('These are the resulting fit parameter:')
    print('--------------------------------------')
    print('Parameter    Value    Min    Max    Stderr')
    for name, param in out.params.items():
        print('{:7s} {:11.5f} {:11.5f} {:11.5f} {}'.format(name, param.value*params_scaling[name], param.min*params_scaling[name], param.max*params_scaling[name], param.stderr))

    return(out)
