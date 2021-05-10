import numpy as np
from scipy import interpolate
from scipy.integrate import simps
import os, sys
from scipy.optimize.optimize import brute
from scipy.optimize import basinhopping
from scipy.optimize import minimize
from nmr_tools import bruker, fileiobase, proc_base


def read_brukerproc(datapath, dict=False):
    """
    This reads in an Bruker processed dataset.

    Args:
        datapath (str): Path string to the Bruker Dataset, ending with '/pdata/1'
        dict (bool, optional): Set True to export the dictionary. Defaults to False.

    Returns:
        ppm_scale (1darray)
        hz_scale (1darray)
        data (ndarray)
        dic (dictionary, optional)
    """


    dic, data = bruker.read_pdata(datapath)
    udic = bruker.guess_udic(dic, data)
    uc = fileiobase.uc_from_udic(udic)
    ppm_scale = uc.ppm_scale()
    hz_scale = uc.hz_scale()

    return (data, ppm_scale, hz_scale, dic) if dict==True else (data, ppm_scale, hz_scale)


def read_brukerfid(datapath, dict=False):
    """
    This reads in an Bruker FID. Zerofilling is removed based on 's TD'.

    Args:
        datapath (str): Path string to the Bruker Dataset, ending with '/pdata/1'
        dict (bool, optional): Set True to export the dictionary. Defaults to False.

    Returns:
        ppm_scale (1darray)
        hz_scale (1darray)
        data (ndarray)
        dic (dictionary, optional)
    """


    dic, data = bruker.read(datapath)
    td = int(dic['acqus']['TD']/2)
    sw_h = float(dic['acqus']['SW_h'])
    dw = 1.0/(sw_h*2.0)
    timescale = np.linspace(0, 0+(dw*td),td,endpoint=False)
    data = data[:td]


    return (data, timescale, dic) if dict==True else (data, timescale)


def read_ascii(datapath, larmor_freq=0.0, skip_header=0, skip_footer=0, delimiter=' '):
    """
    This reads in an ASCII dataset. If provided with the larmor frequency of the observed nuclei, the ppm scale will be calculated

    Args:
        datapath ([type]): Path to datafile
        larmor_freq (float, optional): larmor frequency of observed nuclei. Defaults to 0.0.
        skip_header (int, optional): Skip header lines. Defaults to 0.
        skip_footer (int, optional): Skip footer lines. Defaults to 0.
        delimiter (str, optional): Delimiter. Defaults to ' '.

    Returns:
        ppm_scale (1darray, optional)
        hz_scale (1darray)
        data (ndarray)
    """

    data_temp = np.genfromtxt(datapath, delimiter=delimiter, skip_header=skip_header, skip_footer=skip_footer)

    data = data_temp[:, 1:]
    hz_scale = data_temp[:, 0]
    if(larmor_freq!=0.0):
        ppm_scale = hz_scale/larmor_freq

    data = np.transpose(data)
    data = data[0] + data[1]*1.0j

    return (data, hz_scale) if larmor_freq==0.0 else (data, ppm_scale, hz_scale)


def read_ascii_fid(datapath, skip_header=0, skip_footer=0, delimiter=' '):
    """
    This reads in an ASCII FID and returns timescale and data

    Args:
        datapath ([type]): Path to datafile
        larmor_freq (float, optional): larmor frequency of observed nuclei. Defaults to 0.0.
        skip_header (int, optional): Skip header lines. Defaults to 0.
        skip_footer (int, optional): Skip footer lines. Defaults to 0.
        delimiter (str, optional): Delimiter. Defaults to ' '.

    Returns:
        timescale (1darray)
        data (ndarray)
    """

    data_temp = np.genfromtxt(datapath, delimiter=delimiter, skip_header=skip_header, skip_footer=skip_footer)

    data = data_temp[:, 1]+data_temp[:, 2]*1.0j
    timescale = data_temp[:, 0]

    return (data, timescale)


def read_spe(datapath, larmor_freq=0.0):
    """
    This reads in an ASCII dataset. If provided with the larmor frequency of the observed nuclei, the ppm scale will be calculated

    Args:
        datapath ([type]): Path to datafile
        larmor_freq (float, optional): larmor frequency of observed nuclei. Defaults to 0.0.

    Returns:
        ppm_scale (1darray, optional)
        hz_scale (1darray)
        data (ndarray)
    """

    def lines_that_contain(string, fp):
        return [line for line in fp if string in line]


    #Read npoints from shapefile

    with open(datapath,'r') as myfile:
        head = [next(myfile) for x in range(20)]

    npoints = int(''.join(filter(str.isdigit, str(lines_that_contain('NP=', head)))))
    spectral_width = int(''.join(filter(str.isdigit, str(lines_that_contain('SW=', head)))))

    for num, line in enumerate(head, 1):
        if 'DATA' in line:
            header_count = num

    data = np.genfromtxt(datapath, delimiter=' ', skip_header=header_count, skip_footer=1)

    hz_scale = np.linspace(-spectral_width/2.0, spectral_width/2.0, npoints)
    if(larmor_freq!=0.0):
        ppm_scale = hz_scale/larmor_freq

    return (data, hz_scale) if larmor_freq==0.0 else (data, ppm_scale, hz_scale)


def get_envelope_idx(s, dmin=1, dmax=1, split=False):
    """[summary]

    Args:
        s ([type]): 1d-array, data signal from which to extract high and low envelopes
        dmin (int, optional): size of chunks, use this if the size of the input signal is too big. Defaults to 1.
        dmax (int, optional): size of chunks, use this if the size of the input signal is too big. Defaults to 1.
        split (bool, optional): if True, split the signal in half along its mean, might help to generate the envelope in some cases. Defaults to False.
    
    Output :
        lmin,lmax : high/low envelope idx of input signal s
    """

    # locals min      
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1 
    # locals max
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1 
    

    if split:
        # s_mid is zero if s centered around x-axis or more generally mean of signal
        s_mid = np.mean(s) 
        # pre-sorting of locals min based on relative position with respect to s_mid 
        lmin = lmin[s[lmin]<s_mid]
        # pre-sorting of local max based on relative position with respect to s_mid 
        lmax = lmax[s[lmax]>s_mid]


    # global max of dmax-chunks of locals max 
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0,len(lmin),dmin)]]
    # global min of dmin-chunks of locals min 
    lmax = lmax[[i+np.argmax(s[lmax[i:i+dmax]]) for i in range(0,len(lmax),dmax)]]
    
    return(lmin, lmax)


def save_xri(output_path, output_name, dataset_array):
    """
    Saves numpy array as ASCII file

    Args:
        output_path (str): path to output
        output_name (str): name of output
        dataset_array (np array): a numpy array containing freq., real, imag. columns
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    f = open(output_path + output_name + '.dat', 'w')
    np.savetxt(f, dataset_array, delimiter=' ')
    f.close()

    return()


def save_spe(output_path, output_name, dataset_array):
    """
    Saves numpy array as SPE file for use in Simpson

    Args:
        output_path (str): path to output
        output_name (str): name of output
        dataset_array (np array): a numpy array containing freq., real, imag. columns
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    f = open(output_path + output_name + '.spe', 'w')
    f.write('SIMP\n')
    f.write('NP='+str(int(len(dataset_array)))+'\n')
    f.write('SW='+str(np.round((abs(dataset_array[-1,0])+abs(dataset_array[0,0])), -3))+'\n')
    f.write('REF=0.0\n')
    f.write('TYPE=SPE\n')
    f.write('DATA\n')
    np.savetxt(f, dataset_array[:,1:], delimiter=' ')
    f.write('END')
    f.close()

    return()


def find_nearest_index(array, value):

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    
    return idx


def find_nearest(array, value):

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return array[idx]


def shift_bit_length(x):
    return 1<<(x-1).bit_length()


def combine_stepped_aq(datasets, set_sw=0, precision_multi=1, mode='skyline', sum_tol=1.0, verbose=False, bins=1000):
    """
    This combines multiple Bruker Datasets into one spectrum, using a calculation of the envelope to determine the highest x-values, if multiple exist.

    Args:
        datasets (str): String list containing the Bruker Dataset Paths ['', '', '']
        set_sw (int, optional): Spectral width is set to +-set_sw/2
        precision_multi (int, optional): Multiplier for the interpolation step. Enhances precision for .spe files.
        verbose (bool, optional): If True, activates debug output
    """
    if(mode=='skyline'):
        # Combine Data
        index = 0
        for datapath in datasets:
            if(isinstance(datapath, str)):
                data_temp, ppm_temp, hz_temp = read_brukerproc(datapath)
            elif(isinstance(datapath, (np.ndarray, np.generic)) or isinstance(datapath, (tuple))):
                data_temp, ppm_temp, hz_temp = datapath
            else:
                print('Wrong input format. Use list of datapath strings, or list of ndarrays.')

            dataset = np.zeros([len(data_temp),3])
            dataset[:,0] = data_temp
            dataset[:,1] = ppm_temp
            dataset[:,2] = hz_temp

            if(index == 0):
                dataset_combine = dataset
            else:
                dataset_combine = np.vstack((dataset_combine, dataset))
            index = index + 1

        # Sort Data
        dataset_combine = dataset_combine[dataset_combine[:,2].argsort(),:]
        # Generate Envelope
        _, low_idx = get_envelope_idx(dataset_combine[:,0], dmin=4, dmax=4)
        dataset_array_masked = dataset_combine[low_idx,0]
        dataset_x_masked = dataset_combine[low_idx,2]

        # Generate XRI Dataset
        dataset_new = []
        index = 0
        for elem in dataset_x_masked:
            elem = np.append(elem, dataset_array_masked[index])
            elem = np.append(elem, 0)
            dataset_new.append(elem)
            index = index +1
        dataset_array = np.array(dataset_new)

        # Interpolate Values for equidistant .spe generation
        interpolation_points = shift_bit_length(int(len(dataset_array)))*precision_multi
        f = interpolate.interp1d(dataset_array[:,0], dataset_array[:,1], fill_value='extrapolate')
        xnew = np.linspace(dataset_array[0,0], dataset_array[-1,0], interpolation_points)
        dataset_interpol = f(xnew)
        if(verbose==True):
            print('interpolation_points: ' + str(interpolation_points))


        # Generate XRI Dataset
        dataset_new = []
        index = 0
        for elem in xnew:
            elem = np.append(elem, dataset_interpol[index])
            elem = np.append(elem, 0)
            dataset_new.append(elem)
            index = index +1
        dataset_array = np.array(dataset_new)
    elif(mode=='sum'):
        # Combine Data
        index = 0
        datasets_temp = []
        for datapath in datasets:
            if(isinstance(datapath, str)):
                dataset = read_brukerproc(datapath)
                dataset = np.array(dataset)
            elif(isinstance(datapath, (np.ndarray, np.generic)) or isinstance(datapath, (tuple))):
                dataset = datapath
                dataset = np.array(dataset)
            else:
                print('Wrong input format. Use list of datapath strings, or list of ndarrays.')
            datasets_temp.append(dataset)

        index = 0
        for datasets in datasets_temp:
            if(index == 0):
                dataset_scale_hz = datasets[2,:] 
                dataset_scale_ppm = datasets[1,:]
            else:
                dataset_scale_hz = np.hstack((dataset_scale_hz, datasets[2,:]))
                dataset_scale_ppm = np.hstack((dataset_scale_ppm, datasets[1,:]))
            index = index + 1

            # Sort Data
            dataset_scale_hz = np.sort(dataset_scale_hz)
            dataset_scale_ppm = np.sort(dataset_scale_ppm)

        datasets_interpolated = []
        index = 0
        for datasets in datasets_temp:
            f = interpolate.interp1d(datasets[2,:], datasets[0,:], fill_value=0.0, bounds_error=False)
            datasets_interpolated.append(f(dataset_scale_hz))
            index+=1
        sumdata = [sum(elem) for elem in zip(*datasets_interpolated)]

        dataset_new = []
        index = 0
        for elem in dataset_scale_hz:
            elem = np.append(elem, sumdata[index])
            elem = np.append(elem, 0)
            dataset_new.append(elem)
            index = index +1
        dataset_array = np.array(dataset_new)
    else:
        sys.exit('Wrong mode! Please choose between skyline or sum')

    # Restrict spectral width
    if(set_sw!=0):
        index1 = find_nearest_index(dataset_array[:,0], -set_sw/2)
        index2 = find_nearest_index(dataset_array[:,0], set_sw/2)
        dataset_array = dataset_array[index1:index2,:]

    # Interpolate Values for equidistant .spe generation and a power of 2 number of points
    interpolation_points_power2 = shift_bit_length(int(len(dataset_array)))
    f = interpolate.interp1d(dataset_array[:,0], dataset_array[:,1], fill_value='extrapolate')
    xnew = np.linspace(dataset_array[0,0], dataset_array[-1,0], interpolation_points_power2)
    dataset_interpol = f(xnew)
    if(verbose==True):
        print('interpolation_points for power of 2 interpolation: ' + str(interpolation_points_power2))

    # Generate XRI Dataset
    dataset_new = []
    index = 0
    for elem in xnew:
        elem = np.append(elem, dataset_interpol[index])
        elem = np.append(elem, 0)
        dataset_new.append(elem)
        index = index +1
    data = np.array(dataset_new)

    if(verbose==True):
        precision = np.round((abs(data[-1,0])+abs(data[0,0])), -3)-(abs(data[-1,0])+abs(data[0,0]))
        print('.spe precision: ' + str(int(precision)) + ' Hz')
        print('SW raw: ' + str(int(abs(data[-1,0])+abs(data[0,0]))) + ' Hz')
        print('SW rounded: ' + str(int(np.round((abs(data[-1,0])+abs(data[0,0])), -3))) + ' Hz')

    return(data)


def split_echotrain(datapath, dw, echolength, blankinglength, numecho, dict=False):
    """This splits the echo train of a given Bruker Dataset and coadds each echo.

    Args:
        datapath (str): Path string to the Bruker Dataset, ending with '/pdata/1'
        dw (float): Dwell time
        echolength (float): length of aquisition
        blankinglength (float): Length of blank time between aquisitions
        numecho (int): Number of recorded echos (e.g. l22)
        dict (bool, optional): Set True to export the dictionary. Defaults to False.

    Return:
        ppm_scale
        hz_scale
        data
    """  

    data, timescale, dic = read_brukerfid(datapath, dict=True)

    echopoints = int(echolength/dw/2)
    echotop = np.argmax(np.absolute(data))
    echostart = int(echotop-echopoints/2)

    blankingpoints = int(blankinglength/dw/2)
    fullechopoints = int(echopoints + blankingpoints)

    data = data[echostart:(int(echopoints+blankingpoints)*numecho+echostart)]

    data = np.array_split(data, numecho)
    data = np.vstack(data)
    data = np.delete(data, np.s_[echopoints:fullechopoints], axis=1)

    # Superposition
    data_sum = np.zeros((len(data),len(data[0])),dtype=(data.dtype))
    for i in range(numecho):
        data_sum[i].real = data[i].real + data[i][::-1].real
        data_sum[i].imag = 1.0*data[i].imag + -1.0*data[i][::-1].imag
    data_sum = np.delete(data_sum, np.s_[0:int(echopoints/2)], axis=1)
    data_sum = data_sum.sum(axis=0)

    # Echo Sum
    data = data.sum(axis=0)

    return (data, timescale, dic) if dict==True else (data, timescale)


def calc_mse(data_1, data_2):
    """
    This calculates the Mean Square Error loss function of two 1d arrays.

    Args:
        data_1 (1darray): First data array to be evaluated
        data_2 (1darray): Second data array to be evaluated
    """

    data_rms = data_1 - data_2
    rms = np.mean(data_rms**2)

    return (rms) if rms<1.0e+100 else (1.0e+100)


def calc_residual(data_1, data_2):
    """
    This calculates the Mean Square Error loss function of two 1d arrays.

    Args:
        data_1 (1darray): First data array to be evaluated
        data_2 (1darray): Second data array to be evaluated
    """

    return (data_1 - data_2)


def calc_mae(data_1, data_2):
    """
    This calculates the Mean Absolute Error loss function of two 1d arrays.

    Args:
        data_1 (1darray): First data array to be evaluated
        data_2 (1darray): Second data array to be evaluated
    """

    data_rms = data_1 - data_2
    rms = np.sqrt(np.mean(data_rms**2))

    return(rms)


def calc_logcosh(data_1, data_2):
    """
    This calculates the Log-COSH loss function of two 1d arrays.

    Args:
        data_1 (1darray): First data array to be evaluated
        data_2 (1darray): Second data array to be evaluated
    """

    data_rms = data_1 - data_2
    rms = np.sum(np.log(np.cosh(data_rms)))/1000.0

    return(rms)


def calc_int_sum(data, data_mc, int_sum_cutoff):
    index = np.where(data.real > int_sum_cutoff*np.std(data_mc.real))
    index_sub = np.where(data.real < -int_sum_cutoff*np.std(data_mc.real))
    # integral = simps(data.real[index])
    integral = abs(simps(data.real[index])) - abs(simps(data.real[index_sub]))
    # integral = abs(simps(data.real)) - abs(simps(data.real))
    # integral = simps(data.real)
    return(integral)


def data_rms(x, data, data_mc, loss_func, int_sum_cutoff):
    if(len(x)==2):
        data_phased = proc_base.ps(data, p0=x[0], p1=x[1])      # phase correction
    elif(len(x)==3):
        data_phased = proc_base.ps2(data, p0=x[0], p1=x[1], p2=x[2])      # phase correction
    else:
        sys.exit('Wrong number of boundary conditions! Please set only 2 or 3 conditions')

    data_phased = proc_base.di(data_phased)
    data_mc = proc_base.di(data_mc)

    if(loss_func=='logcosh'):
        rms = calc_logcosh(data_mc, data_phased)
    elif(loss_func=='mse'):
        rms = calc_mse(data_mc, data_phased)
    elif(loss_func=='mae'):
        rms = calc_mae(data_mc, data_phased)
    elif(loss_func=='int_sum'):
        rms = -1.0*calc_int_sum(data_phased, data_mc, int_sum_cutoff)
    else:
        print('Wrong loss function')

    return rms


def phase_minimizer(data, data_mc, bnds, Ns, loss_func, int_sum_cutoff, workers, verb, minimizer, tol, options, stepsize, T, disp, niter):
    resbrute = brute(data_rms, ranges=bnds, args=(data, data_mc, loss_func, int_sum_cutoff,), Ns=Ns, disp=True, workers=workers, finish=None)
    if verb:
        print('Brute-Force Optmization Results:')
        print(resbrute)
    if(minimizer=='Nelder-Mead'):
        res = minimize(data_rms, x0 = resbrute, args=(data, data_mc, loss_func, int_sum_cutoff,), method='Nelder-Mead', tol=tol, options=options)
    elif(minimizer=='COBYLA'):
        res = minimize(data_rms, x0 = resbrute, args=(data, data_mc, loss_func, int_sum_cutoff,),method='COBYLA', tol=tol, options=options)
    elif(minimizer=='basinhopping'):
        minimizer_kwargs={"method":"Nelder-Mead", "args":(data, data_mc, loss_func, int_sum_cutoff,)}
        res = basinhopping(data_rms, x0 = resbrute, minimizer_kwargs=minimizer_kwargs, stepsize=stepsize, T=T, disp=disp, niter=niter)
    else:
        sys.exit('Wrong minimizer! Choose from: Nelder-Mead, COBYLA, or basinhopping')
    if verb:
        print(minimizer + ' Results:')
        print(res)

    return res.x


def autophase(data, bnds=((-360, 360), (0, 200000), (0, 200000)), Ns=50, int_sum_cutoff=1.0, zf=0, verb=False, minimizer='basinhopping', tol=1e-14, options={'rhobeg':1000.0, 'maxiter':5000, 'maxfev':5000}, stepsize=1000, T=100, disp=True, niter=200, loss_func='logcosh', workers=4):
    """
    !!!WIP!!!
    This automatically calculates the phase of the spectrum

    Args:
        data ([type]): [description]
        bnds (tuple, optional): [description]. Defaults to ((-360, 360), (0, 200000)).
        SI (int, optional): [description]. Defaults to 32768.
        Ns (int, optional): [description]. Defaults to 100.
        verb (bool, optional): [description]. Defaults to False.
        loss_func (str, optional): [description]. Defaults to 'logcosh'.
    """

    data_fft = proc_base.fft(proc_base.rev(proc_base.zf(data, pad=zf)))    # Fourier transform
    data_mc = proc_base.mc(data_fft)      # magnitude mode

    # Phasing
    if(type(bnds[0])==int):
        sys.exit('Wrong number of boundary conditions! Please set only 2 or 3 conditions')
    if(len(bnds)==2):
        phase = phase_minimizer(data_fft, data_mc, bnds=bnds, Ns=Ns, loss_func=loss_func, int_sum_cutoff=int_sum_cutoff, workers=workers, verb=verb, minimizer=minimizer, tol=tol, options=options, stepsize=stepsize, T=T, disp=disp, niter=niter)      # automatically calculate phase
        data = proc_base.ps(data_fft, p0=phase[0], p1=phase[1])      # add previously phase values
    elif(len(bnds)==3):
        phase = phase_minimizer(data_fft, data_mc, bnds=bnds, Ns=Ns, loss_func=loss_func, int_sum_cutoff=int_sum_cutoff, workers=workers, verb=verb, minimizer=minimizer, tol=tol, options=options, stepsize=stepsize, T=T, disp=disp, niter=niter)      # automatically calculate phase
        data = proc_base.ps2(data_fft, p0=phase[0], p1=phase[1], p2=phase[2])      # add previously phase values
    else:
        sys.exit('Wrong number of boundary conditions! Please set only 2 or 3 conditions')

    return(data, phase)


def linebroadening(data, lb_variant, lb_const=0.54, lb_n=2):
    """
    This applies linebroadening to a 1darray containing FID data as generated by read_brukerfid().

    Args:
        data (1darray): 1darray containing FID data as generated by read_brukerfid()
        lb_variant (str): Window function (WF) used. Use: hamming, shifted_wurst, gaussian
        lb_const (float, optional): Used for hamming WF. Corresponds to the lowest y value on either side. Defaults to 0.54.
        lb_n (int, optional): Shape parameter used by shifted_wurst and hamming (set to 2). Defaults to 2.

    Returns:
        data (1darray): Linebroadened FID
        y_range (1darray): Y values of window function
    """
    x_range = np.linspace(0, 1, len(data))    # Calculate x values from 0 to 1
    # Calculate y value depending on chosen window function
    if(lb_variant=='hamming'):
        y_range = lb_const+(1-lb_const)*(1-np.power(abs(np.cos(np.pi*x_range)), lb_n))
    elif(lb_variant=='shifted_wurst'):
        y_range = 1-np.power(abs(np.cos(np.pi*x_range)), lb_n)+lb_const
        y_range[y_range > 1.0] = 1.0
    elif(lb_variant=='gaussian'):
        y_range = 1/np.exp(10*np.power((x_range-0.5), 2))
    elif(lb_variant=='gaussian_normal'):
        y_range = 1/np.exp(10*np.power((x_range), 2))
    else:
        sys.exit('Wrong window function! Choose from: hamming, shifted_wurst, gaussian, gaussian_normal')
    data = np.multiply(y_range, data)    # Apply linebroadening

    return(data, y_range)


def signaltonoise(a, axis=0, ddof=0):
    """
    Calculates the signal-to-noise ratio from the mean and std of the whole spectrum. Not sure if correct or comparable.

    Args:
        a (ndarray): Processed dataset
        axis (int, optional): Axis which to look at. Defaults to 0.
        ddof (int, optional): Degree of freedom for std calculation. Defaults to 0.

    Returns:
        sino (float): Signal-to-noise
    """
    a = np.asanyarray(a)
    sino = np.sqrt(np.power(a.mean(axis), 2))/a.std(axis=axis, ddof=ddof)

    return sino


def get_scale(data, dic):
    """
    This generates the ppm and hz scales from a processed dataset. Can be used of the number of points was changed by e.g. zerofilling.

    Args:
        data (ndarray complex): Processed dataset
        dic (dict): Bruker dictionary
    """
    udic = bruker.guess_udic(dic, data)
    uc = fileiobase.uc_from_udic(udic)
    ppm_scale = uc.ppm_scale()
    hz_scale = uc.hz_scale()

    return(ppm_scale, hz_scale)


def fft(data, dic, si=0, mc=True, phase=[0, 0], dict=False):
    """
    This takes the output of read_brukerfid (FID and dic) and applies fft and zerofilling. It can return magnitude or phased data.

    Args:
        data (ndarray complex): FID data
        dic (dict): Bruker dictionary
        si (int, optional): Number of points to zerofill. Defaults to 0.
        mc (bool, optional): Set to False for phased data. Defaults to True.
        dict (bool, optional): Set to True to return the dictionary. Defaults to False.
    """
    data = proc_base.zf_size(data, si)
    data = proc_base.fft(data)
    if(mc==True):
        data = proc_base.mc(data)
    else:
        data = proc_base.ps(data, p0=phase[0], p1=phase[1])

    udic = bruker.guess_udic(dic, data)
    uc = fileiobase.uc_from_udic(udic)

    ppm_scale = uc.ppm_scale()
    hz_scale = uc.hz_scale()

    return(data, ppm_scale, hz_scale, dic) if dict==True else (data, ppm_scale, hz_scale)


def asciifft(data, timescale, si=0, larmor_freq=0.0):
    """
    This takes the output of read_ascii_fid (FID and dic) and applies fft and zerofilling. It can return magnitude or phased data.

    Args:
        data (ndarray complex): FID data
        dic (dict): Bruker dictionary
        si (int, optional): Number of points to zerofill. Defaults to 0.
        mc (bool, optional): Set to False for phased data. Defaults to True.
        dict (bool, optional): Set to True to return the dictionary. Defaults to False.
    """
    data = proc_base.zf_size(data, si)
    data = np.flip(np.fft.fftshift(np.fft.fft(data)))

    hz_scale = np.flip(np.fft.fftshift(np.fft.fftfreq(len(data), d=timescale[1]-timescale[0])))

    if(larmor_freq!=0.0):
            ppm_scale = hz_scale/larmor_freq

    return(data, ppm_scale, hz_scale) if larmor_freq!=0.0 else (data, hz_scale)
