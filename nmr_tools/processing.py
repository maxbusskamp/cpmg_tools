def read_bruker(datapath):
    """
    This reads in Bruker Datasets. At the moment it still uses nmrglue
    Input: Path string to the Bruker Dataset, ending with '/pdata/1'
    """    
    import nmrglue as ng

    dic, data = ng.bruker.read_pdata(datapath)
    udic = ng.bruker.guess_udic(dic, data)
    uc = ng.fileio.fileiobase.uc_from_udic(udic)
    ppm_scale = uc.ppm_scale()
    hz_scale = uc.hz_scale()

    return(ppm_scale, hz_scale, data)


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
    import numpy as np

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
    import numpy as np

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
    import numpy as np

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
    import numpy as np

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    
    return idx


def find_nearest(array, value):
    import numpy as np

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return array[idx]


def shift_bit_length(x):
    return 1<<(x-1).bit_length()


def combine_stepped_aq(datasets, set_sw=0, precision_multi=1, verbose=False):
    """
    This combines multiple Bruker Datasets into one spectrum, using a calculation of the envelope to determine the highest x-values, if multiple exist.

    Args:
        datasets (str): String list containing the Bruker Dataset Paths ['', '', '']
        set_sw (int, optional): Spectral width is set to +-set_sw/2
        precision_multi (int, optional): Multiplier for the interpolation step. Enhances precision for .spe files.
        verbose (bool, optional): If True, activates debug output
    """
    from scipy import interpolate
    import numpy as np

    # Combine Data
    index = 0
    for datapath in datasets:
        dataset = read_bruker(datapath)
        dataset = np.array(dataset)
        
        if(index == 0):
            dataset_combine = dataset
        else:
            dataset_combine = np.hstack((dataset_combine, dataset))
        index = index + 1

    # Sort Data
    dataset_combine = dataset_combine[:,dataset_combine[1].argsort()]
    # Generate Envelope
    high_idx, low_idx = get_envelope_idx(dataset_combine[2], dmin=4, dmax=4)
    dataset_array_masked = dataset_combine[2][low_idx]
    dataset_x_masked = dataset_combine[1][low_idx]

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


    # Restrict spectral width
    if(set_sw is not 0):
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
    dataset_array = np.array(dataset_new)

    if(verbose==True):
        precision = np.round((abs(dataset_array[-1,0])+abs(dataset_array[0,0])), -3)-(abs(dataset_array[-1,0])+abs(dataset_array[0,0]))
        print('.spe precision: ' + str(int(precision)) + ' Hz')
        print('SW raw: ' + str(int(abs(dataset_array[-1,0])+abs(dataset_array[0,0]))) + ' Hz')
        print('SW rounded: ' + str(int(np.round((abs(dataset_array[-1,0])+abs(dataset_array[0,0])), -3))) + ' Hz')
        

    return(dataset_array)


def split_echotrain(datapath, dw, echolength, blankinglength, numecho):
    """This splits the echo train of a given Bruker Dataset and coadds each echo.

    Args:
        datapath (str): Path string to the Bruker Dataset, ending with '/pdata/1'
        dw (float): Dwell time
        echolength (float): length of aquisition
        blankinglength (float): Length of blank time between aquisitions
        numecho (int): Number of recorded echos (e.g. l22)

    Return:
        ppm_scale
        hz_scale
        data
    """  
    import nmrglue as ng
    import numpy as np

    dic, data = ng.bruker.read(datapath)
    udic = ng.bruker.guess_udic(dic, data)
    uc = ng.fileiobase.uc_from_udic(udic)

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

    ppm_scale = uc.ppm_scale()
    hz_scale = uc.hz_scale()

    return(ppm_scale, hz_scale, data)
