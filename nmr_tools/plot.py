import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from plot_module import plot_module
import os
import re
import glob
from scipy.interpolate import griddata
from matplotlib import cm
import nmrglue as ng
import math
import tarfile


def offset_exp(datapath, nexp, stepsize, sfo1, direction='mp'):  # Class for plotting Offset Experiments, extracted rows have to start at 101

    stepsize=stepsize/sfo1
    start = 101
    stop = 100+nexp
    data = [[],[]]
    for i in range(start, stop):
        data_slice = plot_module.plot_1D(datapath+str(i), val_only=True)
        if(direction=='mp'):
            start_ppm = find_nearest_index(data_slice[0], -(i-101)*stepsize+stepsize/2.0)  # find nearest index from ppm
            end_ppm = find_nearest_index(data_slice[0], -(i-101)*stepsize-stepsize/2.0)  # find nearest index from ppm
        else:
            start_ppm = find_nearest_index(data_slice[0], (i-101)*stepsize+stepsize/2.0)  # find nearest index from ppm
            end_ppm = find_nearest_index(data_slice[0], (i-101)*stepsize-stepsize/2.0)  # find nearest index from ppm
        data[0].extend(data_slice[0][start_ppm:end_ppm])
        data[1].extend(data_slice[1][start_ppm:end_ppm])

    offset_range = abs(np.min(data[0])) + abs(np.max(data[0]))
    data[0] = [(val+offset_range/2)*sfo1/1000 for val in data[0]]
    data[1].reverse()

    return(data)


def save(save_path, filename, figure, dpi=300, pdf=True, png=True, constrain=False):
    if(pdf==True):
        if(constrain==False):
            figure.savefig(save_path + filename + '.pdf',
                        format='pdf', facecolor=(1, 1, 1, 0), transparent="True")
        else:
            figure.savefig(save_path + filename + '.pdf',
                        format='pdf', facecolor=(1, 1, 1, 0), transparent="True", bbox_inches='tight',
                        pad_inches=0.1)
    if(png==True):
        if(constrain==False):
            figure.savefig(save_path + filename + '.png',
                        format='png', dpi=dpi, facecolor=(1, 1, 1, 0), transparent="True")
        else:
            figure.savefig(save_path + filename + '.png',
                        format='png', dpi=dpi, facecolor=(1, 1, 1, 0), transparent="True", bbox_inches='tight',
                        pad_inches=0.1)


def heatmap(axis, data_path, main_list, sub_list=[''], subsub_list=[''], levels=[-0.99, 0.0], x_limits=[999.9, 999.9], y_limits=[999.9, 999.9], rect=False, sec_ax=True, double_echo=False, colorbar=True, xlabel=True, ylabel=True, cmap=LinearSegmentedColormap.from_list('mycmap', ['#d73027', '#fdae61', '#ffffbf', '#abd9e9', '#4575b5'])):
    def tokenize(filename):
        digits = re.compile(r'(\d+)')
        return tuple(int(token) if match else token
                    for token, match in
                    ((fragment, digits.search(fragment))
                    for fragment in digits.split(filename)))


    def forward(x):
        global delta
        global tw
        return x/(np.sqrt((float(filename.split('delta_')[-1].split('_')[0])*1000)/(float(filename.split('tw_')[-1].split('_')[0])/1000000))/1000)


    def inverse(x):
        global delta
        global tw
        return x/(np.sqrt((float(filename.split('delta_')[-1].split('_')[0])*1000)/(float(filename.split('tw_')[-1].split('_')[0])/1000000))/1000)



    file_list = []
    for main in main_list:
        for sub in sub_list:
            for subsub in subsub_list:
                file_list.extend(glob.glob(data_path + '*' + main + '*' + sub + '*' + subsub + '*'))
    file_list.sort(key=tokenize)

    x_data = []
    y_data = []
    z_data = []
    for filename in file_list:
            delta = float(filename.split('delta_')[-1].split('_')[0])
            tw = float(filename.split('tw_')[-1].split('_')[0])
            dat = np.genfromtxt(filename, delimiter=' ')

            for i in range(len(dat[:,0])):
                if(double_echo==True):
                    y_data.append(float(filename.split('rffactor_')[-1].split('_')[1]))
                else:
                    y_data.append(float(filename.split('rffactor_')[-1].split('_')[0]))

            x_data.extend(dat[:,0])
            z_data.extend(dat[:,1])

    x_data = np.asarray(list(map(float, x_data))) / 1000
    if(rect==True):
        y_data = np.asarray(list(map(float, y_data))) * 100 * 1000
    else:
        y_data = np.asarray(list(map(float, y_data))) * (np.sqrt((float(filename.split('delta_')[-1].split('_')[0])*1000)/(float(filename.split('tw_')[-1].split('_')[0])/1000000))/1000)
    z_data = np.asarray(list(map(float, z_data))) / max(abs(np.asarray(list(map(float, z_data)))))

    # create x-y points to be used in heatmap
    xi = np.linspace(x_data.min(),x_data.max(),1000)
    yi = np.linspace(y_data.min(),y_data.max(),201)
    # Z is a matrix of x-y values
    zi = griddata((x_data, y_data), z_data, (xi[None,:], yi[:,None]), method='cubic')

    zmin = -1.0
    zmax = 1.0

    # Create the contour plot
    CS=axis.contourf(xi, yi, zi, 199, cmap=cmap, vmax=zmax, vmin=zmin)

    # Plot Level Lines
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    contour = axis.contour(xi, yi, zi, levels, colors='k', ls=[('-','-')], vmax=zmax, vmin=zmin)
    axis.clabel(contour, inline=True, fontsize=12, fmt = '%1.2f')

    lines = []
    for num_level in range(len(levels)):
        for ii, seg in enumerate(contour.allsegs[num_level]):
            lines.append((seg[:,0], seg[:,1]))

    if(sec_ax==True):
        secax = axis.secondary_yaxis('right', functions=(forward, inverse))
        secax.set_ylabel(r'RF-Factor', fontsize=12)

    axis.xaxis.set_minor_locator(MultipleLocator(100))

    if(xlabel==True):
        axis.set_xlabel(r'$\omega_\mathrm{iso}$ / $2\pi$ / kHz', fontsize=12)

    if(ylabel==True):
        axis.set_ylabel(r'$\nu_\mathrm{RF}$ / kHz', fontsize=12)

    if(rect==False):
        axis.yaxis.set_minor_locator(MultipleLocator(50))

    if(ylabel==False):
        axis.set_yticklabels([])
    if(xlabel==False):
        axis.set_xticklabels([])

    if(colorbar==True):
        # colorbar = plt.colorbar(CS, ax=[axis], aspect=50, pad=0.0)
        norm = cm.colors.Normalize(vmin=-1.0, vmax=1.0)
        cmap = LinearSegmentedColormap.from_list('mycmap', ['#660000', 'firebrick', 'lightgrey', 'royalblue', '#142952'])
        colorbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=[axis], aspect=50, pad=0.0)
        colorbar.set_ticks(np.arange(-1.00, 1.10, 0.2))
        colorbar.set_ticklabels(['-1.0', '-0.8', '-0.6', '-0.4', '-0.2', ' 0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
    
    if(x_limits[0] != 999.9):
        axis.set_xlim(x_limits[0], x_limits[1])
    if(y_limits[0] != 999.9):
        axis.set_ylim(y_limits[0], y_limits[1])

    return(CS, zi, lines)


def heatmap_tar(axis, data_path, tar_file, main_list, sub_list=[''], subsub_list=[''], levels=[-0.99, 0.0, 0.99], x_limits=[999.9, 999.9], y_limits=[999.9, 999.9], naming='old', csa_col=False, rect=False, sec_ax=True, double_echo=False, colorbar=True, xlabel=True, ylabel=True, cmap=LinearSegmentedColormap.from_list('mycmap', ['#d73027', '#fdae61', '#ffffbf', '#abd9e9', '#4575b5'])):
    def tokenize(filename):
        digits = re.compile(r'(\d+)')
        return tuple(int(token) if match else token
                    for token, match in
                    ((fragment, digits.search(fragment))
                    for fragment in digits.split(filename)))


    def forward(x):
        global delta
        global tw
        return x/(np.sqrt((float(filename_string.split('delta_')[-1].split('_')[0])*1000)/(float(filename_string.split('tw_')[-1].split('_')[0])/1000000))/1000)


    def inverse(x):
        global delta
        global tw
        return x/(np.sqrt((float(filename_string.split('delta_')[-1].split('_')[0])*1000)/(float(filename_string.split('tw_')[-1].split('_')[0])/1000000))/1000)


    tar = tarfile.open(tar_file)

    indices = []
    for main in main_list:
        for sub in sub_list:
            for subsub in subsub_list:
                for i, elem in enumerate(tar.getnames()):
                    if (data_path in elem and
                        main in elem and
                        sub in elem and
                        subsub in elem):
                            indices.append(i)

    x_data = []
    y_data = []
    z_data = []

    for i in indices:
        filename = tar.getmembers()[i]
        filename_string = str(filename)

        dat = np.genfromtxt(tar.extractfile(filename), delimiter=' ')

        delta = float(filename_string.split('delta_')[-1].split('_')[0])
        tw = float(filename_string.split('tw_')[-1].split('_')[0])

        for i in range(len(dat[:,0])):
            if(double_echo==True):
                if(naming=='old'):
                    y_data.append(float(filename_string.split('rffactor_')[-1].split('_')[1]))
                else:
                    y_data.append(float(filename_string.split('rffactor_')[-1].split('.out')[1]))
            else:
                if(naming=='old'):
                    y_data.append(float(filename_string.split('rffactor_')[-1].split('_')[0]))
                else:
                    y_data.append(float(filename_string.split('rffactor_')[-1].split('.out')[0]))

        if(csa_col==False):
            x_data.extend(dat[:,0])
        else:
            x_data.extend(dat[:,4])
        z_data.extend(dat[:,1])

    # for filename in file_list:
    #         delta = float(filename.split('delta_')[-1].split('_')[0])
    #         tw = float(filename.split('tw_')[-1].split('_')[0])
    #         dat = np.genfromtxt(filename, delimiter=' ')

            # for i in range(len(dat[:,0])):
            #     if(double_echo==True):
            #         y_data.append(float(filename.split('rffactor_')[-1].split('_')[1]))
            #     else:
            #         y_data.append(float(filename.split('rffactor_')[-1].split('_')[0]))

            # x_data.extend(dat[:,0])
            # z_data.extend(dat[:,1])

    x_data = np.asarray(list(map(float, x_data))) / 1000
    if(rect==True):
        y_data = np.asarray(list(map(float, y_data))) * 100 * 1000
    else:
        y_data = np.asarray(list(map(float, y_data))) * (np.sqrt((float(filename_string.split('delta_')[-1].split('_')[0])*1000)/(float(filename_string.split('tw_')[-1].split('_')[0])/1000000))/1000)
    z_data = np.asarray(list(map(float, z_data))) / max(abs(np.asarray(list(map(float, z_data)))))

    # create x-y points to be used in heatmap
    xi = np.linspace(x_data.min(),x_data.max(),1000)
    yi = np.linspace(y_data.min(),y_data.max(),201)
    # Z is a matrix of x-y values
    zi = griddata((x_data, y_data), z_data, (xi[None,:], yi[:,None]), method='cubic')

    zmin = -1.0
    zmax = 1.0

    # Create the contour plot
    CS=axis.contourf(xi, yi, zi, 199, cmap=cmap, vmax=zmax, vmin=zmin)

    # Plot Level Lines
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    contour = axis.contour(xi, yi, zi, levels, colors='k', ls=[('-','-')], vmax=zmax, vmin=zmin)
    axis.clabel(contour, inline=True, fontsize=12, fmt = '%1.2f')

    lines = []
    for num_level in range(len(levels)):
        for ii, seg in enumerate(contour.allsegs[num_level]):
            lines.append((seg[:,0], seg[:,1]))

    if(sec_ax==True):
        secax = axis.secondary_yaxis('right', functions=(forward, inverse))
        secax.set_ylabel(r'RF-Factor', fontsize=12)

    axis.xaxis.set_minor_locator(MultipleLocator(100))

    if(xlabel==True):
        if(csa_col==False):
            axis.set_xlabel(r'$\omega_\mathrm{iso}$ / $2\pi$ / kHz', fontsize=12)
        else:
            axis.set_xlabel(r'$\delta_\mathrm{aniso}$ / $2\pi$ / kHz', fontsize=12)

    if(ylabel==True):
        axis.set_ylabel(r'$\nu_\mathrm{RF}$ / kHz', fontsize=12)

    if(rect==False):
        axis.yaxis.set_minor_locator(MultipleLocator(50))

    if(ylabel==False):
        axis.set_yticklabels([])
    if(xlabel==False):
        axis.set_xticklabels([])

    if(colorbar==True):
        # colorbar = plt.colorbar(CS, ax=[axis], aspect=50, pad=0.0)
        norm = cm.colors.Normalize(vmin=-1.0, vmax=1.0)
        cmap = LinearSegmentedColormap.from_list('mycmap', ['#660000', 'firebrick', 'lightgrey', 'royalblue', '#142952'])
        colorbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=[axis], aspect=50, pad=0.0)
        colorbar.set_ticks(np.arange(-1.00, 1.10, 0.2))
        colorbar.set_ticklabels(['-1.0', '-0.8', '-0.6', '-0.4', '-0.2', ' 0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        
        # colorbar = plt.colorbar(CS, ax=axis)
        # colorbar.set_ticks(np.arange(-1.00, 1.10, 0.25))
        # colorbar.set_ticklabels(['-1.00', '-0.75', '-0.50', '-0.25', ' 0.00', '0.25', '0.50', '0.75', '1.00'])
    if(x_limits[0] != 999.9):
        axis.set_xlim(x_limits[0], x_limits[1])
    if(y_limits[0] != 999.9):
        axis.set_ylim(y_limits[0], y_limits[1])
    
    return(CS, zi, lines)


def heatmap_slice(axis, data_path, main_list, sub_list=[''], rffactor='0.26', xlabel=True, ylabel=True, label='none'):
    def tokenize(filename):
        digits = re.compile(r'(\d+)')
        return tuple(int(token) if match else token
                    for token, match in
                    ((fragment, digits.search(fragment))
                    for fragment in digits.split(filename)))
                    

    file_list = []
    for main in main_list:
        for sub in sub_list:
            file_list.extend(glob.glob(data_path + main + '*' + 'rffactor_' + rffactor + '_' + '*' + sub + '*'))
    file_list.sort(key=tokenize)

    x_data = []
    y_data = []
    temp = []
    for filename in file_list:
            x_data, y_data, temp, temp = np.genfromtxt(filename, delimiter=' ', unpack=True)

    plot = axis.plot(x_data/1000, y_data, label=label)
    axis.set_ylim(-1.0, 1.0)
    axis.axhline(0.0, c='k', ls='--', lw=0.5)

    axis.xaxis.set_minor_locator(MultipleLocator(100))

    if(label!='none'):
        axis.legend(fontsize=10, loc='upper center', frameon=False)

    if(xlabel==True):
        axis.set_xlabel(r'$\omega_\mathrm{iso}$ / $2\pi$ / kHz', fontsize=12)

    # if(ylabel==True):
    #     axis.set_ylabel(r'Z-Magn. / a.u.', family='serif', fontsize=12)

    if(ylabel==False):
        axis.set_yticklabels([])
        axis.set_yticks([])
    if(xlabel==False):
        axis.set_xticklabels([])
        axis.set_xticks([])

    return(plot)


def get_contour_span(contourlines, linenumber, threshold):
    x_contour = contourlines[linenumber][0][:]
    y_contour = contourlines[linenumber][1][:]
    y_center = y_contour[int(len(y_contour)/2)]
    threshold_max = y_center + threshold
    threshold_min = y_center - threshold

    i = int(len(y_contour)/2)
    j = int(len(y_contour)/2)
    mask = np.full(len(y_contour), False)
    try:
        for k in range(int(len(y_contour)/2)):
            if (y_contour[i] >= threshold_min and y_contour[i] <= threshold_max):
                mask[i] = True
            else:
                raise StopIteration
            if (y_contour[j] >= threshold_min and y_contour[j] <= threshold_max):
                mask[j] = True
            else:
                raise StopIteration
            i += 1
            j -= 1
    except StopIteration: pass

    x_masked = x_contour[mask]
    y_masked = y_contour[mask]

    y_left = y_masked[0]
    y_right = y_masked[-1]

    x_left = x_masked[0]
    x_right = x_masked[-1]

    x_span = abs(x_right - x_left)

    return(x_span, x_masked, y_masked, x_contour, y_contour)


def get_popt_data(filename):  # Plot popt Spectra saved via Topspin save as txt
    file = open(filename, 'r')
    data = file.read()
    offset_range = re.findall('# LEFT = \s*(.+?)\s*\n', data)[0].split('RIGHT')
    popt_x = np.linspace(float(re.sub(r'[^-0-9.]', '', offset_range[0])[:-1]),
                         float(re.sub(r'[^-0-9.]', '', offset_range[1])[:-1]),
                         int(re.sub(r'[^-0-9]', '', re.findall('# SIZE = \s*(.+?)\s*\n', data)[0])))
    popt_y = np.genfromtxt(filename, comments='#')
    return popt_x, popt_y


def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def roundup(a, digits=0):
    n = 10**-digits
    return round(math.ceil(a / n) * n, digits)


def rounddown(a, digits=0):
    n = 10**-digits
    return round(math.floor(a / n) * n, digits)


def tokenize(filename):
            digits = re.compile(r'(\d+)')
            return tuple(int(token) if match else token
                        for token, match in
                        ((fragment, digits.search(fragment))
                        for fragment in digits.split(filename)))
