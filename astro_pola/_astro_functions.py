import numpy as np 
import astropy 
import pandas as pd
import pickle
import os
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve
from matplotlib.colors import LogNorm
import sep 


def compare_to(directory, sfx, factor, beforeDate=57072):
    '''
    Returns Jorge Martinez-Palomera or Francisco Forsters code
    
    Input
    -----
    directory :
    sfx :
    factor :
    beforeDate :
    
    Output
    ------
    x, y, yerr
    '''

    SIBLING = directory

    if SIBLING!=None and type(SIBLING)==str:
        Jorge_LC = pd.read_csv(SIBLING, header=5)
        Jorge_LC = Jorge_LC[Jorge_LC['mjd']<beforeDate] 
        if factor==0.5:
            
            param = Jorge_LC['aperture_{}_0'.format(sfx)]
            param_err = Jorge_LC['aperture_{}_err_0'.format(sfx)]
            median_jorge=np.median(param)
            x = Jorge_LC.mjd- min(Jorge_LC.mjd)
            y = param# - median_jorge
            yerr = param_err
            return x, y, yerr
        if factor==0.75:

            
            param = Jorge_LC['aperture_{}_1'.format(sfx)]
            param_err = Jorge_LC['aperture_{}_err_1'.format(sfx)]
            median_jorge = np.median(param)
            mean = np.mean(param)

            x = Jorge_LC.mjd- min(Jorge_LC.mjd)
            y = param #- median_jorge
            yerr = param_err
            return x, y, yerr

        if factor==1:
            
            param = Jorge_LC['aperture_{}_2'.format(sfx)]
            param_err = Jorge_LC['aperture_{}_err_2'.format(sfx)]
            mean = np.mean(param)
            norm = np.linalg.norm(np.array(param))
            median_jorge = np.median(param)

            #if sfx == 'flx':
            #    fluxes_and_err = pc.ABMagToFlux(param, param_err)
            #    param = fluxes_and_err[0]
            #    param_err = fluxes_and_err[1]

            x = Jorge_LC.mjd- min(Jorge_LC.mjd)
            y = param #- median_jorge
            yerr = param_err
            return x, y, yerr
            #plt.errorbar(Jorge_LC.mjd - min(Jorge_LC.mjd), Jorge_LC.aperture_flx_2 - mean, yerr=Jorge_LC.aperture_flx_err_2,  capsize=4, fmt='o', ecolor='m', color='m', label='Jorge & F.Forster LC')
        if factor==1.25:
            #std = np.std(Jorge_LC.aperture_flx_3)
            param = Jorge_LC['aperture_{}_3'.format(sfx)]
            param_err = Jorge_LC['aperture_{}_err_3'.format(sfx)]
            mean = np.mean(param)
            median_jorge= np.median(param)
            #if sfx == 'flx':
            #    fluxes_and_err = pc.ABMagToFlux(param, param_err)
            #    param = fluxes_and_err[0]
            #    param_err = fluxes_and_err[1]

            x = Jorge_LC.mjd- min(Jorge_LC.mjd)
            y = param # - median_jorge
            yerr = param_err
            return x, y, yerr
            #plt.errorbar(Jorge_LC.mjd - min(Jorge_LC.mjd), Jorge_LC.aperture_flx_3 - mean, yerr=Jorge_LC.aperture_flx_err_3,  capsize=4, fmt='o', ecolor='m', color='m', label='Jorge & F.Forster LC')
        if factor==1.5:
            #std = np.std(Jorge_LC.aperture_flx_4)
            
            param = Jorge_LC['aperture_{}_4'.format(sfx)]
            param_err = Jorge_LC['aperture_{}_err_4'.format(sfx)]             
            norm = np.linalg.norm(np.array(param))
            median_jorge= np.median(param)
            mean = np.mean(param)
            
            #if sfx == 'flx':
            #    fluxes_and_err = pc.ABMagToFlux(param, param_err)
            #    param = fluxes_and_err[0]
            #    param_err = fluxes_and_err[1]

            x = Jorge_LC.mjd- min(Jorge_LC.mjd)
            y = param #- median_jorge
            yerr = param_err
            return x, y, yerr
    
    HiTS = directory 

    if HiTS!=None and HiTS[0:24]=="/home/jahumada/HiTS_LCs/" and type(HiTS)==str:
        HiTS_LC = pd.read_csv(HiTS, skiprows = 2, delimiter=' ')
        HiTS_LC = HiTS_LC.dropna()
        HiTS_LC = HiTS_LC[(HiTS_LC['MJD']<beforeDate) & (HiTS_LC['band']=='g')]
        x = HiTS_LC.MJD

        if sfx == 'flx':
            y = HiTS_LC.ADU
            yerr = HiTS_LC.e_ADU
            return x, y, yerr

        if sfx == 'mag':
            y = HiTS_LC.mag
            yerr = HiTS_LC.e1_mag
            return x, y, yerr

    else:
        None

def FluxToMag(flux, fluxerr, magzero=24.3):
    """
    Transform flux and fluxerr to magnitude 
    
    Input
    -----
    flux :
    fluxerr :
    magzero : 

    Output:
    ------
    mag, magerr
    """
    mag = -2.5*np.log10(flux) + magzero
    magerr = np.fabs(-2.5 * fluxerr/(flux * np.log(10)))
    return mag, magerr


def FluxJyToABMag(flux, fluxerr=None):
    """
    Transform flux and fluxerr in Jy to AB magnitude 
    
    Input
    -----
    flux :
    fluxerr :

    Output:
    ------
    magab, magab_err
    """
    magab = -2.5*np.log10(flux) + 8.9
    magab_err = None
    if type(fluxerr)==float:
        magab_err = np.fabs(-2.5 * fluxerr/(flux * np.log(10)))

    return magab, magab_err


def ABMagToFluxJy(mab, mab_err=None):
    """
    Returns AB magnitude to flux in Jy

    Input:
    -----
    mab : [float]

    Output:
    -------
    f : [float]
    """
    f = 10**(-0.4*(mab - 8.90))
    if type(mab_err) != type(None):
        f_err = np.sqrt((f*np.log(10)*0.4*mab_err)**2)
        return f, f_err
    else:
        return f


def convolve(path, field, worst_seeing):
    """
    Convolving images

    input:
    -----
    path : [string] path to the fits images 
    field : [string] name of the field
    worst_seeing : [float]

    output:
    ------


    
    """
    #worst_seeing = 1.44175923
    sigma2fwhm = 2.*np.sqrt(2.*np.log(2.)) 
    stdev = worst_seeing/sigma2fwhm
    arcsec_to_pixel = 0.27#626 # arcsec/pixel
    stdev/=arcsec_to_pixel # we transform to pixel values 
    kernel = Gaussian2DKernel(x_stddev=stdev)

    #directory = 'Blind15A_16_N24'

    file = '{}_convolved.pickle'.format(field)

    if not os.path.exists(file):
        # convolve images if file doesnt exist

        convolved_images = {}
        
        # Here I convolve all the images in the Blind15A_16_N24 directory, and save them in 
        # the convolved_images dictionary 

        for filename in os.listdir(path):
            if filename.endswith('.fits'):
                fitsfile = get_pkg_data_filename(path+'/'+filename)
                img = fits.open(fitsfile)[1]
                astropy_conv = convolve(img.data, kernel)
                visit = filename.split('_')[11]
                convolved_images[visit] = astropy_conv
                
        with open(file, 'wb') as handle:
            
            pickle.dump(convolved_images, handle, protocol = pickle.HIGHEST_PROTOCOL)
    else:
        # load convolve images 
        with open(file, 'rb') as handle:
            convolved_images = pickle.load(handle)

    return convolved_images



### centering codes for photometry ####

def center_brightest_zone(data, thresh, area):
    """
    
    retrieves x and y position of the brightest source 
    in the data matrix

    input:
    -----
    data
    thresh
    area

    output:
    ------
    x, y 

    """
    objects = sep.extract(data, thresh, minarea=area)
    obj, j = Select_largest_flux(data, objects)
    return obj['x'], obj['y']

# select largest flux

def Select_largest_flux(data_sub, objects, na=6):
    """
    Uses source extractor to select the brightest detected source

    Input
    -----
    data_sub : [np.matrix]
    objects : 
    na :

    Output
    -----
    objects, j 
    """
    flux, fluxerr, flag = sep.sum_circle(data_sub, objects['x'], objects['y'],na*objects['a'])
    print(flux)
    j, = np.where(flux == max(flux))
    
    return objects, j  

# Finding peak element in a 2D Array.

def findPeakGrid(mat):
    """
    author : https://www.geeksforgeeks.org/find-peak-element-2d-array/

    finds the peak value in a 2d array

    input:
    -----
    mat : [2D ndarray]

    output:
    ------
    [x,y] : x and y coordinates of the peak value
    
    """

    stcol = 0
    endcol = len(mat[0]) - 1; # Starting po  end po of Search Space
 
    while (stcol <= endcol):  # Bin Search Condition
 
        midcol = stcol + int((endcol - stcol) / 2)
        ansrow = 0;
        # "ansrow" To keep the row number of global Peak
        # element of a column
 
        # Finding the row number of Global Peak element in
        # Mid Column.
        for r in range(len(mat)):
            ansrow = r if mat[r][midcol] >= mat[ansrow][midcol] else ansrow;
         
 
        # Finding next Search space will be left or right
        valid_left =  midcol - 1 >= stcol and mat[ansrow][midcol - 1] > mat[ansrow][midcol];
        valid_right = midcol + 1 <= endcol and mat[ansrow][midcol + 1] > mat[ansrow][midcol];
 
        # if we're at Peak Element
        if (not valid_left and not valid_right) :
            return [ ansrow, midcol ];
         
 
        elif (valid_right):
            stcol = midcol  + 1; # move the search space in right
        else:
            endcol = midcol  - 1; # move the search space in left
     
    return [ -1, -1 ];
 
# Driver Code
#arr = [[7, 8, 5], [9 ,8, 6], [3, 5, 0]];
#result = findPeakGrid(arr);
#print("Peak element found at index:", result)

################## 