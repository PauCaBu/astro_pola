import numpy as np 
import astropy 
import pandas as pd



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