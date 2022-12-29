import sep
import numpy as np
import astropy.units as u
import json
import requests
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge


try: # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
except ImportError:  # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve

try: # Python 3.x
    import http.client as httplib 
except ImportError:  # Python 2.x
    import httplib 



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
    objects[j]
    """
    flux, fluxerr, flag = sep.sum_circle(data_sub, objects['x'], objects['y'],na*objects['a'])
    print(flux)
    j, = np.where(flux == max(flux))
    
    return objects[j]    

def flux_profile(exposure, ra, dec , rmin, rmax, title_plot = '', save_plot =False, field=None, name =None):
    """
    Returns an array of the values across a rectangular slit of a source,
    that is wider in the x-axis

    input:
    -----
    data
    xpix
    ypix
    rmin
    rmax

    output 
    ------
    fluxes_ap
    """
    wcs = exposure.getWcs()
    obj_pos_lsst = lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees)
    x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
    #exp_photocalib = exposure.getPhotoCalib()
    exp_array = np.asarray(exposure.image.array, dtype='float')
    #obj_pos_2d = lsst.geom.Point2D(ra, dec)
    
    fluxes_ap = []
    fluxes_ap_err = []
    apertures = np.linspace(rmin, rmax, 15)
    pixel_to_arcsec = 0.2626

    for r in apertures:
        r/=pixel_to_arcsec
        f, ferr, flag = sep.sum_circle(exp_array, [x_pix], [y_pix], r, var = np.asarray(exposure.variance.array, dtype='float'))
        fluxes_ap.append(f[0])
        fluxes_ap_err.append(ferr[0])

    #fluxes = [exp_photocalib.instFluxToNanojansky(f, obj_pos_2d) for f in adu_values]
    #ai, aip, bi, bip = special.airy(fluxes)
    fluxes_ap /= fluxes_ap[-1]
    fluxes_ap_err /= fluxes_ap[-1] #sum(fluxes_ap)

    #plt.figure(figsize=(10,6))
    #plt.plot(apertures, fluxes_ap, '*', color='magenta')
    #plt.xlabel('arcsec aperture')
    #plt.ylabel('Normalized flux counts')

    if save_plot:
        #f.savefig('light_curves/{}/{}.jpeg'.format(field, name), bbox_inches='tight')
        pass
    #plt.show()

    return fluxes_ap


def getFluxesFromStars(repo, visit, ccdnum, collection_diff):
    """
    Calculates the magnitude of the stars used for photometric calibration and PSF measurement
    using the coadd image 
    
    Input:
    -----
    repo : [str] 
    visit : [int] 
    ccdnum : [int] 
    collection_diff : [int]
    
    Output: 
    ------
    df : [pd.DataFrame]
    
    """
    pixel_to_arcsec = 0.2626 #arcsec/pixel 
    butler = Butler(repo)
    #LSST_stars =  lp.Join_Tables_from_LSST(repo, visit, ccdnum, collection_diff, well_subtracted=False)
    
    LSST_stars = lp.Select_table_from_one_exposure(repo, visit, ccdnum, collection_diff, well_subtracted=False)
    LSST_stars_to_pandas = LSST_stars.to_pandas()
    LSST_stars_to_pandas = LSST_stars_to_pandas.reset_index()

    calexp = butler.get('calexp',visit=visit, detector=ccdnum , collections=collection_diff, instrument='DECam')
    #coadd = butler.get('goodSeeingDiff_matchedExp',visit=visit, detector=ccdnum , collections=collection_diff, instrument='DECam')
    df = pd.DataFrame(columns=['ra', 'dec','Panstarss_dr1_mag','Panstarss_dr1_flx', 'calculated_byme_flx', 'seeing','m_inst', 'airmass', 'expoTime'])
    #print('LSST stars df: ', LSST_stars)
    for i in range(len(LSST_stars_to_pandas)):
        ra_star =LSST_stars_to_pandas.loc[i]['coord_ra_ddegrees']
        dec_star = LSST_stars_to_pandas.loc[i]['coord_dec_ddegrees']

        # from Panstarss catalog
        constraints = {'nDetections.gt':1}
        results = ps1cone(ra_star, dec_star, 0.00028, **constraints)
        tab = ascii.read(results)
        # improve the format
        for filter in 'grizy':
            col = filter+'MeanPSFMag'
            try:
                tab[col].format = ".4f"
                tab[col][tab[col] == -999.0] = np.nan
            except KeyError:
                print("{} not found".format(col))
        ps1_mag = float(tab['gMeanPSFMag'][0])
        ps1_flux = ABMagToFlux(ps1_mag)*1e9 # to nJy flux
        
        # calculated by me using Source Extractor 
        obj_pos_lsst = lsst.geom.SpherePoint(ra_star, dec_star, lsst.geom.degrees)
        obj_pos_2d = lsst.geom.Point2D(ra_star, dec_star)
        wcs = calexp.getWcs()
        x_pix, y_pix = wcs.skyToPixel(obj_pos_lsst)
        data = np.asarray(calexp.image.array, dtype='float')
        psf = calexp.getPsf()
        sigma2fwhm = 2.*np.sqrt(2.*np.log(2.))
        pixel_to_arcsec = 0.2626 #arcsec/pixel 
        seeing = psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()*sigma2fwhm * pixel_to_arcsec
        r = seeing*2
        r/=pixel_to_arcsec
        flux, fluxerr, flag = sep.sum_circle(data, [x_pix], [y_pix], r=r, var = np.asarray(calexp.variance.array, dtype='float'))
        calc_flx = flux[0]

        # m_instrumental 
        expoTime = float(calexp.getInfo().getVisitInfo().exposureTime)
        m_instrumental = -2.5*np.log10(calc_flx/expoTime)
        airmass = float(calexp.getInfo().getVisitInfo().boresightAirmass)
        
        df.loc[len(df)] = [ra_star, dec_star, ps1_mag, ps1_flux, calc_flx, seeing, m_instrumental, airmass, expoTime]
    
    # making sure is clean:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['Panstarss_dr1_mag', 'm_inst'])
 
    return df 



def DoCalibration(repo, visit, ccdnum, collection_diff, config='SIBLING'):
    """
    Calibrate stars doing a Huber regression 
    """
    stars = getFluxesFromStars(repo, visit, ccdnum, collection_diff)
    print('Doing photometric calibration with {} stars'.format(len(stars)))
    
    #ps1_flux = np.array(stars.Panstarss_dr1_flx)#.reshape(-1, 1)
    #my_counts = np.array(stars.calculated_byme_flx)#.reshape(-1, 1)
    #print(ps1_flux)
    #print(my_counts)
    exp_time = 1 #np.unique(stars.expoTime)[0]
    ps1_flux = np.array(stars.Panstarss_dr1_flx)#.reshape(-1, 1)
    my_counts = np.array(stars.calculated_byme_flx).reshape(-1, 1)/exp_time
    X, y = my_counts, ps1_flux
    #print(ps1_flux)
    #print(my_counts)
    x = np.linspace(X.min(), X.max(), len(X))
    if config == 'SIBLING':
        epsilon = 1.35
    elif config == 'eridanus':
        epsilon = 1.5
    
    huber = HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100, epsilon=epsilon)
    huber.fit(X, y)
    coef_ = huber.coef_ * x + huber.intercept_
    plt.figure(1, figsize=(10,6))
    f, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]},sharex=True, figsize=(10,6))
    ax1.plot(x, coef_, 'magenta', label="huber loss, %s" % epsilon)
    ax1.plot(X, y, '*', color='blue')
    if config == 'SIBLING':
        #ax1.set_xlim(0,20000)
        #ax1.set_ylim(0,0.1e6)
        #ax2.set_ylim(-2500,2500)
        pass
    
    ax2.set_xlabel('Aperture fotometry from flux in counts [ADU^2]', fontsize=17)
    ax1.set_ylabel('PS1 flux [nJy]', fontsize=17)
    ax1.set_title('Calibration found : c {} + {}'.format(huber.coef_[0], huber.intercept_), fontsize=17)
    ax1.legend()
    #plt.subplot(313)
    model_ = huber.coef_ * X.ravel() + huber.intercept_ 
    #print('model_ - y : ',model_ - y)
    #print('X: ',X.ravel())
    ax2.plot(X.ravel(), model_ - y, 'o', color='purple', label='residuals')
    

    ax2.axhline(y=0, color='grey', linestyle='--')
    #ax2.xlim(0,100000)
    #ax2.show()    
    plt.show()
    return huber.coef_[0], huber.intercept_




####### this functions comes from the Panstarss DR1 API ###################################### 

def ps1cone(ra,dec,radius,table="mean",release="dr1",format="csv",columns=None,
           baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,
           **kw):
    """Do a cone search of the PS1 catalog
    
    Parameters
    ----------
    ra (float): (degrees) J2000 Right Ascension
    dec (float): (degrees) J2000 Declination
    radius (float): (degrees) Search radius (<= 0.5 degrees)
    table (string): mean, stack, or detection
    release (string): dr1 or dr2
    format: csv, votable, json
    columns: list of column names to include (None means use defaults)
    baseurl: base URL for the request
    verbose: print info about request
    **kw: other parameters (e.g., 'nDetections.min':2)
    """
    
    data = kw.copy()
    data['ra'] = ra
    data['dec'] = dec
    data['radius'] = radius
    return ps1search(table=table,release=release,format=format,columns=columns,
                    baseurl=baseurl, verbose=verbose, **data)


def ps1search(table="mean",release="dr1",format="csv",columns=None,
           baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,
           **kw):
    """Do a general search of the PS1 catalog (possibly without ra/dec/radius)
    
    Parameters
    ----------
    table (string): mean, stack, or detection
    release (string): dr1 or dr2
    format: csv, votable, json
    columns: list of column names to include (None means use defaults)
    baseurl: base URL for the request
    verbose: print info about request
    **kw: other parameters (e.g., 'nDetections.min':2).  Note this is required!
    """
    
    data = kw.copy()
    if not data:
        raise ValueError("You must specify some parameters for search")
    checklegal(table,release)
    if format not in ("csv","votable","json"):
        raise ValueError("Bad value for format")
    url = f"{baseurl}/{release}/{table}.{format}"
    if columns:
        # check that column values are legal
        # create a dictionary to speed this up
        dcols = {}
        for col in ps1metadata(table,release)['name']:
            dcols[col.lower()] = 1
        badcols = []
        for col in columns:
            if col.lower().strip() not in dcols:
                badcols.append(col)
        if badcols:
            raise ValueError('Some columns not found in table: {}'.format(', '.join(badcols)))
        # two different ways to specify a list of column values in the API
        # data['columns'] = columns
        data['columns'] = '[{}]'.format(','.join(columns))

# either get or post works
#    r = requests.post(url, data=data)
    r = requests.get(url, params=data)

    if verbose:
        print(r.url)
    r.raise_for_status()
    if format == "json":
        return r.json()
    else:
        return r.text
    
def checklegal(table,release):
    """Checks if this combination of table and release is acceptable
    
    Raises a VelueError exception if there is problem
    """
    
    releaselist = ("dr1", "dr2")
    if release not in ("dr1","dr2"):
        raise ValueError("Bad value for release (must be one of {})".format(', '.join(releaselist)))
    if release=="dr1":
        tablelist = ("mean", "stack")
    else:
        tablelist = ("mean", "stack", "detection")
    if table not in tablelist:
        raise ValueError("Bad value for table (for {} must be one of {})".format(release, ", ".join(tablelist)))


def ps1metadata(table="mean",release="dr1",
           baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"):
    """Return metadata for the specified catalog and table
    
    Parameters
    ----------
    table (string): mean, stack, or detection
    release (string): dr1 or dr2
    baseurl: base URL for the request
    
    Returns an astropy table with columns name, type, description
    """
    
    checklegal(table,release)
    url = f"{baseurl}/{release}/{table}/metadata"
    r = requests.get(url)
    r.raise_for_status()
    v = r.json()
    # convert to astropy table
    tab = Table(rows=[(x['name'],x['type'],x['description']) for x in v],
               names=('name','type','description'))
    return tab


def mastQuery(request):
    """Perform a MAST query.

    Parameters
    ----------
    request (dictionary): The MAST request json object

    Returns head,content where head is the response HTTP headers, and content is the returned data
    """
    
    server='mast.stsci.edu'

    # Grab Python Version 
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "User-agent":"python-requests/"+version}

    # Encoding the request as a json string
    requestString = json.dumps(request)
    requestString = urlencode(requestString)
    
    # opening the https connection
    conn = httplib.HTTPSConnection(server)

    # Making the query
    conn.request("POST", "/api/v0/invoke", "request="+requestString, headers)

    # Getting the response
    resp = conn.getresponse()
    head = resp.getheaders()
    content = resp.read().decode('utf-8')

    # Close the https connection
    conn.close()

    return head,content


def resolve(name):
    """Get the RA and Dec for an object using the MAST name resolver
    
    Parameters
    ----------
    name (str): Name of object

    Returns RA, Dec tuple with position"""

    resolverRequest = {'service':'Mast.Name.Lookup',
                       'params':{'input':name,
                                 'format':'json'
                                },
                      }
    headers,resolvedObjectString = mastQuery(resolverRequest)
    resolvedObject = json.loads(resolvedObjectString)
    # The resolver returns a variety of information about the resolved object, 
    # however for our purposes all we need are the RA and Dec
    try:
        objRa = resolvedObject['resolvedCoordinate'][0]['ra']
        objDec = resolvedObject['resolvedCoordinate'][0]['decl']
    except IndexError as e:
        raise ValueError("Unknown object '{}'".format(name))
    return (objRa, objDec)

####################################### end of PS1 functions #################################