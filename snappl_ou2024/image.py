__all__ = [ 'OpenUniverse2024FITSImage' ]

import re
import pathlib

import pandas
import numpy as np

import astropy
from astropy.table import Table

import galsim.roman

from snappl.config import Config
from snappl.image import CompressedFITSImage


# ======================================================================
# OpenUniverse 2024 Images are gzipped FITS files
#  HDU 0 : (something, no data)
#  HDU 1 : SCI (32-bit float)
#  HDU 2 : ERR (32-bit float)
#  HDU 3 : DQ (32-bit integer)

class OpenUniverse2024FITSImage( CompressedFITSImage ):
    def __init__( self, *args, imagehdu=1, noisehdu=2, flagshdu=3, **kwargs ):
        super().__init__( *args, imagehdu=imagehdu, noisehdu=noisehdu, flagshdu=flagshdu, **kwargs )
        # Not doing the is_superclass thing here because parent class FITSImage consumes all the explicit
        #   keywords that we do, so it can go ahead and pretend to not be a superclass for purposes
        #   of kwargs validation

        self._zeropoint = None

    _image_class_base_path_config_item = 'system.ou24.images'

    _filenamere = re.compile( r'^Roman_TDS_simple_model_(?P<band>[^_]+)_(?P<pointing>\d+)_(?P<sca>\d+).fits' )

    @property
    def truthpath( self ):
        """Path to truth catalog.  WARNING: this is OpenUniverse2024FITSImage-specific, use with care."""
        tds_base = pathlib.Path( Config.get().value( 'system.ou24.tds_base' ) )
        return ( tds_base / f'truth/{self.band}/{self.observation_id}/'
                 f'Roman_TDS_index_{self.band}_{self.observation_id}_{self.sca}.txt' )

    def _get_internal_attribute( self, prop ):
        if prop == 'observation_id':
            mat = self._filenamere.search( self.filepath.name )
            if mat is None:
                raise ValueError( f"Failed to parse {self.filepath.name} for pointing" )
            self._observation_id = mat.group( 'pointing' )

        elif prop == 'exptime':
            header = self.get_fits_header()
            if 'EXPTIME' in header:
                self._exptime = float( header['EXPTIME'] )
            else:
                exptimes = {'F184': 901.175,
                            'J129': 302.275,
                            'H158': 302.275,
                            'K213': 901.175,
                            'R062': 161.025,
                            'Y106': 302.275,
                            'Z087': 101.7 }
                if self.band not in exptimes:
                    raise ValueError( f"Can't find exptime for band {self.band}" )
                self._exptime = exptimes[ self.band ]

        else:
            kwmap = { 'sca': ( 'SCA_NUM', int ),
                      'band': ( 'FILTER', lambda x: str(x).strip() ),
                      'mjd': ( 'MJD-OBS', float ),
                      'sky_level': ( 'SKY_MEAN', float )
                     }
            if prop not in kwmap.keys():
                raise RuntimeError( f"Called OpenUniverse2024FITSImage._get_internal_attribute({prop}); "
                                    f"this should never bappen." )
            header = self.get_fits_header()
            setattr( self, f"_{prop}", kwmap[prop][1]( header[ kwmap[prop][0] ] ) )

    def _get_image_shape( self ):
        header = self.get_fits_header()
        self._width = int( header['NAXIS1'] )
        self._height = int( header['NAXIS2'] )

    def _get_ra_dec( self ):
        header = self.get_fits_header()
        self._ra = float( header['RA_TARG'] )
        self._dec = float( header['DEC_TARG'] )

    def _get_corners( self ):
        ny, nx = self.image_shape
        wcs = self.get_wcs()
        self._ra_corner_00, self._dec_corner_00 = wcs.pixel_to_world( 0, 0 )
        self._ra_corner_01, self._dec_corner_01 = wcs.pixel_to_world( 0, ny-1 )
        self._ra_corner_10, self._dec_corner_10 = wcs.pixel_to_world( nx-1, 0 )
        self._ra_corner_11, self._dec_corner_11 = wcs.pixel_to_world( nx-1, ny-1 )

    def get_zeropoint( self, x=None, y=None ):
        if self._zeropoint is None:
            header = self.get_fits_header()
            self._zeropoint = galsim.roman.getBandpasses()[self.band].zeropoint + header['ZPTMAG']
        return self._zeropoint

    def _get_zeropoint_the_hard_way( self, psf, ap_r=9 ):
        """This is here hopefully as legacy code.

        If, however, it turns out that
        galsim.roman.getBandpasses()[self.band].zeropoint +
        header['ZPTMAG'] is not a good enough zeropoint, we may need to
        resort to this.

        """
        raise RuntimeError( "Not up to date." )

        # Get stars from the truth
        truth_colnames = ['object_id', 'ra', 'dec', 'x', 'y', 'realized_flux', 'flux', 'mag', 'obj_type']
        truth_pd = pandas.read_csv(self.truthpath, comment='#', skipinitialspace=True, sep=' ', names=truth_colnames)
        star_tab = Table.from_pandas(truth_pd)
        star_tab['mag'].name = 'mag_truth'
        star_tab['flux'].name = 'flux_truth'
        # Gotta do the FITS vs. C offset
        star_tab['x'] -= 1
        star_tab['y'] -= 1

        star_tab = star_tab[ ( star_tab['obj_type'] == 'star' )
                             & ( star_tab['x'] >= 0 ) & ( star_tab['x'] < self.image_shape[1] )
                             & ( star_tab['y'] >= 0 ) & ( star_tab['y'] < self.image_shape[0] ) ]


        init_params = self.ap_phot( star_tab, ap_r=ap_r )
        # Needs to be 'xcentroid' and 'ycentroid' for PSF photometry.
        init_params['object_id'] = star_tab['object_id'].value
        init_params.rename_column( 'xcenter', 'xcentroid' )
        init_params.rename_column( 'ycenter', 'ycentroid' )
        init_params.rename_column( 'aperture_sum', 'flux_init' )
        final_params = self.psf_phot( init_params, psf, forced_phot=True )

        # Do not need to cross match. Can just merge tables because they
        # will be in the same order.  Remove redundant column flux_init
        final_params.remove_columns( [ 'flux_init'] )
        photres = astropy.table.join(star_tab, init_params, keys=['object_id'])
        photres = astropy.table.join(photres, final_params, keys=['id'])

        # Get the zero point.
        gs_zpt = galsim.roman.getBandpasses()[self.band].zeropoint
        area_eff = galsim.roman.collecting_area
        star_ap_mags = -2.5 * np.log10(photres['flux_init'])
        star_fit_mags = -2.5 * np.log10(photres['flux_fit'])
        star_truth_mags = ( -2.5 * np.log10(photres['flux_truth']) + gs_zpt
                            + 2.5 * np.log10(self.exptime * area_eff) )

        # Eventually, this should be a S/N cut, not a mag cut.
        zpt_mask = np.logical_and(star_truth_mags > 19, star_truth_mags < 21.5)
        zpt = np.nanmedian(star_truth_mags[zpt_mask] - star_fit_mags[zpt_mask])
        _ap_zpt = np.nanmedian(star_truth_mags[zpt_mask] - star_ap_mags[zpt_mask])

        return zpt
