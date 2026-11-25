__all__ = [ 'GalsimWCS' ]

import collections

import numpy as np

import galsim

from snappl.wcs import BaseWCS, AstropyWCS


class GalsimWCS(BaseWCS):
    """A WCS speicifc to Galsim."""

    def __init__( self, gsimwcs=None ):
        super().__init__()
        self._gsimwcs = gsimwcs

    @classmethod
    def from_header( cls, header ):
        """Create a GalsimWCS from a FITS header.

        Does TAN-TPV conversion the same as AstropyWCS.from_header.

        Parameters
        ----------
          header: astropy.io.fits.header.Header
            See AstropyWCS.from_header

        Returns
        -------
          GalsimWCS

        """
        wcs = GalsimWCS()
        wcs._gsimwcs = galsim.AstropyWCS( header=AstropyWCS._fix_wcs_tan_with_pv1_0( header ) )
        return wcs

    def to_fits_header( self ):
        return self._gsimwcs.wcs.to_header( relax=True )

    def get_galsim_wcs( self ):
        return self._gsimwcs

    def pixel_to_world( self, x, y ):
        if isinstance( x, collections.abc.Sequence ) and not isinstance( x, np.ndarray ):
            x = np.array( x )
            y = np.array( y )
        # Galsim WCSes are 1-indexed
        ra, dec = self._gsimwcs.toWorld( x+1, y+1, units='deg' )
        # Decide by dimensionality, not size: a 1-element array is still an array,
        #  and numpy refuses float() on arrays with ndim > 0. This causes
        # campari to break. -Cole
        if np.ndim( x ) == 0:
            ra = float( ra )
            dec = float( dec )
        return ra, dec

    def world_to_pixel( self, ra, dec ):
        if isinstance( ra, collections.abc.Sequence ) and not isinstance( ra, np.ndarray ):
            ra = np.array( ra )
            dec = np.array( dec )
        x, y = self._gsimwcs.toImage( ra, dec, units='deg' )
        # Convert from 1-indexed galsim pixel coordaintes to 0-indexed
        x -= 1
        y -= 1
        if np.ndim( ra ) == 0:
            x = float( x )
            y = float( y )
        return x, y
