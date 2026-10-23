__all__ = [ 'ImageCollectionOU2024' ]

import pathlib

from snappl.config import Config
from snappl.snappl_http import retry_post
from snappl_ou2024.image import OpenUniverse2024FITSImage


class ImageCollectionOU2024:
    """Collection of OpenUniverse 2024 FITS images."""

    def __init__( self, base_path=None ):
        self._base_path = None if base_path is None else pathlib.Path( base_path )

    @property
    def base_path( self ):
        if self._base_path is None:
            self._base_path = pathlib.Path( Config.get().value( 'system.ou24.images' ) )
        return self._base_path

    def get_image( self, image_id=None, path=None, observation_id=None, band=None, sca=None,
                   base_path=None, dbclient=None ):
        """Return a OpenUniverse2024FITSImage based on specifications.

        If you specify all of path, observation_id, band, and sca, they must
        be consistent.

        If you just give path, then observation_id, band, and sca will be read
        from the header.

        If you give observation_id, band, and sca, but not path, it will use
        get_image_path to find the image.

        NOTE: in the ou2024 files, "observation_id" is called "pointing"
        and is an int.  In snappl, to match what will come from Roman,
        we use observation_id, and make it a string (even though it's
        composed of characters [0-9], it's a string in the Roman
        headers).

        """

        if image_id is not None:
            raise ValueError( "image_id is invalid for ImageCollectionOU2024" )

        base_path = self.base_path if base_path is None else pathlib.Path( base_path )
        if path is not None:
            img = OpenUniverse2024FITSImage( base_path / path, format=-1 )
            if ( observation_id is not None ) and ( int(observation_id) != int(img.observation_id) ):
                raise ValueError( "Observation_Id {observation_id} inconsistent with what's in {path}" )
            if ( sca is not None ) and ( int(sca) != int(img.sca) ):
                raise ValueError( "SCA {sca} inconsistent with what's in {path}" )
            if ( band is not None ) and ( str(band) != str(img.band) ):
                raise ValueError( "Band {band} inconsistent with what's in {path}" )
            return img

        if any( i is None for i in [ observation_id, band, sca ] ):
            raise ValueError( "Must specify either path or all of (observation_id, band, sca)" )

        path = self.get_image_path( observation_id, band, sca, base_path=base_path )
        img = OpenUniverse2024FITSImage( path, observation_id=observation_id, sca=sca, format=-1 )
        return img


    def get_image_path( self, observation_id, band, sca, base_path=None ):
        """Return the absolute path to the desired OU2024 FITS image.

        See ImageCollection.get_image_path for documentation.

        """
        base_path = self.base_path if base_path is None else pathlib.Path( base_path )
        path = ( base_path / band / str(observation_id) /
                 f'Roman_TDS_simple_model_{band}_{str(observation_id)}_{str(sca)}.fits.gz' )
        return path

    def find_images( self,
                     subset=None,
                     path=None,
                     mjd_min=None,
                     mjd_max=None,
                     ra=None,
                     dec=None,
                     band=None,
                     exptime_min=None,
                     exptime_max=None,
                     sca=None ):
        params = {}

        if ( ra is None ) != ( dec is None ):
            raise ValueError( "Specify either both or neither of ra and dec" )

        if ra is not None:
            params['containing'] = ( float(ra), float(dec) )

        if mjd_min is not None:
            params['mjd_min'] = float(mjd_min)
        if mjd_max is not None:
            params['mjd_max'] = float(mjd_max)
        if band is not None:
            params['filter'] = str(band)
        if exptime_min is not None:
            params['exptime_min'] = float(exptime_min)
        if exptime_max is not None:
            params['exptime_max'] = float(exptime_max)

        simdex = Config.get().value( 'system.ou24.simdex_server' )
        res = retry_post( f"{simdex}/findromanimages", json=params ).json()

        images = []
        for i in range( len(res['pointing']) ):
            path = self.get_image_path( res['pointing'][i], res['filter'][i], res['sca'][i] )
            image = OpenUniverse2024FITSImage(path,
                                              band=res['filter'][i],
                                              observation_id=str( res['pointing'][i] ),
                                              sca=res["sca"][i],
                                              mjd=res['mjd'][i],
                                              format=-1 )
            image.mjd = res['mjd'][i]
            images.append( image )


        return images
