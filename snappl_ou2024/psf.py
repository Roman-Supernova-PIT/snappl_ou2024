__all__ = [ 'ou24PSF_slow', 'ou24PSF', 'ou24PSF_photonshoot', 'ou24PSF_slow_photonshoot' ]

import numbers

import numpy as np

import galsim

from snappl.logger import SNLogger
from snappl.config import Config
from snappl.psf import PSF


class ou24PSF_slow( PSF ):
    """Wrap the roman_imsim PSFs.

    Each time you call get_stamp it will render a new one, with all the
    photon ops and so forth.  This is why it's called "_slow".  Look at
    ou24PSF for something that only does the photonops stuff once.

    (An object of this class will cache, so if you call get_stamp with
    identical arguments it will return the cached version).

    Currently, does not support any oversampling, because SFFT doesn't #
    TODO: support oversampling!

    """

    def __init__( self, sed=None, config_file=None, stamp_size=201,
                   n_photons=1000000, _parent_class=False,  _include_photonOps=False, **kwargs
                 ):

        # ****
        # TEMPORARY
        # We were regularizing arguments.  This is here to catch people who haven't updated.
        # We can take this out when we think everyboyd is caught up.
        if "size" in kwargs:
            raise ValueError( "Don't use the argument size to PSF, use stamp_size." )
        # ****

        super().__init__( _parent_class=True, **kwargs )
        self._consumed_args.update( [ 'sed', 'config_file', 'stamp_size', '_include_photonOps', 'n_photons' ] )
        self._warn_unknown_kwargs( kwargs, _parent_class=_parent_class )

        if ( self._observation_id is None ) or ( self._sca is None ):
            raise ValueError( "Need a observation_id and an sca to make an ou24PSF_slow" )
        if ( stamp_size % 2 == 0 ) or ( int(stamp_size) != stamp_size ):
            raise ValueError( "stamp_size must be an odd integer." )
        stamp_size = int( stamp_size )

        if sed is None:
            SNLogger.warning( "No sed passed to ou24PSF_slow, using a flat SED between 0.1μm and 2.6μm" )
            self.sed = galsim.SED( galsim.LookupTable( [1000, 26000], [1, 1], interpolant='linear' ),
                              wave_type='Angstrom', flux_type='fphotons' )
        elif not isinstance( sed, galsim.SED ):
            raise TypeError( f"sed must be a galsim.SED, not a {type(sed)}" )
        else:
            self.sed = sed

        if config_file is None:
            config_file = Config.get().value( 'system.ou24.config_file' )
        self.config_file = config_file
        self._stamp_size = stamp_size
        self.sca_size = 4088
        self._x = self.sca_size // 2 if self._x is None else self._x
        self._y = self.sca_size // 2 if self._y is None else self._y
        self._include_photonOps = _include_photonOps
        self.n_photons = n_photons
        self._stamps = {}


    @property
    def stamp_size( self ):
        return self._stamp_size


    def get_stamp( self, x=None, y=None, x0=None, y0=None, flux=1., seed=None ):
        """Return a 2d numpy image of the PSF at the image resolution.

        Parameters are as in PSF.get_stamp, plus:

        Parameters
        ----------

          seed : int
            A random seed to pass to galsim.BaseDeviate for photonOps.
            NOTE: this is not part of the base PSF interface (at least,
            as of yet), so don't use it in production pipeline code.
            However, it will be useful in tests for purposes of testing
            reproducibility.

          image : snappl.image.Image or None
            The image that the PSF is associated with. This image will be used to
            determine the WCS of the PSF stamp. If None, the WCS will be determined
            using rmutils.getLocalWCS.

        """
        from roman_imsim.utils import roman_utils
        from roman_imsim import ChargeDiff

        SNLogger.debug("Getting ou24PSF_slow stamp at x=%s, y=%s, x0=%s, y0=%s", x, y, x0, y0)

        # If a position is not given, assume the middle of the SCA
        #   (within 1/2 pixel; by default, we want to make x and y
        #   centered on a pixel).
        x = x if x is not None else float( self._x )
        y = y if y is not None else float( self._y )

        xc = int( np.floor( x + 0.5 ) )
        yc = int( np.floor( y + 0.5 ) )
        x0 = xc if x0 is None else x0
        y0 = yc if y0 is None else y0
        if ( not isinstance( x0, numbers.Integral ) ) or ( not isinstance( y0, numbers.Integral ) ):
            raise TypeError( f"x0 and y0 must be integers; got x0 as a {type(x0)} and y0 as a {type(y0)}" )
        stampx = self.stamp_size // 2 + ( x - x0 )
        stampy = self.stamp_size // 2 + ( y - y0 )

        if ( ( stampx < -self.stamp_size ) or ( stampx > 2.*self.stamp_size ) or
             ( stampy < -self.stamp_size ) or ( stampy > 2.*self.stamp_size ) ):
            raise ValueError( f"PSF would be rendered at ({stampx},{stampy}), which is too far off of the "
                              f"edge of a {self.stamp_size}-pixel stamp." )

        SNLogger.debug( f"Initializing ou24PSF_slow with observation_id {self._observation_id} "
                        f"and sca {self._sca}" )
        if (x, y, stampx, stampy) not in self._stamps:
            SNLogger.debug("configfile = " + str(self.config_file))
            rmutils = roman_utils( self.config_file, int(self._observation_id), self._sca )
            if seed is not None:
                rmutils.rng = galsim.BaseDeviate( seed )

            # It seems that galsim.ChromaticObject.drawImage won't function without stamp having
            # a wcs.  Without a WCS, the stamp was coming out all zeros.
            # TODO : does rmutils.getLocalWCS want 1-indexed or 0-indexed coordinates???
            # wcs = rmutils.getLocalWCS( x+1, y+1 )self._

            if self._image is None:
                self._wcs = rmutils.getLocalWCS( x+1, y+1 )
                SNLogger.debug("No image passed to ou24PSF; using rmutils.getLocalWCS.")
            else:
                image_wcs = self._image.get_wcs()
                if image_wcs is None:
                    SNLogger.warning( f"The image passed to {self.__class__.__name__}"
                    " has no WCS; using rmutils.getLocalWCS." )
                    self._wcs = rmutils.getLocalWCS( x+1, y+1 )
                else:
                    self._wcs = image_wcs.get_galsim_wcs().local( image_pos = galsim.PositionD(x+1, y+1 ))

            stamp = galsim.Image( self.stamp_size, self.stamp_size, wcs=self._wcs )
            point = ( galsim.DeltaFunction() * self.sed ).withFlux( 1, rmutils.bpass )
            # TODO : make sure that rmutils.getPSF wants 1-indexed positions (which we assume here).
            # (This is not that big a deal, because the PSF is not going to vary significantly
            # over 1 pixel.)
            photon_ops = [ rmutils.getPSF( x+1, y+1, pupil_bin=8 ) ]
            if self._include_photonOps:
                photon_ops += rmutils.photon_ops
                photon_ops += [ ChargeDiff(rng=rmutils.rng) ]

            # Note the +1s in galsim.PositionD below; galsim uses 1-indexed pixel positions,
            # whereas snappl uses 0-indexed pixel positions
            center = galsim.PositionD(stampx+1, stampy+1)
            # Note: self._include_photonOps is a bool that states whether we are
            #  shooting photons or not, photon_ops is the actual map (not sure
            #  if that's the correct word) that describes where the photons
            # should be shot, with some randomness.

            # Note from Cole, it seems like the photon ops method is achromatic, but the other method is using a
            # chromatic object. I am not currently sure if this matters.

            if self._include_photonOps:
                point.drawImage(rmutils.bpass, method='phot', rng=rmutils.rng, photon_ops=photon_ops,
                                n_photons=self.n_photons, maxN=self.n_photons, poisson_flux=False,
                                center=center, use_true_center=True, image=stamp)

            else:
                psf = galsim.Convolve(point, photon_ops[0])
                psf.drawImage(rmutils.bpass, method="auto", center=center,
                              use_true_center=True, image=stamp, wcs=self._wcs)

            self._stamps[(x, y, stampx, stampy)] = stamp.array

        return self._stamps[(x, y, stampx, stampy)] * flux


# TODO : make a ou24PSF that makes an image and caches... when things are working better
class ou24PSF( ou24PSF_slow ):
    """Wrap the roman_imsim PSFs, only more efficiently (we hope) than ou24PSF_slow.

    TODO: document what is different, what is cached.

    """

    def __init__(self, _parent_class=False, **kwargs):
        super().__init__(_parent_class=True, **kwargs)
        self._warn_unknown_kwargs( kwargs, _parent_class=_parent_class )
        self._psf = None

    def _init_psf_object( self, x0=None, y0=None, flux=1.):
        """Create the galsim PSF object, WCS, and galsim.chromatic.SimpleChromaticTransformation
           that can be reused for multiple calls to get_stamp.

        Parameters are as in PSF.get_stamp, plus:

        Parameters
        ----------

        seed : int
            A random seed to pass to galsim.BaseDeviate for photonOps.
            NOTE: this is not part of the base PSF interface (at least,
            as of yet), so don't use it in production pipeline code.
            However, it will be useful in tests for purposes of testing
            reproducibility.


        """
        from roman_imsim.utils import roman_utils

        SNLogger.debug(f"Initializing ou24PSF PSF object with {self._observation_id} and "
                       f"sca {self._sca} at x0={x0}, y0={y0}")
        self._rmutils = roman_utils(self.config_file, int(self._observation_id), self._sca)
        self._psf = self._rmutils.getPSF(x0+1, y0+1, pupil_bin=8)
        # TODO : does rmutils.getLocalWCS want 1-indexed or 0-indexed coordinates???
        if self._image is None:
            SNLogger.debug("No image passed to ou24PSF; using rmutils.getLocalWCS.")
            self._wcs = self._rmutils.getLocalWCS( x0+1, y0+1 )

        else:
            image_wcs = self._image.get_wcs()
            if image_wcs is None:
                SNLogger.warning( f"The image passed to {self.__class__.__name__}"
                " has no WCS; using rmutils.getLocalWCS." )
                self._wcs = self._rmutils.getLocalWCS( x0+1, y0+1 )
            else:
                SNLogger.debug(f"Using the WCS from the image passed to {self.__class__.__name__}.")
                self._wcs = image_wcs.get_galsim_wcs().local( image_pos = galsim.PositionD(x0+1, y0+1 ))
        self._stamp = galsim.Image( self.stamp_size, self.stamp_size, wcs=self._wcs )
        self._point = ( galsim.DeltaFunction() * self.sed ).withFlux( 1, self._rmutils.bpass )
        self._convolved_psf = galsim.Convolve(self._point, self._psf)
        # This is only used to ensure the user isn't trying to move the PSF around
        self._stored_x0 = x0
        self._stored_y0 = y0

    def get_stamp(self, x=None, y=None, x0=None, y0=None, flux=1.0, seed=None, image=None):
        """Return a 2d numpy image of the PSF at the image resolution.
        Parameters are as in PSF.get_stamp, plus:

        Parameters
        ----------
          wcs : BaseWCS or galsim.BaseWCS
            WARNING: DO NOT USE. Not part of a standard interface, for testing purposes only.
            An alternative WCS to use for the stamp.

          seed : int
            A random seed to pass to galsim.BaseDeviate for photonOps.
            NOTE: this is not part of the base PSF interface (at least,
            as of yet), so don't use it in production pipeline code.
            However, it will be useful in tests for purposes of testing
            reproducibility.

          image : snappl.image.Image or None
            The image that the PSF is associated with. This image will be used to
            determine the WCS of the PSF stamp. If None, the WCS will be determined
            using rmutils.getLocalWCS.
        """

        from roman_imsim import ChargeDiff

        # If a position is not given, assume the middle of the SCA
        #   (within 1/2 pixel; by default, we want to make x and y
        #   centered on a pixel).

        x = x if x is not None else float( self._x )
        y = y if y is not None else float( self._y )

        xc = int( np.floor( x + 0.5 ) )
        yc = int( np.floor( y + 0.5 ) )
        x0 = xc if x0 is None else x0
        y0 = yc if y0 is None else y0

        if ( not isinstance( x0, numbers.Integral ) ) or ( not isinstance( y0, numbers.Integral ) ):
            raise TypeError( f"x0 and y0 must be integers; got x0 as a {type(x0)} and y0 as a {type(y0)}" )

        if self._psf is None:
            SNLogger.debug( "Initializing ou24PSF galsim PSF object." )
            # If we don't have a psf object, then we need to initialize it, we then re use it for multiple calls to
            # get_stamp.
            self._init_psf_object( x0=x0, y0=y0, flux=flux )
        else:
            if x0 != self._stored_x0 or y0 != self._stored_y0:
                raise ValueError("ou24PSF.get_stamp called with x0 or y0 that does not match the x0 or y0 used"
                                 "to initialize the PSF object. If you want to recreate the PSF object, use "
                                 "ou24PSF_slow instead.")

        stampx = self.stamp_size // 2 + ( x - x0 )
        stampy = self.stamp_size // 2 + ( y - y0 )

        if ( ( stampx < -self.stamp_size ) or ( stampx > 2.*self.stamp_size ) or
             ( stampy < -self.stamp_size ) or ( stampy > 2.*self.stamp_size ) ):
            raise ValueError( f"PSF would be rendered at ({stampx},{stampy}), which is too far off of the "
                              f"edge of a {self.stamp_size}-pixel stamp." )

        if (x, y, stampx, stampy) not in self._stamps:

            if seed is not None:
                self._rmutils.rng = galsim.BaseDeviate( seed )

            photon_ops = [ self._psf ]
            if self._include_photonOps:
                photon_ops += self._rmutils.photon_ops
                photon_ops += [ ChargeDiff(rng=self._rmutils.rng) ]

            # Note the +1s in galsim.PositionD below; galsim uses 1-indexed pixel positions,
            # whereas snappl uses 0-indexed pixel positions
            center = galsim.PositionD(stampx+1, stampy+1)
            # Note: self.include_photonOps is a bool that states whether we are
            #  shooting photons or not, photon_ops is the actual map (not sure
            #  if that's the correct word) that describes where the photons
            # should be shot, with some randomness.
            if self._include_photonOps:
                self._point.drawImage(self._rmutils.bpass, method='phot', rng=self._rmutils.rng, photon_ops=photon_ops,
                                      n_photons=self.n_photons, maxN=self.n_photons, poisson_flux=False,
                                      center=center, use_true_center=True, image=self._stamp)

            else:
                self._convolved_psf.drawImage(self._rmutils.bpass, method="auto", center=center,
                                              use_true_center=True, image=self._stamp, wcs=self._wcs)

            self._stamps[(x, y, stampx, stampy)] = self._stamp.array

        return self._stamps[(x, y, stampx, stampy)] * flux


class ou24PSF_photonshoot( ou24PSF ):
    """ The ou24 PSF but with photon shooting turned on."""

    def __init__(self, _parent_class=False, **kwargs):
        super().__init__(_parent_class=True, _include_photonOps = True, **kwargs)


class ou24PSF_slow_photonshoot( ou24PSF_slow ):
    """ The ou24 slow PSF but with photon shooting turned on."""

    def __init__(self, _parent_class=False, **kwargs):
        super().__init__(_parent_class=True, _include_photonOps = True, **kwargs)

# class ou24PSF( OversampledImagePSF ):
#     """An OversampledImagePSF that renders its internally stored image from a galsim roman_imsim PSF.

#     Use this just like you use an OversampledImagePSF.  However, to
#     construct one, you need to give it a observation_id (pointing) and
#     an SCA from the OpenUniverse2024 sims.  It will only work if all
#     that OU2024 data is available on disk.

#     """

#     def __init__( self, x=2044., y=2044., oversample_factor=5, oversampled_size=201,
#                   observation_id=None, sca=None, sed=None, config_file=None,
#                   include_photonOps=True, n_photons=1000000, seed=None,
#                   **kwargs ):
#         """Construct an ou24PSF.

#         Will render an image oversampled by oversample_factor and save
#         it internally.  Thereafter, get_stamp will just interpolate and
#         resample this image.  This should be faster than re-rendering a
#         galsim PSF every time.

#         Parameters
#         ----------
#           x, y : float
#             Position on the SCA where to evalute the PSF.  Will use (2044, 2044) if not passed.

#           oversample_factor: int (or float?), default 5
#             The once-generated, interally-stored PSF image will be
#             oversampled by this factor.  You probably want this to be an
#             odd integer so that the center of the PSF is not ambiguous.
#             TODO: experiment with different oversample_factors to figure
#             out what the smallest oversampling we can get away with is.

#           oversampled_size: int, default 201
#             The size of a stamp in image pixels on the image for which
#             this is the PSF.  Must be an odd integer.  The stamp you get
#             from get_stamp will have size
#             floor(oversampled_size/oversample_factor), though the size
#             will be increated by one if it would come out to an even
#             number.  (So get_stamp will always return a stamp with an
#             odd side length.)

#             The default of oversampled_size=201 and oversample_factor=5
#             will yield a 41-pixel stamp from get_stamp (since 201/5 =
#             40.2, the floor of which is 40, which is even, so 1 is added
#             to make it an odd 41).

#             (You can read a psf object's stamp_size property to figure
#             out what size of a stamp you'll get when you run
#             get_stamp().)

#           observation_id: str
#             Required.  int(observation_id) is the OpenUniverse2024 pointing.

#           sca: int
#             Required.  The SCA.

#           sed: galsim.SED
#             The SED to render the PSF for.  If not given, will use a flat SED.

#           config_file: str or Path
#             The OU2024 config file that tells it where to find all of
#             its images and so forth.  Usually you don't want to pass
#             this, in which case it will use the ou24psf.config_file
#             config value.

#           include_photonOps: bool, default True
#             TODO

#           n_photons: int, default 1000000
#             Number of photons with photon ops

#           seed: int, default None
#             If given, use this random seed when generating the
#             internally stored oversampled psf image.  Usually you
#             probably want this to be None (and if you don't leave it at
#             None, you may be repeating an error that was made in the
#             OU2024 simulations...), but pass an integer for tests if you
#             need precise reproducibility.

#         """
#         super().__init__( x=x, y=y, oversample_factor=oversample_factor, **kwargs )
#         self._warn_unknown_kwargs( kwargs )

#         if self._data is not None:
#             raise ValueError( "Error, do not pass data when constructing an ou24PSF" )

#         if ( observation_id is None ) or ( sca is None ):
#             raise ValueError( "Need a observation_id and an sca to make an ou24PSF" )
#         if ( oversampled_size % 2 == 0 ) or ( int(oversampled_size) != oversampled_size ):
#             raise ValueError( "Size must be an odd integer." )
#         oversampled_size = int( oversampled_size )

#         if sed is None:
#             SNLogger.warning( "No sed passed to ou24PSF, using a flat SED between 0.1μm and 2.6μm" )
#             self.sed = galsim.SED( galsim.LookupTable( [1000, 26000], [1, 1], interpolant='linear' ),
#                               wave_type='Angstrom', flux_type='fphotons' )
#         elif not isinstance( sed, galsim.SED ):
#             raise TypeError( f"sed must be a galsim.SED, not a {type(sed)}" )
#         else:
#             self.sed = sed

#         if config_file is None:
#             config_file = Config.get().value( 'system.ou24.config_file' )
#         self.config_file = config_file
#         self.observation_id = observation_id
#         self.sca = sca
#         self.oversampled_size = oversampled_size
#         self.sca_size = 4088
#         self.include_photonOps = include_photonOps
#         self.n_photons = n_photons
#         self.seed = seed

#     @property
#     def oversampled_data( self ):
#         if self._data is None:
#             # Render the oversampled PSF
#             x = self._x
#             y = self._y
#             stampx = self.oversampled_size // 2
#             stampy = self.oversampled_size // 2

#             rmutils = roman_utils( self.config_file, int(self.observation_id), self.sca )
#             if self.seed is not None:
#                 rmutils.rng = galsim.BaseDeviate( self.seed )
#             wcs = rmutils.getLocalWCS( x+1, y+1 )
#             wcs = galsim.JacobianWCS(dudx=wcs.dudx / self.oversample_factor,
#                                      dudy=wcs.dudy / self.oversample_factor,
#                                      dvdx=wcs.dvdx / self.oversample_factor,
#                                      dvdy=wcs.dvdy / self.oversample_factor)
#             stamp = galsim.Image( self.oversampled_size, self.oversampled_size, wcs=wcs )
#             point = ( galsim.DeltaFunction() * self.sed ).withFlux( 1., rmutils.bpass )
#             photon_ops = [ rmutils.getPSF( x+1, y+1, pupil_bin=8 ) ]
#             if self.include_photonOps:
#                 photon_ops += rmutils.photon_ops
#                 photon_ops += [ ChargeDiff(rng=rmutils.rng) ]

#             point.drawImage( rmutils.bpass, method='phot', rng=rmutils.rng, photon_ops=photon_ops,
#                              n_photons=self.n_photons, maxN=self.n_photons, poisson_flux=False,
#                              center=galsim.PositionD( stampx+1, stampy+1 ), use_true_center=True,
#                              image=stamp )
#             self._data = stamp.array

#         return self._data
