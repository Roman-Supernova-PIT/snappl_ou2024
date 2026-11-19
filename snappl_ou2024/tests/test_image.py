import pytest

import numpy as np

import astropy.io

from snappl.psf import PSF


# ======================================================================
# OpenUniverse2024FITSImage tests

def test_ou2024_get_data( ou2024image ):
    try:
        # Notice the types are explicitly called out as big-endian below, even
        #   if the architecture of the system  is little-endian.  This is because
        #   FITS files are defined to be big-endian, and astropy reads them
        #   straight.  (...except that the integer flags don't seem to
        #   be big-endian, so I am confused as to what astropy and/or fitsio
        #   and/or numpy really does.)
        #
        # ... but now that we're using FITSIO, it looks like it byte swaps,
        #   so everything changed below.

        # I choose an area around a nice galaxy to sum
        data, = ou2024image.get_data( 'data' )
        assert isinstance( data, np.ndarray )
        assert data.shape == ( 4088, 4088 )
        # assert data.dtype == ">f8"
        assert data.dtype == np.dtype('float64')
        assert data[2300:2336, 2482:2589].sum() == pytest.approx( 429719.0, rel=1e-5 )

        noise, = ou2024image.get_data( 'noise' )
        assert isinstance( noise, np.ndarray )
        assert noise.shape == ( 4088, 4088 )
        # assert noise.dtype == ">f4"
        assert noise.dtype == np.dtype('float32')
        assert noise[2300:2336, 2482:2589].sum() == pytest.approx( 427641, rel=1e-5 )

        flags, = ou2024image.get_data( 'flags' )
        assert isinstance( flags, np.ndarray )
        assert flags.shape == ( 4088, 4088 )
        assert flags.dtype == "uint32"
        assert flags[2300:2336, 2482:2589].sum() == 0

        assert ou2024image._data is None
        assert ou2024image._noise is None
        assert ou2024image._flags is None

        data2, noise2, flags2 = ou2024image.get_data()
        assert data2 is not data
        assert noise2 is not noise
        assert flags2 is not flags
        assert np.all( data2 == data )
        assert np.all( noise2 == noise )
        assert np.all( flags2 == flags )
        assert ou2024image._data is None
        assert ou2024image._noise is None
        assert ou2024image._flags is None

        props = [ 'data', 'noise', 'flags' ]
        for prop in props:
            res = ou2024image.get_data( prop, cache=True )
            assert isinstance( res, list )
            assert len(res) == 1
            assert isinstance( res[0], np.ndarray )
            assert res[0].shape == ( 4088, 4088 )
            assert res[0] is getattr( ou2024image, f"_{prop}" )
            if prop == 'data':
                assert isinstance( ou2024image._header, astropy.io.fits.header.Header )
            else:
                assert ou2024image._header is None
            for otherprop in props:
                if otherprop == prop:
                    continue
                assert getattr( ou2024image, f"_{otherprop}" ) is None
            setattr( ou2024image, f"_{prop}", None)
            setattr( ou2024image, "_header", None )

        data3, noise3, flags3 = ou2024image.get_data( cache=True )
        assert isinstance( ou2024image._header, astropy.io.fits.header.Header )
        assert np.all( data3 == data )
        assert np.all( noise3 == noise )
        assert np.all( flags3 == flags )
        assert data3 is not data
        assert noise3 is not noise
        assert flags3 is not flags
        assert data3 is ou2024image._data
        assert noise3 is ou2024image._noise
        assert flags3 is ou2024image._flags

        data4, noise4, flags4 = ou2024image.get_data( always_reload=True )
        assert data3 is ou2024image._data
        assert noise3 is ou2024image._noise
        assert flags3 is ou2024image._flags
        assert data4 is not data3
        assert noise4 is not noise3
        assert flags4 is not flags3
        assert np.all( data4 == data )
        assert np.all( noise4 == noise )
        assert np.all( flags4 == flags )
    finally:
        # ...I don't think there's anything needing cleaning up
        # We didn't use the module-scope fixture
        pass


def test_ou2024_properties( ou2024image ):
    im = ou2024image

    assert im.observation_id == '13205'
    assert im.sca == 1
    assert im.ra == pytest.approx( 7.42991, abs=1e-5 )
    assert im.dec == pytest.approx( -44.8697, abs=1e-5 )
    assert im.ra_corner_00 == pytest.approx( 7.49436, abs=1e-4 )
    assert im.ra_corner_01 == pytest.approx( 7.42167, abs=1e-4 )
    assert im.ra_corner_10 == pytest.approx( 7.65462, abs=1e-4 )
    assert im.ra_corner_11 == pytest.approx( 7.58169, abs=1e-4 )
    assert im.dec_corner_00 == pytest.approx( -44.95508, abs=1e-4 )
    assert im.dec_corner_01 == pytest.approx( -44.84399, abs=1e-4 )
    assert im.dec_corner_10 == pytest.approx( -44.90312, abs=1e-4 )
    assert im.dec_corner_11 == pytest.approx( -44.79213, abs=1e-4 )
    assert im.band == 'Y106'
    assert im.mjd == pytest.approx( 62170.424, abs=1e-3 )
    assert im.position_angle == pytest.approx( -24.75, abs=0.01 )
    assert im.exptime == pytest.approx( 302.275, abs=1e-3 )
    assert im.sky_level == pytest.approx( 112.0, abs=0.1 )
    assert im.get_zeropoint() == pytest.approx( 32.6617, abs=1e-4 )

    # Setting updates properties, but not the header

    im.observation_id = '5'
    assert im.observation_id == '5'

    im.sca = 42
    assert im.sca == 42
    assert im.get_fits_header()['SCA_NUM'] == 1

    im.ra = 128.
    assert im.ra == 128.
    assert im.get_fits_header()['RA_TARG'] == pytest.approx( 7.42991, abs=1e-5 )

    im.dec = -13.
    assert im.dec == -13.
    assert im.get_fits_header()['DEC_TARG'] == pytest.approx( -44.8697, abs=1e-5 )

    origband = im.band
    im.band = 'foo'
    assert im.band == 'foo'
    assert im.get_fits_header()['FILTER'].strip() == 'Y106'
    # Gotta reset it or _get_zeropoint below fails
    im._band = origband

    im.mjd = 60000.
    assert im.mjd == 60000.
    assert im.get_fits_header()['MJD-OBS'] == pytest.approx( 62170.424, abs=1e-3 )

    im.position_angle = 66.
    assert im.position_angle == 66.
    im._get_position_angle()
    assert im.position_angle == pytest.approx( -24.75, abs=0.01 )

    im.exptime = 23.
    assert im.exptime == 23.
    assert im.get_fits_header()['EXPTIME'] == pytest.approx( 302.275, abs=1e-3 )

    im.sky_level = 666.
    assert im.sky_level == 666.
    assert im.get_fits_header()['SKY_MEAN'] == pytest.approx( 112.0, abs=0.1 )

    for axis in [ 'ra', 'dec' ]:
        for corner in [ '00', '01', '10', '11' ]:
            orig = getattr( im, f'_{axis}_corner_{corner}' )
            setattr( im, f'{axis}_corner_{corner}', 0. )
            assert getattr( im, f'{axis}_corner_{corner}' ) == 0.
            im._get_corners()
            assert getattr( im, f'{axis}_corner_{corner}' ) == orig


def test_ou2024_get_fits_header( ou2024image, ou2024image_module ):
    assert isinstance( ou2024image_module._header, astropy.io.fits.header.Header )
    assert ou2024image_module._header == ou2024image_module.get_fits_header()
    assert ou2024image._header is None
    hdr = ou2024image.get_fits_header()
    assert isinstance( hdr, astropy.io.fits.header.Header )
    assert hdr is ou2024image._header


@pytest.mark.xfail( reason="Issue #50" )
def test_ou2024_compare_zeropoints( ou2024image ):
    zp1 = ou2024image._get_zeropoint()
    psf = PSF.get_psf_object( 'A25ePSF', band=ou2024image.band, sca=ou2024image.sca, x=1277.5, y=1277.5 )
    zp2 = ou2024image._get_zeropoint_the_hard_way( psf, ap_r=9 )
    assert zp1 == pytest.approx( zp2, abs=0.01 )


@pytest.mark.xfail( reason="OpenUniverse2024FITSImage.get_zeropoint_the_hard_way doesn't currently work" )
def test_ou2024_compare_zeropoints_with_not_enough_precision( ou2024image ):
    zp1 = ou2024image.get_zeropoint()
    assert zp1 is not None
    psf = PSF.get_psf_object( 'A25ePSF', band=ou2024image.band, sca=ou2024image.sca, x=1277.5, y=1277.5 )
    zp2 = ou2024image._get_zeropoint_the_hard_way( psf, ap_r=9 )
    assert zp1 == pytest.approx( zp2, abs=0.1 )
