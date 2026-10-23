import galsim

from snappl.wcs import BaseWCS, AstropyWCS
from snappl_ou2024.wcs import GalsimWCS


def test_galsimwcs( ou2024image, check_wcs ):
    wcs = GalsimWCS.from_header( ou2024image.get_fits_header() )
    assert isinstance( wcs, BaseWCS )
    assert isinstance( wcs, GalsimWCS )
    assert wcs._wcs is None
    assert not wcs._wcs_is_astropy
    assert isinstance( wcs._gsimwcs, galsim.AstropyWCS )
    check_wcs( wcs )

    wcs =  GalsimWCS( gsimwcs = wcs._gsimwcs )
    assert isinstance( wcs, BaseWCS )
    assert isinstance( wcs, GalsimWCS )
    assert wcs._wcs is None
    assert not wcs._wcs_is_astropy
    assert isinstance( wcs._gsimwcs, galsim.AstropyWCS )
    check_wcs( wcs )


def test_get_astropywcs_get_galsimwcs( ou2024image, check_wcs ):
    wcs = ou2024image.get_wcs()
    assert isinstance( wcs, AstropyWCS )
    rawgswcs = wcs.get_galsim_wcs()
    assert isinstance( rawgswcs, galsim.AstropyWCS )
    gswcs = GalsimWCS( rawgswcs )
    check_wcs( gswcs )
